"""
Module: hf_benchmark

Provides unified functions to benchmark contrastive vision-language models in real-time
on compositional tasks (color or position), loading data from the Hugging Face Hub.
"""
import re
from itertools import product, permutations
from typing import List, Dict, Tuple
import torch
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

# --- Unified Grammar Rules ---
PLURAL_NOUNS = { "sneakers", "glasses", "gloves", "leather shoes", "boots", "slippers", "sandals", "high heels", "scissors", "chopsticks", "headphones", "pliers", "binoculars", "earphone" }
VOWEL_LETTERS = "aeiou"

def load_hf_dataset(hf_repo_id: str, num_objects: int):
    """Loads and filters a benchmark dataset from the Hugging Face Hub."""
    logger.info(f"Loading dataset for N={num_objects} from HF Hub: {hf_repo_id}...")
    try:
        hf_dataset = load_dataset(hf_repo_id, split="train")
        filtered_dataset = hf_dataset.filter(lambda example: example.get('num_objects') == num_objects)
        logger.info(f"Loaded and filtered {len(filtered_dataset)} entries.")
        return list(filtered_dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset '{hf_repo_id}': {e}")
        return []

def _substitute_caption(orig_caption: str, task_type: str, orig_elements: Dict, new_elements: Dict) -> str:
    """Robustly substitutes elements, handling grammar for both tasks."""
    caption = orig_caption
    
    # Use placeholder substitution for robustness
    placeholders = {elem.lower(): f"__PLACEHOLDER_{i}__" for i, elem in enumerate(orig_elements)}
    
    sorted_orig = sorted(placeholders.keys(), key=len, reverse=True)
    for elem in sorted_orig:
        caption = re.sub(rf"\b{re.escape(elem)}\b", placeholders[elem], caption, flags=re.IGNORECASE)

    for i, elem in enumerate(orig_elements):
        caption = caption.replace(placeholders[elem.lower()], new_elements[i].lower())
        
    # Apply task-specific grammar fixes after substitution
    if task_type == 'color':
        # Re-evaluate articles based on new colors
        def get_article(obj, color):
            if obj.lower() in PLURAL_NOUNS: return "a pair of"
            return "an" if color.lower().startswith(tuple(VOWEL_LETTERS)) else "a"
            
        # This is a simplified regex pass for fixing articles after substitution
        words = caption.split()
        for i, word in enumerate(words):
            if word.lower() in ['a', 'an'] and i + 2 < len(words):
                obj_name = words[i+2].strip(".,?!")
                color_name = words[i+1].strip(".,?!")
                correct_article = get_article(obj_name, color_name)
                # This logic is complex to generalize perfectly with regex,
                # but for template-based captions, this is a strong heuristic.
                if (word.lower() == 'a' and correct_article == 'an') or \
                   (word.lower() == 'an' and correct_article == 'a'):
                    words[i] = correct_article
        caption = " ".join(words)
        
    return caption.capitalize()

def _generate_swap_negatives(item: Dict, task_type: str) -> List[str]:
    """Generates all negative captions for the Swap Benchmark."""
    caption = item.get('handmade_caption') or item.get('llm_caption')
    objects = item['objects']
    attributes = item.get('relations') or item.get('colors')
    
    if task_type == 'color':
        attr_perms = list(permutations(attributes))
        attr_perms.remove(tuple(attributes))
        return [_substitute_caption(caption, 'color', attributes, list(p)) for p in attr_perms]
    else: # position
        obj_perms = list(permutations(objects))
        obj_perms.remove(tuple(objects))
        return [_substitute_caption(caption, 'position', objects, list(p)) for p in obj_perms]

def _generate_confusion_candidates(item: Dict) -> Tuple[List[str], int]:
    """Generates all candidate captions for the Confusion Benchmark."""
    caption = item.get('handmade_caption') or item.get('llm_caption')
    objects = item['objects']
    attributes = item.get('relations') or item.get('colors')
    
    obj_combos = list(product(objects, repeat=len(objects)))
    attr_combos = list(product(attributes, repeat=len(attributes)))
    all_configs = list(product(obj_combos, attr_combos))
    
    positive_config = (tuple(objects), tuple(attributes))
    try:
        positive_label = all_configs.index(positive_config)
    except ValueError:
        return [], -1
        
    # Note: A fully robust substitute function for confusion is very complex.
    # The original scripts' logic is preserved here as a strong approximation.
    candidates = []
    for p_obj, p_attr in all_configs:
        # A simplified substitution for generating all variants
        temp_caption = caption
        for orig_o, new_o in zip(objects, p_obj):
            temp_caption = re.sub(rf"\b{orig_o}\b", new_o, temp_caption, flags=re.IGNORECASE)
        for orig_a, new_a in zip(attributes, p_attr):
            temp_caption = re.sub(rf"\b{orig_a}\b", new_a, temp_caption, flags=re.IGNORECASE)
        candidates.append(temp_caption)
        
    return candidates, positive_label

def run_single_benchmark(benchmark_name: str, data: list, model, batch_size: int, caption_source: str, task_type: str):
    """Runs a single benchmark (Swap or Confusion) and returns accuracy."""
    logger.info(f"--- Starting {benchmark_name} Benchmark ({task_type}, {caption_source}) ---")
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        images = [entry[f"{caption_source}_image"] for entry in batch if entry.get(f"{caption_source}_image")]
        if not images: 
            images = [entry["image"] for entry in batch if entry.get("image")]
        
        img_embs = model.compute_image_embeddings(images)
        
        all_candidate_captions = []
        entry_infos = []
        
        for entry in batch:
            try:
                positive_caption = entry[f"{caption_source}_caption"]
            except Exception as e:
                positive_caption = entry["caption"]
            if benchmark_name == 'Swap':
                negatives = _generate_swap_negatives(entry, task_type)
                candidates = [positive_caption] + negatives
                label = 0
            else: # Confusion
                candidates, label = _generate_confusion_candidates(entry)
            
            all_candidate_captions.extend(candidates)
            entry_infos.append({'num_candidates': len(candidates), 'label': label})
            
        txt_embs = model.compute_text_embeddings(all_candidate_captions)
        
        text_offset = 0
        for j, info in enumerate(entry_infos):
            num_cands = info['num_candidates']
            img_emb = img_embs[j]
            candidate_embs = txt_embs[text_offset : text_offset + num_cands]
            
            sims = (img_emb @ candidate_embs.T).squeeze(0)
            pred_idx = sims.argmax().item()
            
            results.append(pred_idx == info['label'])
            text_offset += num_cands
            
    accuracy = sum(results) / len(results) if results else 0
    logger.info(f"{benchmark_name} Done: {sum(results)}/{len(results)} correct ({accuracy:.2%})")
    return accuracy

def run_benchmarks(data: list, model, batch_size: int, caption_source: str, task_type: str, log_wandb: bool):
    """Orchestrator to run all benchmarks for a given configuration."""
    swap_acc = run_single_benchmark('Swap', data, model, batch_size, caption_source, task_type)
    conf_acc = run_single_benchmark('Confusion', data, model, batch_size, caption_source, task_type)
    
    if log_wandb:
        try:
            import wandb
            wandb.log({
                f"{caption_source}/{task_type}/swap_accuracy": swap_acc,
                f"{caption_source}/{task_type}/confusion_accuracy": conf_acc,
            })
        except ImportError:
            logger.warning("wandb not installed. Skipping logging.")
            
    return {'swap_accuracy': swap_acc, 'confusion_accuracy': conf_acc}