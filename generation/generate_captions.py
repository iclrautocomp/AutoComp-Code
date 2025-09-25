import argparse
import random
import json
import os
import re
from typing import List, Dict, Tuple

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import logging

# --- Configuration ---
DEFAULT_GEMMA_MODEL_ID = "google/gemma-3-12b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CaptionGenerator:
    """
    A unified class to generate grammatically correct captions for both color-binding
    and position-binding compositional tasks.
    """
    def __init__(self, task_type: str, model_name: str = DEFAULT_GEMMA_MODEL_ID, max_regeneration_attempts: int = 3):
        if task_type not in ['color', 'position']:
            raise ValueError("task_type must be either 'color' or 'position'")

        self.task_type = task_type
        self.objects = self._load_objects()
        
        # Nouns that require "a pair of"
        self.plural_nouns = {
            "Sneakers", "Glasses", "Gloves", "Leather Shoes", "Boots", "Slippers",
            "Sandals", "High Heels", "Scissors", "Chopsticks", "Headphones", "Pliers",
            "Binoculars", "Earphone"
        }
        
        if self.task_type == 'color':
            self.attributes = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "silver", "gold", "teal", "maroon", "navy", "olive", "lime", "cyan", "magenta"]
        else: # position
            self.attributes = ["under", "above", "to the left of", "to the right of"]

        self.max_regeneration_attempts = max_regeneration_attempts
        self._load_model_and_processor(model_name)

    def _load_model_and_processor(self, model_name: str):
        """Loads the LLM and its processor."""
        logger.info(f"Loading model '{model_name}' on {DEVICE} with dtype {TORCH_DTYPE}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=TORCH_DTYPE,
                device_map="auto"
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_name)
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
            logger.info("Model and processor loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
            raise

    def _load_objects(self) -> List[str]:
        """Loads the comprehensive object list."""
        return [
            "Sneakers", "Chair", "Hat", "Car", "Lamp", "Glasses", "Bottle", "Desk", "Cup", "Cabinet", "Shelf",
            "Handbag", "Bracelet", "Plate", "Picture", "Frame", "Helmet", "Book", "Gloves", "Storage Box",
            "Boat", "Leather Shoes", "Bench", "Potted Plant", "Bowl", "Basin", "Flag", "Pillow", "Boots",
            "Vase", "Microphone", "Necklace", "Ring", "SUV", "Wine Glass", "Belt", "Monitor", "Backpack",
            "Umbrella", "Speaker", "Watch", "Tie", "Trash Bin", "Slippers", "Bicycle", "Stool", "Barrel",
            "Bucket", "Van", "Couch", "Sandals", "Basket", "Drum", "Pen", "Pencil", "Bus", "High Heels",
            "Motorcycle", "Guitar", "Carpet", "Cell Phone", "Camera", "Truck", "Traffic Cone", "Cymbal",
            "Lifesaver", "Towel", "Toy", "Candle", "Sailboat", "Laptop", "Bed", "Faucet", "Tent", "Mirror",
            "Sink", "Knife", "Paddle", "Pickup Truck", "Fork", "Traffic Sign", "Balloon", "Tripod", "Spoon",
            "Clock", "Pot", "Cake", "Dining Table", "Hanger", "Napkin", "Keyboard", "Lantern", "Fan",
            "Baseball Glove", "Airplane", "PC Mouse", "Train", "Luggage", "Nightstand", "Tea Pot", "Telephone",
            "Trolley", "Headphones", "Sports Car", "Dessert", "Scooter", "Stroller", "Crane", "Remote",
            "Refrigerator", "Oven", "Baseball Bat", "Jug", "Piano", "Pizza", "Skateboard", "Surfboard",
            "Gas Stove", "Donut", "Bow Tie", "Toilet", "Kite", "Shovel", "Computer Case", "Toilet Paper",
            "Cleaning Products", "Chopsticks", "Microwave", "Baseball", "Cutting Board", "Coffee Table",
            "Side Table", "Scissors", "Marker", "Ladder", "Snowboard", "Cookies", "Radiator", "Fire Hydrant",
            "Fire Extinguisher", "Fire Truck", "Billiards", "Converter", "Bathtub", "Wheelchair", "Briefcase",
            "Paint Brush", "Heavy Truck", "Extractor Fan", "Extension Cord", "Tennis Racket", "Folder",
            "Earphone", "Mask", "Kettle", "Ship", "Swing", "Coffee Machine", "Slide", "Carriage", "Projector",
            "Frisbee", "Washing Machine", "Dryer", "Printer", "Saxophone", "Tissue Box", "Toothbrush",
            "Ice Cream", "Hot Air Balloon", "Cello", "Scale", "Trophy", "Blender", "Wallet", "Purse",
            "Tape", "Tablet", "Cosmetics", "Trumpet", "Golf Ball", "Parking Meter", "Key", "Hurdle",
            "Fishing Rod", "Medal", "Flute", "Brush", "Megaphone", "Broom", "Trombone", "Router", "Modem",
            "Poker Card", "Toaster", "Notepaper", "Pliers", "CD", "DVD", "Hammer", "Flask", "Screwdriver",
            "Soap", "Recorder", "Board Eraser", "Tape Measure", "Ruler", "Showerhead", "Globe", "Chips",
            "Stapler", "Formula 1 Car", "Dishwasher", "Hoverboard", "Rice Cooker", "Tuba", "Calculator",
            "Electric Drill", "Hair Dryer", "Egg Tart", "Treadmill", "Lighter", "Game Board", "Mop",
            "Target", "Pencil Case", "Binoculars", "Barbell", "Noodles", "Comb", "Dumpling", "Chainsaw",
            "Eraser", "Lipstick", "Cosmetics Mirror"
        ]

    def _get_article(self, obj: str, next_word: str) -> str:
        """Determines the correct article ('a', 'an', or 'a pair of') for an object."""
        if obj in self.plural_nouns:
            return "a pair of"
        return "an" if next_word.lower().startswith(('a', 'e', 'i', 'o', 'u')) else "a"

    def _generate_color_handmade(self, objects_with_attrs: List[Tuple[str, str]]) -> str:
        """Generates a grammatically correct handmade caption for the color binding task."""
        caption_parts = []
        for obj, color in objects_with_attrs:
            article = self._get_article(obj, color)
            caption_parts.append(f"{article} {color} {obj.lower()}")

        if len(caption_parts) == 1:
            return f"{caption_parts[0]} on a white background."
        else:
            last_part = caption_parts.pop()
            return ", ".join(caption_parts) + f", and {last_part} on a white background."

    def _generate_position_handmade(self, objects: List[str], relations: List[str]) -> str:
        """Generates a grammatically correct handmade caption for the position binding task."""
        # For position, the article choice depends on the object name itself.
        article1 = self._get_article(objects[0], objects[0])
        article2 = self._get_article(objects[1], objects[1])
        
        caption = f"{article1.capitalize()} {objects[0].lower()} is {relations[0]} {article2} {objects[1].lower()}"
        
        for i in range(1, len(relations)):
            next_obj = objects[i+1]
            article_next = self._get_article(next_obj, next_obj)
            caption += f", which is {relations[i]} {article_next} {next_obj.lower()}"
            
        caption += " on a white background."
        return caption

    def _is_llm_output_valid(self, caption: str, objects: List[str], attributes: List[str]) -> bool:
        """Routes to the correct validation logic based on task type."""
        if not caption: return False
        caption_lower = caption.lower()

        if self.task_type == 'color':
            for obj, color in zip(objects, attributes):
                obj_l, color_l = obj.lower(), color.lower()
                if not re.search(rf"\b{re.escape(color_l)}\s+([a-zA-Z]+\s+)?{re.escape(obj_l)}\b", caption_lower):
                    return False
            return True
        else: # position
            for i in range(len(attributes)): # attributes are relations here
                obj1_l, obj2_l, rel_l = objects[i].lower(), objects[i+1].lower(), attributes[i].lower()
                if not re.search(rf"\b{re.escape(obj1_l)}\b.*?\b{re.escape(rel_l)}\b.*?\b{re.escape(obj2_l)}\b", caption_lower):
                    return False
            return True

    def _prepare_llm_prompt(self, objects: List[str], attributes: List[str]) -> List[Dict]:
        """Prepares the chat prompt for the LLM based on the task type."""
        if self.task_type == 'color':
            obj_attr_pairs = [f"'{c} {o.lower()}'" for o, c in zip(objects, attributes)]
            obj_attr_text = ", ".join(obj_attr_pairs)
            sys_prompt = "You are an expert image caption writer. Your task is to generate a single, natural-sounding sentence accurately describing a scene with specific objects and their colors. You MUST include ALL specified objects and their colors. Generate ONLY the caption text."
            user_prompt = f"""Rewrite the following concept into a natural, single-sentence image caption: {obj_attr_text}.
Place the objects in a realistic context.
Example: for 'a green chair' and 'a yellow lamp', a good caption is: 'A green chair stands beside a yellow lamp in a brightly lit room.'
Caption:"""
        else: # position
            chain_desc = f"'{objects[0].lower()} {attributes[0]} {objects[1].lower()}'"
            for i in range(1, len(attributes)):
                chain_desc += f", which is then '{attributes[i]} {objects[i+1].lower()}'"
            sys_prompt = "You are an expert image caption writer. Your task is to generate a single, natural-sounding sentence that accurately describes a scene with multiple objects in a specific chained spatial relationship. Generate ONLY the caption text."
            user_prompt = f"""Rewrite the following chained relationship into a natural, single-sentence image caption: {chain_desc}.
Place the objects in a realistic context.
Example: for 'chair beside table, which is near window', a good caption is: 'A wooden chair is set beside a small table, which is positioned near a large, sunlit window.'
Caption:"""

        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

    def _generate_with_llm(self, chat_prompt: List[Dict]) -> str:
        """Generates a caption using the loaded LLM."""
        try:
            inputs = self.processor.apply_chat_template(
                chat_prompt, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device)
            prompt_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
            generated_text = self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            return ""

    def generate_captions(self, num_objects: int, num_captions: int, output_path: str):
        """Main loop to generate and save captions."""
        if self.task_type == 'position' and num_objects < 2:
            raise ValueError("Position task requires at least 2 objects.")
        results = []
        for i in tqdm(range(num_captions), desc=f"Generating {self.task_type.capitalize()} Captions (N={num_objects})"):
            sampled_objects = random.sample(self.objects, num_objects)
            num_attributes_needed = num_objects if self.task_type == 'color' else num_objects - 1
            sampled_attributes = random.sample(self.attributes, num_attributes_needed)
            if self.task_type == 'color':
                objects_with_attrs = list(zip(sampled_objects, sampled_attributes))
                handmade_caption = self._generate_color_handmade(objects_with_attrs)
            else: # position
                handmade_caption = self._generate_position_handmade(sampled_objects, sampled_attributes)
            llm_caption = ""
            for _ in range(self.max_regeneration_attempts):
                prompt = self._prepare_llm_prompt(sampled_objects, sampled_attributes)
                candidate_caption = self._generate_with_llm(prompt)
                if self._is_llm_output_valid(candidate_caption, sampled_objects, sampled_attributes):
                    llm_caption = candidate_caption
                    break
            if not llm_caption:
                logger.warning(f"Failed to generate valid LLM caption for concept {i+1}. Skipping.")
                continue
            results.append({ "id": len(results) + 1, "task_type": self.task_type, "num_objects": num_objects, "objects": sampled_objects, "attributes": sampled_attributes, "handmade_caption": handmade_caption, "llm_caption": llm_caption, })
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully generated and saved {len(results)} captions to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate compositional captions for VLM benchmarks.")
    parser.add_argument("--task_type", type=str, required=True, choices=['color', 'position'], help="The type of compositional task.")
    parser.add_argument("--num_objects", type=int, required=True, help="The number of objects per caption.")
    parser.add_argument("--num_captions", type=int, default=1000, help="The total number of captions to generate.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()
    if not args.output_file:
        args.output_file = f"generated_captions_{args.task_type}_n{args.num_objects}.json"
    generator = CaptionGenerator(task_type=args.task_type)
    generator.generate_captions(num_objects=args.num_objects, num_captions=args.num_captions, output_path=args.output_file)

if __name__ == "__main__":
    main()