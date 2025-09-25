import argparse
import json
import os
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import logging
import time

# --- Configuration ---
VQA_MODEL_ID = "google/gemma-3-12b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
MAX_VQA_ATTEMPTS = 3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttributeValidator:
    """
    Validates attributes (color or relations) in images using a VQA model.
    """
    def __init__(self, task_type: str, model_name: str = VQA_MODEL_ID):
        if task_type not in ['color', 'position']:
            raise ValueError("task_type must be 'color' or 'position'")
        self.task_type = task_type
        
        # Define attribute vocabularies
        self.color_vocab = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "silver", "gold", "teal", "maroon", "navy", "olive", "lime", "cyan", "magenta"]
        self.relation_vocab = ["under", "above", "to the left of", "to the right of"]
        
        self.processor, self.model = self._load_vqa_model(model_name)

    def _load_vqa_model(self, model_name: str):
        """Loads the VQA model and processor."""
        logger.info(f"Loading VQA model '{model_name}' on {DEVICE}...")
        try:
            # Using AutoModelForCausalLM for broader compatibility with vision models
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=TORCH_DTYPE,
                device_map="auto"
            ).eval()
            processor = AutoProcessor.from_pretrained(model_name)
            logger.info("VQA model loaded successfully.")
            return processor, model
        except Exception as e:
            logger.error(f"Failed to load VQA model: {e}", exc_info=True)
            raise

    def _query_vqa(self, image: Image.Image, prompt: str) -> str:
        """Sends a single prompt and image to the VQA model."""
        messages = [{"role": "user", "content": [prompt, image]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=50)
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Clean up the response to get just the answer part
        answer = response.split("user\n")[-1].split("assistant\n")[-1].strip()
        logger.debug(f"Prompt: '{prompt}' | Raw VQA Answer: '{answer}'")
        return answer

    def _validate_color(self, image: Image.Image, obj: str, expected_color: str) -> bool:
        """Validates the color of a single object."""
        prompt = f"In the image, what color is the '{obj}'? Choose only from: {', '.join(self.color_vocab)}."
        for _ in range(MAX_VQA_ATTEMPTS):
            answer = self._query_vqa(image, prompt)
            if expected_color.lower() in answer.lower():
                logger.debug(f"✔ '{obj}': expected '{expected_color}', got '{answer}'")
                return True
            time.sleep(1) # Simple backoff
        logger.warning(f"✘ '{obj}': expected '{expected_color}', but failed to get correct answer after {MAX_VQA_ATTEMPTS} attempts.")
        return False

    def _validate_position(self, image: Image.Image, obj1: str, obj2: str, expected_relation: str) -> bool:
        """Validates the relation between two objects."""
        prompt = f"In the image, what is the spatial relationship between the '{obj1}' and the '{obj2}'? Choose only from: {', '.join(self.relation_vocab)}."
        for _ in range(MAX_VQA_ATTEMPTS):
            answer = self._query_vqa(image, prompt)
            if expected_relation.lower() in answer.lower():
                logger.debug(f"✔ '{obj1}'->'{obj2}': expected '{expected_relation}', got '{answer}'")
                return True
            time.sleep(1)
        logger.warning(f"✘ '{obj1}'->'{obj2}': expected '{expected_relation}', but failed to get correct answer after {MAX_VQA_ATTEMPTS} attempts.")
        return False

    def run(self, args):
        """Runs the full validation pipeline."""
        try:
            with open(args.input_json, 'r') as f:
                data_to_check = json.load(f)
            logger.info(f"Loaded {len(data_to_check)} samples from '{args.input_json}' that passed object presence.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load input JSON: {e}")
            return

        fully_verified_samples = []
        for item in tqdm(data_to_check, desc=f"Validating {self.task_type} attributes"):
            image_id = item.get("id")
            image_path = Path(args.image_folder) / f"concept_{image_id:05d}_{args.caption_type}.webp"
            
            if not image_path.exists():
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            
            is_valid = True
            if self.task_type == 'color':
                objects, colors = item.get("objects"), item.get("attributes")
                for obj, color in zip(objects, colors):
                    if not self._validate_color(image, obj, color):
                        is_valid = False
                        break
            else: # position
                objects, relations = item.get("objects"), item.get("attributes")
                for i in range(len(relations)):
                    if not self._validate_position(image, objects[i], objects[i+1], relations[i]):
                        is_valid = False
                        break
            
            if is_valid:
                fully_verified_samples.append(item)
                logger.info(f"Image {image_id} PASSED all validation stages.")
        
        with open(args.output_json, 'w') as f:
            json.dump(fully_verified_samples, f, indent=2)
        logger.info(f"Finished. Found {len(fully_verified_samples)} fully verified samples. Saved to '{args.output_json}'.")

def main():
    parser = argparse.ArgumentParser(description="Validate image attributes (color/position) using a VQA model.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON file that has passed object presence check.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--caption_type", type=str, required=True, choices=['handmade', 'llm'], help="Image set to validate.")
    parser.add_argument("--task_type", type=str, required=True, choices=['color', 'position'], help="The type of attribute to validate.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the final, fully verified JSON file.")
    args = parser.parse_args()

    validator = AttributeValidator(task_type=args.task_type)
    validator.run(args)

if __name__ == "__main__":
    main()