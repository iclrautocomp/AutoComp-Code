import argparse
import json
import re
from itertools import product, permutations
from typing import List, Dict
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardNegativeGenerator:
    """
    Generates and saves hard negative captions for both color and position benchmarks
    from a file of validated positive samples.
    """
    def __init__(self, task_type: str):
        if task_type not in ['color', 'position']:
            raise ValueError("task_type must be either 'color' or 'position'")
        self.task_type = task_type
        logger.info(f"Initialized HardNegativeGenerator for task: '{self.task_type}'")

    def _substitute_caption(self, orig_caption: str, orig_elements: Dict, new_elements: Dict) -> str:
        """
        Robustly substitutes elements (objects, colors, or relations) in a caption
        using a two-pass placeholder strategy.
        """
        caption = orig_caption.lower()
        
        # Combine all original elements to be replaced
        all_orig_elements = []
        if 'objects' in orig_elements: all_orig_elements.extend(orig_elements['objects'])
        if 'attributes' in orig_elements: all_orig_elements.extend(orig_elements['attributes'])
        
        # Create unique placeholders for every original element
        placeholders = {elem.lower(): f"__PLACEHOLDER_{i}__" for i, elem in enumerate(all_orig_elements)}
        
        # Pass 1: Replace original elements with placeholders
        # Sort by length descending to replace longer names first (e.g., "to the left of")
        sorted_orig = sorted(placeholders.keys(), key=len, reverse=True)
        for elem in sorted_orig:
            placeholder = placeholders[elem]
            caption = re.sub(rf"\b{re.escape(elem)}\b", placeholder, caption, flags=re.IGNORECASE)
            
        # Pass 2: Replace placeholders with the final new elements
        if 'objects' in orig_elements:
            for i, orig_obj in enumerate(orig_elements['objects']):
                placeholder = placeholders[orig_obj.lower()]
                new_obj = new_elements['objects'][i]
                caption = caption.replace(placeholder, new_obj.lower())
        
        if 'attributes' in orig_elements:
            for i, orig_attr in enumerate(orig_elements['attributes']):
                placeholder = placeholders[orig_attr.lower()]
                new_attr = new_elements['attributes'][i]
                caption = caption.replace(placeholder, new_attr.lower())
                
        # Capitalize the first letter of the sentence
        return caption.capitalize()

    def _generate_swap_negatives(self, item: Dict) -> List[str]:
        """Generates negatives for the Swap Benchmark."""
        orig_caption = item.get('handmade_caption') or item.get('llm_caption')
        objects = item['objects']

        attributes = item.get('relations') or item.get('colors')
        
        if self.task_type == 'color':
            # Permute attributes (colors)
            attr_perms = list(permutations(attributes))
            attr_perms.remove(tuple(attributes)) # Remove the original
            
            orig_elements = {'objects': objects, 'attributes': attributes}
            negatives = [
                self._substitute_caption(orig_caption, orig_elements, {'objects': objects, 'attributes': list(p)})
                for p in attr_perms
            ]
        else: # position
            # Permute objects
            obj_perms = list(permutations(objects))
            obj_perms.remove(tuple(objects)) # Remove the original
            
            orig_elements = {'objects': objects, 'attributes': attributes}
            negatives = [
                self._substitute_caption(orig_caption, orig_elements, {'objects': list(p), 'attributes': attributes})
                for p in obj_perms
            ]
        print(orig_caption, negatives)
        return negatives

    def _generate_confusion_negatives(self, item: Dict) -> List[str]:
        """Generates negatives for the Confusion Benchmark."""
        orig_caption = item.get('handmade_caption') or item.get('llm_caption')
        objects = item['objects']
        attributes = item.get('relations') or item.get('colors')

        obj_combos = list(product(objects, repeat=len(objects)))
        attr_combos = list(product(attributes, repeat=len(attributes)))
        
        all_configs = list(product(obj_combos, attr_combos))
        original_config = (tuple(objects), tuple(attributes))
        all_configs.remove(original_config) # Remove the original
        
        orig_elements = {'objects': objects, 'attributes': attributes}
        negatives = [
            self._substitute_caption(orig_caption, orig_elements, {'objects': list(p_obj), 'attributes': list(p_attr)})
            for p_obj, p_attr in all_configs
        ]
        print(orig_caption, negatives)
        return negatives

    def process_file(self, input_path: str, output_path: str):
        """
        Reads a JSON file of positive samples, generates hard negatives for each,
        and saves the augmented data to a new JSON file.
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load input file '{input_path}': {e}")
            return

        augmented_data = []
        for item in tqdm(data, desc="Generating Hard Negatives"):

            swap_negs = self._generate_swap_negatives(item)
            conf_negs = self._generate_confusion_negatives(item)

            input('stop')
            
            item['swap_negatives'] = swap_negs
            item['confusion_negatives'] = conf_negs
            augmented_data.append(item)
            
        with open(output_path, 'w') as f:
            json.dump(augmented_data, f, indent=2)
            
        logger.info(f"Processing complete. Saved {len(augmented_data)} items with hard negatives to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Generate and save hard negative captions for compositional benchmarks.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file of positive samples.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the output JSON file with added hard negatives.")
    parser.add_argument("--task_type", type=str, required=True, choices=['color', 'position'], help="The benchmark task type.")
    args = parser.parse_args()

    generator = HardNegativeGenerator(task_type=args.task_type)
    generator.process_file(args.input_json, args.output_json)

if __name__ == "__main__":
    main()