# Auto-Comp Pipeline: Code Supplement

This repository contains the official implementation for the paper "Auto-Comp: An Automated Pipeline for Scalable Compositional Probing of Contrastive Vision-Language Models". It includes all scripts required to generate the benchmarks and reproduce our results.

## Benchmarks on Hugging Face

All generated benchmark datasets (**Auto-Comp-CP**) are available for immediate use on the Hugging Face Hub at the following organization URL:

[https://huggingface.co/AutoComp](https://huggingface.co/AutoComp)

## Setup and Requirements

1.  **Clone this repository:**
    ```bash
    git clone [URL_TO_YOUR_ANONYMOUS_REPO]
    cd [REPO_NAME]
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Benchmark Generation Pipeline

The pipeline consists of three main stages. The scripts are designed to be run in order, with the output of one stage serving as the input for the next.

### Step 1: Generate Captions

Use the unified `generate_captions.py` script to create the paired Minimal (handmade) and Contextual (LLM) captions.

**Usage:**
```bash
cd generation
python generate_captions.py --task_type [color|position] --num_objects [N] --num_captions [TOTAL] --output_file [path/to/captions.json]
```
* `--task_type`: Specify `color` for attribute binding or `position` for relational binding.
* `--num_objects`: The number of objects (e.g., 2 or 3).
* `--num_captions`: The total number of initial concepts to generate.

### Step 2: Generate Images

Use the `generate_images.py` script to generate images from the captions created in Step 1. This script is designed for distributed generation across multiple GPUs.

**Usage (example for one of four GPUs):**
```bash
cd generation
python generate_images.py \
    --input_json [path/to/captions.json] \
    --output_dir [path/to/save/images] \
    --segment_index 0 \
    --total_segments 4
```

### Step 3: Validate Generated Data

Validation is a two-part process.

**Part A: Object Presence & Background Check**

This step uses GroundedSAM2. Its environment is best managed via Docker, as recommended by the official repository.

1.  **Clone the GroundedSAM2 repository:**
    ```bash
    git clone [https://github.com/IDEA-Research/Grounded-Segment-Anything-2.git](https://github.com/IDEA-Research/Grounded-Segment-Anything-2.git)
    ```
2.  **Copy our script into their directory:**
    ```bash
    cp filtering/evaluate_object_presence.py Grounded-Segment-Anything-2/
    ```
3.  **Build and run their Docker container:** (Follow instructions in the GroundedSAM2 README)
    ```bash
    cd Grounded-Segment-Anything-2
    # ... commands to build and run docker ...
    ```
4.  **Run the script inside the Docker container:**
    ```bash
    python evaluate_object_presence.py \
        --input_json [path/to/captions.json] \
        --image_folder [path/to/generated/images/handmade] \
        --caption_type handmade \
        --output_json [path/to/object_presence_verified.json] \
        --check_background # Add this flag only for Minimal (handmade) images
    ```

**Part B: Attribute Correctness Check (VQA)**

This script takes the JSON file from the previous step and performs the final VQA check.

**Usage:**
```bash
cd filtering
python validate_attributes_vqa.py \
    --input_json [path/to/object_presence_verified.json] \
    --image_folder [path/to/generated/images/handmade] \
    --caption_type handmade \
    --task_type [color|position] \
    --output_json [path/to/fully_validated_data.json]
```

### Step 4 (Optional): Offline Hard Negative Generation
Our main evaluation scripts generate hard negatives in real-time for efficiency. However, we also provide a script to generate all Swap and Confusion negatives offline and save them to a new JSON file. This is useful for creating a complete, self-contained benchmark file.
```bash
cd generation
python generate_swap_confusion.py \
    --input_json [path/to/fully_validated_data.json] \
    --output_json [path/to/data_with_negatives.json] \
    --task_type [color|position]
```

## Evaluation Scripts

To reproduce the evaluation results from the paper, use the `run_all_evals.py` script. You must first configure the `BENCHMARK_CONFIGS` and `MODELS_TO_EVALUATE` lists at the top of the script with the desired Hugging Face repo IDs and model names.
```bash
cd evaluation
# This will loop through all configured models and benchmarks
python evaluate_all_models.py
```
This will call `evaluate_model.py` for each configuration, which in turn uses the real-time `hf_benchmark.py` module to generate negatives and compute scores.