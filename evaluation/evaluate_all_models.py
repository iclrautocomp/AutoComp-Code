import subprocess
import shlex

# --- Master Configuration ---

# Define all datasets to be evaluated. Each entry specifies the HF repo, task details,
# and the available caption sources ('minimal', 'contextual', or both).
BENCHMARK_CONFIGS = [
    # --- Standalone Minimal Benchmarks ---
    {'name': 'Minimal Color N=2', 'hf_repo_id': 'AutoComp/AutoComp-Minimal-N2-C', 'task_type': 'color', 'num_objects': 2, 'sources': ['minimal']},
    {'name': 'Minimal Color N=3', 'hf_repo_id': 'AutoComp/AutoComp-Minimal-N3-C', 'task_type': 'color', 'num_objects': 3, 'sources': ['minimal']},
    {'name': 'Minimal Position N=2', 'hf_repo_id': 'AutoComp/AutoComp-Minimal-N2-P', 'task_type': 'position', 'num_objects': 2, 'sources': ['minimal']},
    {'name': 'Minimal Position N=3', 'hf_repo_id': 'AutoComp/AutoComp-Minimal-N3-P', 'task_type': 'position', 'num_objects': 3, 'sources': ['minimal']},

    # --- Standalone Contextual Benchmarks ---
    {'name': 'Contextual Color N=2', 'hf_repo_id': 'AutoComp/AutoComp-Contextual-N2-C', 'task_type': 'color', 'num_objects': 2, 'sources': ['contextual']},
    {'name': 'Contextual Color N=3', 'hf_repo_id': 'AutoComp/AutoComp-Contextual-N3-C', 'task_type': 'color', 'num_objects': 3, 'sources': ['contextual']},
    {'name': 'Contextual Position N=2', 'hf_repo_id': 'AutoComp/AutoComp-Contextual-N2-P', 'task_type': 'position', 'num_objects': 2, 'sources': ['contextual']},
    {'name': 'Contextual Position N=3', 'hf_repo_id': 'AutoComp/AutoComp-Contextual-N3-P', 'task_type': 'position', 'num_objects': 3, 'sources': ['contextual']},
    
    # --- Paired Benchmarks (for A/B Test Analysis) ---
    {'name': 'Paired Color N=2', 'hf_repo_id': 'AutoComp/AutoComp-Paired-N2-C', 'task_type': 'color', 'num_objects': 2, 'sources': ['minimal', 'contextual']},
    {'name': 'Paired Color N=3', 'hf_repo_id': 'AutoComp/AutoComp-Paired-N3-C', 'task_type': 'color', 'num_objects': 3, 'sources': ['minimal', 'contextual']},
    {'name': 'Paired Position N=2', 'hf_repo_id': 'AutoComp/AutoComp-Paired-N2-P', 'task_type': 'position', 'num_objects': 2, 'sources': ['minimal', 'contextual']},
    {'name': 'Paired Position N=3', 'hf_repo_id': 'AutoComp/AutoComp-Paired-N3-P', 'task_type': 'position', 'num_objects': 3, 'sources': ['minimal', 'contextual']},
]

# --- Master list of all models to evaluate, using your exact names ---
CLIP_MODELS = [
    {'family': 'clip', 'name': 'ViT-B-32-quickgelu', 'pretrained': 'openai'},
    {'family': 'clip', 'name': 'ViT-B-16-quickgelu', 'pretrained': 'openai'},
    {'family': 'clip', 'name': 'ViT-L-14-quickgelu', 'pretrained': 'openai'},
    {'family': 'clip', 'name': 'ViT-H-14-quickgelu', 'pretrained': 'openai'},
    {'family': 'clip', 'name': 'ViT-H-14-378-quickgelu', 'pretrained': 'dfn5b'},
    {'family': 'clip', 'name': 'ViT-SO400M-14-SigLIP-384', 'pretrained': 'siglip-384'},
    {'family': 'clip', 'name': 'EVA02-B-16', 'pretrained': 'laion2b_s34b_b88k'},
    {'family': 'clip', 'name': 'EVA02-E-14', 'pretrained': 'laion2b_s32b_b79k'},
]

SIGLIP_MODELS = [
    {'family': 'siglip', 'name': 'google/siglip-base-patch16-224'},
    {'family': 'siglip', 'name': 'google/siglip-base-patch16-384'},
    {'family': 'siglip', 'name': 'google/siglip-large-patch16-256'},
    {'family': 'siglip', 'name': 'google/siglip-large-patch16-384'},
    {'family': 'siglip', 'name': 'google/siglip-so400m-patch14-384'},
    {'family': 'siglip', 'name': 'google/siglip2-base-patch16-224'},
    {'family': 'siglip', 'name': 'google/siglip2-base-patch16-256'},
    {'family': 'siglip', 'name': 'google/siglip2-base-patch16-384'},
    {'family': 'siglip', 'name': 'google/siglip2-large-patch16-256'},
    {'family': 'siglip', 'name': 'google/siglip2-large-patch16-384'},
    {'family': 'siglip', 'name': 'google/siglip2-so400m-patch14-384'},
    {'family': 'siglip', 'name': 'google/siglip2-giant-opt-patch16-256'},
]

MODELS_TO_EVALUATE = CLIP_MODELS + SIGLIP_MODELS


# --- General Settings ---
BATCH_SIZE = 4
LOG_WANDB = True
WANDB_PROJECT = "Auto-Comp-Eval"

def main():
    """
    Loops through all models and benchmarks, executing an evaluation for each valid combination.
    """
    for bench_config in BENCHMARK_CONFIGS:
        for model_config in MODELS_TO_EVALUATE:
            # Iterate through only the available sources for the current benchmark
            for source in bench_config['sources']:
                
                model_family = model_config['family']
                model_name = model_config['name']
                
                print("\n" + "="*80)
                print(f"Queueing Evaluation for: {bench_config['name']}")
                print(f"  - Model: {model_name} ({model_family})")
                print(f"  - Caption Source: {source}")
                print("="*80)

                cmd_parts = [
                    "python", "evaluate_model.py",
                    "--model_family", model_family,
                    "--model_name", shlex.quote(model_name),
                    "--hf_repo_id", shlex.quote(bench_config['hf_repo_id']),
                    "--task_type", bench_config['task_type'],
                    "--num_objects", str(bench_config['num_objects']),
                    "--caption_source", source, # Pass 'minimal' or 'contextual'
                    "--batch_size", str(BATCH_SIZE),
                ]

                if model_family == 'clip' and 'pretrained' in model_config:
                    cmd_parts.extend(["--clip_pretrained", shlex.quote(model_config['pretrained'])])

                if LOG_WANDB:
                    sanitized_name = model_name.replace("/", "--")
                    run_name = f"{sanitized_name}_{bench_config['task_type']}_N{bench_config['num_objects']}_{source}"
                    cmd_parts.extend([
                        "--log_wandb",
                        "--wandb_project", WANDB_PROJECT,
                        "--wandb_run_name", run_name,
                    ])
                
                cmd_str = " ".join(cmd_parts)
                print(f"Executing: {cmd_str}\n")
                
                try:
                    subprocess.run(shlex.split(cmd_str), check=True)
                except subprocess.CalledProcessError as e:
                    print(f"\n--- Evaluation FAILED for {model_name} with error code {e.returncode} ---\n")

if __name__ == "__main__":
    main()