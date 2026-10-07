import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
import json
import argparse
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

from peft import LoraConfig, get_peft_model  # type: ignore

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.paths import PROMPTS_DIR, CHECKPOINTS_DIR, TEACHERS_CHECKPOINTS_DIR, MODELS_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class RecommendationRankingDataset(Dataset):
    """
    Dataset wrapping generated Ranking Prompts and Target Responses for LLM Instruction Tuning.
    Supports any Causal LLM architecture (e.g. Qwen, LLaMA, Mistral).
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, max_samples: int = 0):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
                    if max_samples > 0 and len(self.samples) >= max_samples:
                        break
                    
        self.tokenizer = tokenizer
        self.max_length = max_length
        logging.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = "Predict whether the user will interact with the candidate item. Return exactly 1 if yes, 0 if no."
        prompt_text = f"### Instruction:\n{instruction}\n### Input:\n{sample['prompt']}\n### Response:\n"
        response_text = f"{sample['response']}{self.tokenizer.eos_token}"

        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        max_prompt_len = max(10, self.max_length - len(response_ids))
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True, truncation=True, max_length=max_prompt_len)

        input_ids = prompt_ids + response_ids
        labels = ([-100] * len(prompt_ids)) + response_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def load_config(config_path: Path) -> dict:
    """Loads fine-tuning configuration parameters from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def finetune_llm_teacher(config: dict):
    """
    Fine-tunes any Causal LLM Teacher (e.g. Qwen, LLaMA, Mistral) using parameters from YAML config.
    """
    model_path = config.get("model_path", "Qwen/Qwen3.5-4B")
    dataset_name = config.get("dataset", "movielens")

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    mock_cfg = config.get("mock", {})

    epochs = train_cfg.get("epochs", 3)
    batch_size = train_cfg.get("batch_size", 4)
    grad_steps = train_cfg.get("grad_steps", 4)
    max_length = train_cfg.get("max_length", 2048)
    learning_rate = float(train_cfg.get("learning_rate", 2e-4))
    lr_scheduler_type = train_cfg.get("lr_scheduler_type", "cosine")
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    save_steps = train_cfg.get("save_steps", None)
    eval_steps = train_cfg.get("eval_steps", None)

    lora_r = lora_cfg.get("r", 32)
    lora_alpha = lora_cfg.get("alpha", 64)
    lora_dropout = lora_cfg.get("dropout", 0.05)
    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    max_samples = mock_cfg.get("max_samples", 0)
    max_steps = mock_cfg.get("max_steps", -1)

    data_path = PROMPTS_DIR / f"{dataset_name}_train_prompts.jsonl"
    if not data_path.exists():
        logging.error(f"Dataset file not found: {data_path}")
        logging.error(f"Please run `python knowlight/prompts/prompt_builder.py --dataset {dataset_name}` first.")
        return

    # Extract model folder name to avoid checkpoint collisions between different LLM models
    model_name_for_dir = Path(model_path).name
    output_dir = CHECKPOINTS_DIR / "teachers" / model_name_for_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # If the model was already fine-tuned, skip re-training and run score extraction directly
    save_path = output_dir / "finetuned_teacher_best"
    if save_path.exists() and (save_path / "adapter_config.json").exists() and not config.get("force_retrain", False):
        logging.info(f"Found existing completed Fine-Tuned LLM Teacher at: {save_path}")
        logging.info("Skipping training phase and proceeding directly to LLM score extraction...")
        from knowlight.teachers.llm.extract_scores import extract_dataset_llm_scores
        extract_cfg = config.get("extraction", {})
        extract_batch_size = extract_cfg.get("batch_size", 16)
        extract_dataset_llm_scores(
            dataset_name=dataset_name,
            model_path=str(save_path),
            teacher_name=model_name_for_dir,
            batch_size=extract_batch_size
        )
        return

    # Resolve model_path (supports local directory in models/, absolute paths, or HF repo IDs)
    target_path = Path(model_path)
    if not target_path.exists():
        candidate_models = MODELS_DIR / model_path
        if candidate_models.exists():
            target_path = candidate_models
        else:
            repo_name = Path(model_path).name
            candidate_repo = MODELS_DIR / repo_name
            if candidate_repo.exists():
                target_path = candidate_repo

    is_local = target_path.exists() and target_path.is_dir()
    model_identifier = str(target_path) if is_local else model_path

    logging.info(f"Loading tokenizer from '{model_identifier}' (local_files_only={is_local})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_identifier,
        local_files_only=is_local,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect GPU hardware capabilities for bfloat16 vs float16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    logging.info(f"Precision setup -> bf16: {use_bf16}, fp16: {use_fp16}, dtype: {compute_dtype}")

    # Force direct CUDA GPU 0 mapping when CUDA is available for instant 4-bit loading
    if torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = None

    quantization = lora_cfg.get("quantization", config.get("quantization", None))
    if quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            logging.info(f"Loading LLM model in 4-bit QLoRA mode from '{model_identifier}'...")
            model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                quantization_config=quant_config,
                device_map=device_map or "auto",
                local_files_only=is_local,
                trust_remote_code=True
            )
            try:
                from peft import prepare_model_for_kbit_training #type: ignore
                model = prepare_model_for_kbit_training(model)
            except Exception:
                pass
        except Exception as e:
            logging.warning(f"Could not load in 4-bit mode ({e}). Falling back to standard precision loading.")
            model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                torch_dtype=compute_dtype,
                device_map=device_map,
                local_files_only=is_local,
                trust_remote_code=True
            )
    else:
        logging.info(f"Loading LLM model from '{model_identifier}' (local_files_only={is_local}, device_map={device_map})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_identifier,
            torch_dtype=compute_dtype,
            device_map=device_map,
            local_files_only=is_local,
            trust_remote_code=True
        )

    # 1. Setup LoRA PEFT Model 
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if hasattr(model, "config"):
        model.config.use_cache = False

    if quantization != "4bit":
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            model.gradient_checkpointing_enable()

    # 2. Load Dataset and split into Train / Eval
    dataset = RecommendationRankingDataset(str(data_path), tokenizer, max_length=max_length, max_samples=max_samples)
    if max_samples and max_samples < len(dataset):
        dataset, _ = torch.utils.data.random_split(
            dataset, [max_samples, len(dataset) - max_samples], generator=torch.Generator().manual_seed(42)
        )
        logging.info(f"Mock test mode: truncated dataset to {max_samples} samples.")
    
    # 95% Train, 5% Eval split for Early Stopping and monitoring
    train_size = max(1, int(0.95 * len(dataset)))
    eval_size = max(1, len(dataset) - train_size)
    if train_size + eval_size != len(dataset):
        train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42)
    )

    logging.info(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # 3. Dynamic Evaluation & Save Steps (capped at 500 steps ~4 hours to survive 24h cluster walltime)
    calc_steps = max(5, len(train_dataset) // max(1, (batch_size * grad_steps * 5)))
    if max_steps > 0:
        calc_steps = max(2, max_steps // 2)
    
    if save_steps is None:
        save_steps = min(500, calc_steps)
    if eval_steps is None:
        eval_steps = save_steps
    
    logging.info(f"Checkpoint save_steps: {save_steps} | eval_steps: {eval_steps}")

    # 4. Setup TrainingArguments safely across transformers versions
    is_mock = max_steps > 0 or max_samples > 0
    import inspect
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    args_dict = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler_type,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "num_train_epochs": epochs,
        "max_steps": max_steps if max_steps > 0 else -1,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "logging_steps": 1 if is_mock else 10,
        "save_total_limit": 3,
        "load_best_model_at_end": not is_mock,
        "metric_for_best_model": "eval_loss",
        "optim": "adamw_torch",
        "report_to": "none"
    }

    if "warmup_ratio" in sig_params:
        args_dict["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in sig_params:
        args_dict["warmup_steps"] = max(0, int(warmup_ratio * 100))

    if "eval_strategy" in sig_params:
        args_dict["eval_strategy"] = "no" if is_mock else "steps"
        args_dict["eval_steps"] = eval_steps if not is_mock else 999999
    elif "evaluation_strategy" in sig_params:
        args_dict["evaluation_strategy"] = "no" if is_mock else "steps"
        args_dict["eval_steps"] = eval_steps if not is_mock else 999999

    if "save_strategy" in sig_params:
        args_dict["save_strategy"] = "no" if is_mock else "steps"
        args_dict["save_steps"] = save_steps if not is_mock else 999999

    filtered_args = {k: v for k, v in args_dict.items() if k in sig_params}
    training_args = TrainingArguments(**filtered_args)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if max_steps <= 0 else []
    )

    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = None
    if os.path.isdir(str(output_dir)):
        last_checkpoint = get_last_checkpoint(str(output_dir))
        if last_checkpoint is not None:
            logging.info(f"Resuming training from checkpoint: {last_checkpoint}")

    logging.info(f"Starting Fine-Tuning for LLM: {model_name_for_dir}...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 6. Save Final Best Model to data/checkpoints/teachers/<model_name>/<dataset_name>/finetuned_teacher_best
    save_path = output_dir / "finetuned_teacher_best"
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logging.info(f"Successfully saved Fine-Tuned LLM Teacher checkpoint to: {save_path}")

    # Explicitly free training memory (model, optimizer states, trainer caches, datasets)
    del trainer
    del model
    if "train_dataset" in locals():
        del train_dataset
    if "eval_dataset" in locals():
        del eval_dataset
    if "dataset" in locals():
        del dataset

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logging.info("Cleaned up fine-tuning training memory from GPU.")

    # 7. Automatically extract scores and save to cache
    logging.info("Automatically extracting LLM probability scores to cache...")
    from knowlight.teachers.llm.extract_scores import extract_dataset_llm_scores
    extract_cfg = config.get("extraction", {})
    extract_batch_size = extract_cfg.get("batch_size", 16)
    extract_dataset_llm_scores(
        dataset_name=dataset_name,
        model_path=str(save_path),
        teacher_name=model_name_for_dir,
        batch_size=extract_batch_size
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Causal LLM Teacher using YAML Configuration.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/finetune_full.yaml", 
        help="Path to YAML configuration file (e.g. config/finetune_full.yaml or config/finetune_mock.yaml)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name from YAML configuration (e.g. movielens, amazon_cd_2014, amazon_beauty, amazon_toys_and_games, yelp)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Override model path from YAML configuration (e.g. models/Qwen3.5-4B, models/Llama-3-8B-Instruct)"
    )
    parser.add_argument(
        "--extract_batch_size",
        type=int,
        default=None,
        help="Override batch size for score extraction after fine-tuning (e.g. 16, 8, 4)"
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force re-training even if finetuned_teacher_best checkpoint already exists"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Number of steps between checkpoint saves (default: max 500 steps to ensure checkpoints within 24h jobs)"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Number of steps between evaluations"
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)

    if args.dataset:
        config["dataset"] = args.dataset
    if args.model_path:
        config["model_path"] = args.model_path
    if args.force_retrain:
        config["force_retrain"] = True
    if args.save_steps:
        if "training" not in config:
            config["training"] = {}
        config["training"]["save_steps"] = args.save_steps
    if args.eval_steps:
        if "training" not in config:
            config["training"] = {}
        config["training"]["eval_steps"] = args.eval_steps
    if args.extract_batch_size:
        if "extraction" not in config:
            config["extraction"] = {}
        config["extraction"]["batch_size"] = args.extract_batch_size

    model_path = config.get("model_path", "Qwen/Qwen3.5-4B")
    dataset_name = config.get("dataset", "movielens")

    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    mock_cfg = config.get("mock", {})

    print("=" * 70)
    print("  KnowLight LLM Teacher Fine-Tuning ")
    print(f"  Config File: {config_path}")
    print(f"  Model Path : {model_path}")
    print(f"  Dataset    : {dataset_name}")
    print(f"  LoRA Config: r={lora_cfg.get('r', 32)}, alpha={lora_cfg.get('alpha', 64)}, dropout={lora_cfg.get('dropout', 0.05)}, quant={lora_cfg.get('quantization', None)}")
    print(f"  Training   : Epochs={train_cfg.get('epochs', 3)}, Batch={train_cfg.get('batch_size', 4)}x{train_cfg.get('grad_steps', 4)} (LR={train_cfg.get('learning_rate', 2e-4)}, MaxLen={train_cfg.get('max_length', 2048)})")
    if mock_cfg.get("max_samples", 0) > 0 or mock_cfg.get("max_steps", -1) > 0:
        print(f"  [MOCK TEST MODE] max_samples={mock_cfg.get('max_samples')}, max_steps={mock_cfg.get('max_steps')}")
    print("=" * 70)

    finetune_llm_teacher(config)


if __name__ == "__main__":
    main()
