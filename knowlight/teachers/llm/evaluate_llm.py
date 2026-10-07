import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.paths import PROMPTS_DIR, CHECKPOINTS_DIR, MODELS_DIR
from knowlight.utils.metrics import compute_topk_metrics

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class LLMTeacherEvaluator:
    """
    Evaluates fine-tuned LLM Teacher model on test ranking prompts.
    Computes Hit Ratio (HR@K), NDCG@K, Accuracy, and MRR.
    """
    def __init__(self, model_path: str, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        logging.info(f"Loading Fine-Tuned LLM Teacher from '{model_path}' for Evaluation...")
        
        path_obj = Path(model_path)
        adapter_cfg_file = path_obj / "adapter_config.json"
        
        if adapter_cfg_file.exists():
            from peft import PeftModel, PeftConfig  # type: ignore
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path
            
            # Resolve local base model path if available in models/
            base_path_obj = Path(base_model_path)
            if not base_path_obj.exists():
                candidate = MODELS_DIR / base_path_obj.name
                if candidate.exists():
                    base_model_path = str(candidate)
            
            logging.info(f"Detected PEFT/LoRA adapter. Loading base model from '{base_model_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            dev_map = {"": 0} if torch.cuda.is_available() else None
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=dev_map,
                local_files_only=Path(base_model_path).exists()
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            dev_map = {"": 0} if torch.cuda.is_available() else None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=dev_map
            )
            
        self.model.eval()

    @torch.no_grad()
    def evaluate_sample(self, prompt_text: str, target_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluates prompt likelihood / score for candidate items.
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]  # Logits over vocabulary for next token
        
        item_scores = {}
        for item_id in target_scores.keys():
            # Tokenize item candidate label to evaluate score
            tokens = self.tokenizer.encode(str(item_id), add_special_tokens=False)
            if tokens:
                token_id = tokens[0]
                item_scores[item_id] = logits[0, token_id].item()
            else:
                item_scores[item_id] = 0.0
                
        return item_scores


def evaluate_llm_teacher(
    dataset_name: str,
    model_path: str = None,
    split: str = "test",
    prompts_dir: Path = PROMPTS_DIR
):
    """
    Loads test prompts and evaluates the fine-tuned LLM Teacher on ranking metrics.
    """
    prompt_file = prompts_dir / split / f"{dataset_name}_ranking_prompts.jsonl"
    if not prompt_file.exists():
        prompt_file = prompts_dir / f"{dataset_name}_ranking_prompts.jsonl"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt dataset file not found at {prompt_file}. "
            f"Please run `python knowlight/prompts/prompt_builder.py --dataset {dataset_name} --split {split}` first."
        )

    # Resolve checkpoint path if model_path is not specified
    if not model_path:
        teachers_dir = CHECKPOINTS_DIR / "teachers"
        found_checkpoint = None
        if teachers_dir.exists():
            for ckpt in teachers_dir.glob(f"*/{dataset_name}/finetuned_teacher_best"):
                if ckpt.exists():
                    found_checkpoint = ckpt
                    break
            if not found_checkpoint:
                for ckpt in teachers_dir.glob(f"*/*/{dataset_name}/finetuned_teacher_best"):
                    if ckpt.exists():
                        found_checkpoint = ckpt
                        break
        if found_checkpoint:
            model_path = str(found_checkpoint)
        else:
            raise FileNotFoundError(f"No fine-tuned teacher checkpoint found for dataset '{dataset_name}'.")

    logging.info(f"Loading '{split}' prompts from {prompt_file}...")
    samples = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    evaluator = LLMTeacherEvaluator(model_path)
    
    all_pos_indices = []
    all_rank_scores = []
    mrr_list = []

    logging.info(f"Evaluating LLM Teacher on {len(samples)} test samples...")
    for idx, sample in enumerate(samples):
        prompt_text = sample["prompt"]
        response_str = sample["response"]
        try:
            target_scores = json.loads(response_str)
        except Exception:
            continue
            
        pos_item = None
        for item_id, score in target_scores.items():
            if score == 1.0:
                pos_item = item_id
                break
        if not pos_item:
            continue

        predicted_item_scores = evaluator.evaluate_sample(prompt_text, target_scores)
        candidate_ids = list(target_scores.keys())
        
        try:
            pos_idx = candidate_ids.index(pos_item)
        except ValueError:
            pos_idx = 0

        scores_vector = [predicted_item_scores.get(cid, 0.0) for cid in candidate_ids]
        
        all_pos_indices.append(pos_idx)
        all_rank_scores.append(scores_vector)

        # Compute rank for MRR
        ranked_items = sorted(predicted_item_scores.keys(), key=lambda x: predicted_item_scores[x], reverse=True)
        try:
            rank = ranked_items.index(pos_item) + 1
        except ValueError:
            rank = len(ranked_items)
        mrr_list.append(1.0 / rank)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            logging.info(f"Evaluated {idx + 1}/{len(samples)} test prompts...")

    if all_rank_scores:
        pos_tensor = torch.tensor(all_pos_indices, dtype=torch.long)
        scores_tensor = torch.tensor(all_rank_scores, dtype=torch.float32)
        topk_results = compute_topk_metrics(pos_tensor, scores_tensor, top_k_list=[1, 3, 5])
    else:
        topk_results = {}

    mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    model_dir_name = Path(model_path).name

    print("=" * 70)
    print(f"  KnowLight LLM Teacher Evaluation Results ({model_dir_name} - {dataset_name.upper()})")
    print("=" * 70)
    print(f"  Evaluated Samples : {len(all_pos_indices)}")
    print(f"  HR@1  (Accuracy)  : {topk_results.get('HR@1', 0.0):.4f}")
    print(f"  HR@3              : {topk_results.get('HR@3', 0.0):.4f}")
    print(f"  HR@5              : {topk_results.get('HR@5', 0.0):.4f}")
    print(f"  NDCG@3            : {topk_results.get('NDCG@3', 0.0):.4f}")
    print(f"  NDCG@5            : {topk_results.get('NDCG@5', 0.0):.4f}")
    print(f"  Precision@1       : {topk_results.get('Precision@1', 0.0):.4f}")
    print(f"  Precision@3       : {topk_results.get('Precision@3', 0.0):.4f}")
    print(f"  Precision@5       : {topk_results.get('Precision@5', 0.0):.4f}")
    print(f"  Recall@1          : {topk_results.get('Recall@1', 0.0):.4f}")
    print(f"  Recall@3          : {topk_results.get('Recall@3', 0.0):.4f}")
    print(f"  Recall@5          : {topk_results.get('Recall@5', 0.0):.4f}")
    print(f"  MRR               : {mrr:.4f}")
    print("=" * 70)

    # Save metrics to metrics.json in checkpoint directory
    save_dir = Path(model_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = save_dir / "metrics.json"

    metrics_dict = {
        "dataset": dataset_name,
        "split": split,
        "evaluated_samples": len(all_pos_indices),
        "metrics": {
            "HR@1": round(topk_results.get("HR@1", 0.0), 4),
            "HR@3": round(topk_results.get("HR@3", 0.0), 4),
            "HR@5": round(topk_results.get("HR@5", 0.0), 4),
            "NDCG@1": round(topk_results.get("NDCG@1", 0.0), 4),
            "NDCG@3": round(topk_results.get("NDCG@3", 0.0), 4),
            "NDCG@5": round(topk_results.get("NDCG@5", 0.0), 4),
            "Precision@1": round(topk_results.get("Precision@1", 0.0), 4),
            "Precision@3": round(topk_results.get("Precision@3", 0.0), 4),
            "Precision@5": round(topk_results.get("Precision@5", 0.0), 4),
            "Recall@1": round(topk_results.get("Recall@1", 0.0), 4),
            "Recall@3": round(topk_results.get("Recall@3", 0.0), 4),
            "Recall@5": round(topk_results.get("Recall@5", 0.0), 4),
            "MRR": round(mrr, 4)
        }
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    logging.info(f"Successfully saved evaluation metrics to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned LLM Teacher Model on Ranking Test Prompts")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name")
    parser.add_argument("--model_path", type=str, default=None, help="Fine-tuned LLM model checkpoint path (optional)")
    parser.add_argument("--split", type=str, default="test", help="Data split to evaluate on ('test' or 'train')")

    args = parser.parse_args()
    evaluate_llm_teacher(args.dataset, args.model_path, args.split)


if __name__ == "__main__":
    import numpy as np
    main()
