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
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress expected library warnings to clean up log output
warnings.filterwarnings("ignore", category=UserWarning, module="peft.*")
warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.paths import PROMPTS_DIR, CHECKPOINTS_DIR, CACHE_DIR, TEACHERS_CHECKPOINTS_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")



class LLMScoreExtractor:
    """
    Extracts continuous probability scores (Logits for token '1') from the fine-tuned LLM Teacher
    evaluating specific (user, item) pairs from the instruction prompts.
    """
    tokenizer: Any
    model: Any
    
    def __init__(self, model_path: str, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        logging.info(f"Loading LLM Teacher for Feature Extraction from '{model_path}'...")
        
        path_obj = Path(model_path)
        adapter_cfg_file = path_obj / "adapter_config.json"
        
        # Precision detection
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
        dev_map = {"": 0} if torch.cuda.is_available() else None

        if adapter_cfg_file.exists():
            from peft import PeftModel, PeftConfig  # type: ignore[import]
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path
            
            # Resolve local base model path if it exists in models/
            base_path_obj = Path(base_model_path)
            if not base_path_obj.exists():
                candidate = MODELS_DIR / base_path_obj.name
                if candidate.exists():
                    base_model_path = str(candidate)
            
            logging.info(f"Detected PEFT/LoRA adapter. Loading base model from '{base_model_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).exists())
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=compute_dtype,
                device_map=dev_map,
                local_files_only=Path(base_model_path).exists()
            )
            peft_model = PeftModel.from_pretrained(base_model, model_path)
            try:
                self.model = peft_model.merge_and_unload()
                logging.info("Merged LoRA adapter weights into base model for fast inference.")
            except Exception:
                self.model = peft_model
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).exists())
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=compute_dtype,
                device_map=dev_map,
                local_files_only=Path(model_path).exists()
            )
            
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

    @torch.inference_mode()
    def extract_interaction_probabilities_batch(self, prompt_texts: List[str], batch_size: int = 16) -> List[float]:
        """
        Passes a batch of instruction prompts into LLM and extracts the softmax probability
        of generating the '1' token at the next step.
        Uses optimized logits projection (num_logits_to_keep=1) to prevent memory allocation spikes.
        """
        from tqdm import tqdm

        instruction = "Predict whether the user will interact with the candidate item. Return exactly 1 if yes, 0 if no."
        formatted_prompts = [
            f"### Instruction:\n{instruction}\n### Input:\n{p}\n### Response:\n"
            for p in prompt_texts
        ]

        orig_pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        token_1_id = self.tokenizer.encode("1", add_special_tokens=False)[-1]
        token_0_id = self.tokenizer.encode("0", add_special_tokens=False)[-1]

        all_probs = []
        current_batch_size = max(1, batch_size)

        pbar = tqdm(total=len(formatted_prompts), desc="Extracting LLM Scores", unit="sample")
        i = 0
        while i < len(formatted_prompts):
            batch = formatted_prompts[i:i + current_batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            try:
                # Optimized forward pass: project logits only for the final token position
                try:
                    outputs = self.model(**inputs, num_logits_to_keep=1, use_cache=False)
                except TypeError:
                    try:
                        outputs = self.model(**inputs, logits_to_keep=1, use_cache=False)
                    except TypeError:
                        outputs = self.model(**inputs, use_cache=False)

                last_logits = outputs.logits[:, -1, :]
                logit_1 = last_logits[:, token_1_id]
                logit_0 = last_logits[:, token_0_id]
                pair_logits = torch.stack([logit_0, logit_1], dim=-1)
                probs = torch.softmax(pair_logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().tolist())

                pbar.update(len(batch))
                i += len(batch)
            except torch.cuda.OutOfMemoryError as oom:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    logging.warning(
                        f"CUDA OutOfMemory encountered during score extraction. "
                        f"Automatically reducing batch size to {current_batch_size} and retrying..."
                    )
                else:
                    pbar.close()
                    raise oom

        pbar.close()
        self.tokenizer.padding_side = orig_pad_side
        return all_probs

    @torch.inference_mode()
    def extract_interaction_probability(self, prompt_text: str) -> float:
        """
        Passes full instruction prompt into LLM and extracts the softmax probability
        of generating the '1' token at the next step.
        """
        return self.extract_interaction_probabilities_batch([prompt_text], batch_size=1)[0]


def extract_dataset_llm_scores(
    dataset_name: str,
    model_path: str = None,
    prompts_dir: Path = PROMPTS_DIR,
    cache_dir: Path = CACHE_DIR,
    teacher_name: str = "Qwen3.5-4B",
    batch_size: int = 16
) -> Path:
    """
    Loads saved .jsonl prompts directly from disk and extracts the continuous probability 
    scores for each (user, candidate_item) pair using the teacher model.
    Automatically generates prompts if missing, and extracts/caches scores on-the-fly.
    """
    prompt_file = prompts_dir / f"{dataset_name}_train_prompts.jsonl"
    if not prompt_file.exists():
        logging.info(f"Ranking prompt dataset not found at {prompt_file}. Automatically generating prompts now...")
        from knowlight.prompts.prompt_builder import generate_dataset_prompts
        generate_dataset_prompts(dataset_name)

    teacher_str = (teacher_name or "").lower()

    # Resolve fine-tuned teacher checkpoint path or base model
    if not model_path:
        teachers_dir = CHECKPOINTS_DIR / "teachers"
        found_checkpoint = None
        if teachers_dir.exists():
            # 1. Search for specific teacher: <CHECKPOINTS_DIR>/teachers/<teacher_name>/<dataset_name>/finetuned_teacher_best
            teacher_ckpt = teachers_dir / teacher_name / dataset_name / "finetuned_teacher_best"
            if teacher_ckpt.exists():
                found_checkpoint = teacher_ckpt
            else:
                # 2. Search for any checkpoint matching teacher keyword
                for ckpt in teachers_dir.glob(f"*{teacher_name}*/{dataset_name}/finetuned_teacher_best"):
                    if ckpt.exists():
                        found_checkpoint = ckpt
                        break

        if found_checkpoint:
            model_path = str(found_checkpoint)
            logging.info(f"Using fine-tuned teacher checkpoint from: {model_path}")
        else:
            candidate_model = MODELS_DIR / teacher_name
            if candidate_model.exists():
                model_path = str(candidate_model)
            elif "instruct" in teacher_str and "recgpt" in teacher_str:
                candidate_recgpt = MODELS_DIR / "RecGPT-7B-Instruct"
                model_path = str(candidate_recgpt) if candidate_recgpt.exists() else "vinai/RecGPT-7B-Instruct"
            elif "recgpt" in teacher_str:
                candidate_recgpt = MODELS_DIR / "RecGPT-7B"
                model_path = str(candidate_recgpt) if candidate_recgpt.exists() else "vinai/RecGPT-7B"
            else:
                model_path = teacher_name
            logging.info(f"Using teacher model path: {model_path}")
    else:
        # If model_path is explicitly provided, align teacher_name if needed
        model_path_obj = Path(model_path)
        if teacher_name == "Qwen3.5-4B" and model_path_obj.parent.parent.name in [p.name for p in (CHECKPOINTS_DIR / "teachers").glob("*") if p.is_dir()]:
            teacher_name = model_path_obj.parent.parent.name

    logging.info(f"Loading saved ranking prompts from {prompt_file}...")
    samples = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    logging.info(f"Extracting teacher scores for {len(samples)} prompt samples in '{dataset_name}' using '{teacher_name}' (batch_size={batch_size})...")

    user_item_scores: Dict[str, Dict[str, float]] = {}

    if "recgpt" in teacher_str:
        try:
            from knowlight.teachers.recgpt.recgpt import RecGPTTeacher
            recgpt_inst = RecGPTTeacher(model_name_or_path=model_path, lazy_load=False)
            prompt_texts = [s["prompt"] for s in samples]
            probs = recgpt_inst._predict_batch_prompts(prompt_texts)
            for s, p in zip(samples, probs):
                u_id = str(s["user_id"])
                i_id = str(s["item_id"])
                if u_id not in user_item_scores:
                    user_item_scores[u_id] = {}
                user_item_scores[u_id][i_id] = float(p)
            del recgpt_inst
        except Exception as e:
            logging.warning(f"RecGPT score extraction encountered: {e}. Falling back to LLMScoreExtractor.")
            extractor = LLMScoreExtractor(model_path)
            prompt_texts = [s["prompt"] for s in samples]
            probs = extractor.extract_interaction_probabilities_batch(prompt_texts, batch_size=batch_size)
            for s, p in zip(samples, probs):
                u_id = str(s["user_id"])
                i_id = str(s["item_id"])
                if u_id not in user_item_scores:
                    user_item_scores[u_id] = {}
                user_item_scores[u_id][i_id] = float(p)
            del extractor
    else:
        extractor = LLMScoreExtractor(model_path)
        prompt_texts = [s["prompt"] for s in samples]
        probs = extractor.extract_interaction_probabilities_batch(prompt_texts, batch_size=batch_size)
        for s, p in zip(samples, probs):
            u_id = str(s["user_id"])
            i_id = str(s["item_id"])
            if u_id not in user_item_scores:
                user_item_scores[u_id] = {}
            user_item_scores[u_id][i_id] = float(p)
        del extractor

    # Cleanup extraction memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher_folder = "recgpt" if "recgpt" in teacher_str else teacher_name
    scores_subdir = cache_dir / teacher_folder / dataset_name / "scores"
    scores_subdir.mkdir(parents=True, exist_ok=True)

    if "recgpt" in teacher_str:
        out_path = scores_subdir / "recgpt_scores.pt"
    else:
        out_path = scores_subdir / f"{dataset_name}_llm_scores_cache.pt"
    
    torch.save(user_item_scores, out_path)

    logging.info(f"Successfully saved {len(user_item_scores)} user score profiles to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract Continuous LLM Interaction Probabilities from Saved JSONL Prompts")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name")
    parser.add_argument("--model_path", type=str, default=None, help="Fine-tuned LLM model path (optional)")
    parser.add_argument("--teacher", type=str, default="Qwen3.5-4B", help="Name of the LLM teacher for output directory structure")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size for score extraction")

    args = parser.parse_args()
    extract_dataset_llm_scores(args.dataset, args.model_path, teacher_name=args.teacher, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
