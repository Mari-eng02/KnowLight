import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.paths import RAW_DATA_DIR, CACHE_DIR, TEACHERS_CHECKPOINTS_DIR, MODELS_DIR, dataset_directory

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def resolve_model_path(model_path: str):
    """
    Resolves model_path to a local directory if present in MODELS_DIR or absolute path.
    Returns (model_identifier, is_local_boolean).
    """
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
    return model_identifier, is_local


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes masked mean pooling over hidden states using attention mask.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    sum_embeds = (last_hidden_state * mask).sum(dim=1)
    count_embeds = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeds / count_embeds


class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (InfoNCE) for SCFT.
    Computes bidirectional InfoNCE loss between pair of augmented item views X and Y.
    """
    def __init__(self, tau: float = 0.2):
        super().__init__()
        self.tau = tau

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        loss_xy = self._compute_cl(X, Y)
        loss_yx = self._compute_cl(Y, X)
        return loss_xy + loss_yx

    def _compute_cl(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        bs = X.size(0)
        X_norm = F.normalize(X.float(), dim=-1, eps=1e-8)
        Y_norm = F.normalize(Y.float(), dim=-1, eps=1e-8)

        sim_matrix = torch.matmul(X_norm, Y_norm.T) / self.tau
        labels = torch.arange(bs, device=X.device)
        return F.cross_entropy(sim_matrix, labels)


def extract_item_embeddings_from_llm(
    model_name_or_path: str,
    jsonline_file: Path,
    output_path: Path,
    peft_path: Optional[Path] = None,
    pool_type: str = "avg",
    batch_size: int = 32,
    max_length: int = 512,
    pca_dim: Optional[int] = 2048,
    out_dim: int = 64,
    device: str = "cuda"
) -> np.ndarray:
    """
    Loads an LLM (Qwen/LLaMA/BERT/etc.), applies LoRA weights,
    computes item text embeddings for item_str_llmemb.jsonline, and saves output array.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("HuggingFace 'transformers' library is required for SCFT extraction.")

    if not torch.cuda.is_available() and device == "cuda":
        logging.warning("CUDA not available, falling back to CPU for embedding extraction.")
        device = "cpu"

    model_identifier, is_local = resolve_model_path(model_name_or_path)
    logging.info(f"Loading LLM model from '{model_identifier}' (local_files_only={is_local}) on device {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_identifier,
        local_files_only=is_local,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32

    model = AutoModel.from_pretrained(
        model_identifier,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        local_files_only=is_local,
        device_map="auto" if device == "cuda" else None
    )

    if peft_path and peft_path.exists():
        try:
            # pyrefly: ignore [missing-import]
            from peft import PeftModel
            logging.info(f"Loading LoRA fine-tuned weights from {peft_path}...")
            model = PeftModel.from_pretrained(model, peft_path)
        except Exception as e:
            logging.warning(f"Could not load PEFT weights from {peft_path}: {e}")

    model.eval()

    # Load item prompt jsonline file
    items_data = []
    with open(jsonline_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items_data.append(json.loads(line))

    # Sort by item_id (1-indexed)
    items_data.sort(key=lambda x: int(x.get("item_id", 0)))
    num_items = len(items_data)
    logging.info(f"Extracting LLM embeddings for {num_items} items (pool_type='{pool_type}')...")

    all_embeddings = []

    for i in tqdm(range(0, num_items, batch_size), desc="Extracting LLM Embeddings"):
        batch = items_data[i:i + batch_size]
        texts = [b["input"] for b in batch]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

            if pool_type == "avg":
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                sum_embeds = (hidden_states * mask).sum(dim=1)
                count_embeds = mask.sum(dim=1).clamp(min=1e-9)
                pooled = sum_embeds / count_embeds
            elif pool_type == "cls":
                pooled = hidden_states[:, 0, :]
            else:  # last token
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lens]

            all_embeddings.append(pooled.cpu().float().numpy())

    item_embeddings_matrix = np.concatenate(all_embeddings, axis=0)
    logging.info(f"Extracted raw LLM item embeddings matrix of shape: {item_embeddings_matrix.shape}")

    # Apply Level 1 PCA reduction to intermediate dimension (e.g. 1536)
    # The Level 2 MLP (W1, W2, b1, b2) will be trained dynamically during distillation.
    target_pca_dim = 1536
    logging.info(f"Applying Level 1 PCA reduction from raw LLM dim ({item_embeddings_matrix.shape[1]}) to intermediate dim ({target_pca_dim})...")
    from knowlight.teachers.llmemb.rat_adapter import apply_pca_reduction
    intermediate_tensor = apply_pca_reduction(item_embeddings_matrix, pca_dim=target_pca_dim)
    intermediate_embeddings = intermediate_tensor.cpu().numpy().astype(np.float32)

    logging.info(f"Intermediate PCA item embeddings shape: {intermediate_embeddings.shape}")

    # Save intermediate embeddings to destination
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, intermediate_embeddings)
    
    # Also save .pkl format for compatibility
    pkl_path = output_path.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(intermediate_embeddings, f)

    logging.info(f"Saved intermediate {intermediate_embeddings.shape[1]}-dim item embeddings to:\n - {output_path}\n - {pkl_path}")
    return intermediate_embeddings


def run_scft_fine_tuning(
    dataset_name: str,
    model_name_or_path: str,
    output_dir: Path,
    lora_rank: int = 32,
    learning_rate: float = 2e-4,
    max_steps: int = 2000,
    batch_size: int = 8,
    tau: float = 0.1,
    max_length: int = 256,
    pool_type: str = "avg"
):
    """
    Performs Supervised Contrastive Fine-Tuning (SCFT) with LoRA on LLM using item prompts.
    """
    try:
        # pyrefly: ignore [missing-import]
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoTokenizer, AutoModel
        from torch.optim import AdamW
    except ImportError:
        raise ImportError("PEFT and Transformers are required for SCFT fine-tuning.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Starting SCFT Fine-Tuning on dataset '{dataset_name}' with model '{model_name_or_path}'...")

    raw_handled_dir = RAW_DATA_DIR / dataset_name / "handled"
    jsonline_file = raw_handled_dir / "item_str_llmemb.jsonline"

    if not jsonline_file.exists():
        raise FileNotFoundError(
            f"Item prompts file not found at {jsonline_file}. "
            f"Please run 'python knowlight/datasets/prepare_llmemb_data.py --dataset {dataset_name}' first."
        )

    # 1. Load Model & Tokenizer
    model_identifier, is_local = resolve_model_path(model_name_or_path)
    logging.info(f"Loading LLM model from '{model_identifier}' (local_files_only={is_local})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_identifier,
        local_files_only=is_local,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Detect GPU precision support for bfloat16 vs float32 (FP16 without scaler leads to NaN overflow)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
    logging.info(f"SCFT Precision setup -> bf16: {use_bf16}, dtype: {compute_dtype}")

    base_model = AutoModel.from_pretrained(
        model_identifier,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        local_files_only=is_local
    ).to(device)

    # Enable Gradient Checkpointing to drastically reduce activation VRAM
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        logging.info("Enabled Gradient Checkpointing for memory-efficient SCFT training.")

    # 2. Setup LoRA Config
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 3. Load item texts
    items_data = []
    with open(jsonline_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items_data.append(json.loads(line)["input"])

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = ContrastiveLoss(tau=tau)

    grad_accum_steps = max(1, 32 // max(1, batch_size))
    logging.info(
        f"Training SCFT LoRA for {max_steps} steps "
        f"(batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, effective_batch_size={batch_size * grad_accum_steps})..."
    )

    model.train()
    step = 0
    epoch = 0
    accum_count = 0
    optimizer.zero_grad()

    while step < max_steps:
        epoch += 1
        np.random.shuffle(items_data)
        for i in range(0, len(items_data) - batch_size, batch_size):
            batch_texts = items_data[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

            # Two passes for dropout-based contrastive view augmentation
            out1_raw = model(**inputs).last_hidden_state
            out2_raw = model(**inputs).last_hidden_state

            out1 = mean_pooling(out1_raw, inputs["attention_mask"])
            out2 = mean_pooling(out2_raw, inputs["attention_mask"])

            loss = loss_fn(out1, out2)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()
            accum_count += 1

            if accum_count % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 100 == 0 or step == max_steps:
                    logging.info(f"Step [{step}/{max_steps}] - SCFT Contrastive Loss: {loss.item():.4f}")

            if step >= max_steps:
                break

    # 4. Save LoRA model checkpoint to data/checkpoints/teachers/llmemb/{dataset}/
    save_path = TEACHERS_CHECKPOINTS_DIR / "llmemb" / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    logging.info(f"SCFT LoRA fine-tuning completed. Model checkpoint saved to: {save_path}")

    # 5. Automatically extract item embeddings with fine-tuned model and save directly to cache
    extracted_path = CACHE_DIR / "llmemb" / dataset_name / "item_emb_llmemb.npy"
    extract_item_embeddings_from_llm(
        model_name_or_path=model_name_or_path,
        jsonline_file=jsonline_file,
        output_path=extracted_path,
        peft_path=save_path,
        pool_type=pool_type,
        batch_size=batch_size,
        max_length=max_length,
        device=device
    )

    # 6. Automatically compute and cache user barycenter embeddings
    from knowlight.datasets.data_loader import KnowLightDataLoader
    from knowlight.teachers.llmemb.llmemb_teacher import LLMEmbTeacher
    logging.info(f"Precomputing and caching user barycenter embeddings for dataset '{dataset_name}'...")
    data_loader = KnowLightDataLoader(dataset_name=dataset_name)
    data_loader.load_data()
    teacher = LLMEmbTeacher(dataset_name=dataset_name)
    teacher.load_item_embeddings(num_items=data_loader.num_items)
    teacher.load_user_embeddings(num_users=data_loader.num_users, train_dict=data_loader.train_dict, force_recompute=True)



def main():
    parser = argparse.ArgumentParser(description="KnowLight SCFT Fine-Tuning & Embedding Extractor (LLMEmb)")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-4B", help="LLM base model name or path")
    parser.add_argument("--do_train", action="store_true", help="Execute SCFT LoRA fine-tuning and automatically extract embeddings")
    parser.add_argument("--do_predict", action="store_true", help="Extract item embeddings (using saved checkpoint if available)")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8 with gradient accumulation)")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature hyperparameter for SCFT Contrastive Loss")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for item text tokenization")
    parser.add_argument("--pool_type", type=str, default="avg", choices=["avg", "cls", "last"], help="Pooling type")

    args = parser.parse_args()

    checkpoint_dir = TEACHERS_CHECKPOINTS_DIR / "llmemb" / args.dataset
    cache_dir = CACHE_DIR / "llmemb" / args.dataset
    jsonline_file = RAW_DATA_DIR / args.dataset / "handled" / "item_str_llmemb.jsonline"
    extracted_path = cache_dir / "item_emb_llmemb.npy"

    print("=" * 70)
    print(" KnowLight LLMEmb SCFT Fine-Tuning & Embedding Extractor ")
    print(f" Dataset: {args.dataset} | Model: {args.model_name_or_path}")
    print("=" * 70)

    if args.do_train:
        run_scft_fine_tuning(
            dataset_name=args.dataset,
            model_name_or_path=args.model_name_or_path,
            output_dir=checkpoint_dir,
            lora_rank=args.lora_rank,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            tau=args.tau,
            max_length=args.max_length,
            pool_type=args.pool_type
        )
    elif args.do_predict:
        if not jsonline_file.exists():
            raise FileNotFoundError(f"JSONL prompts file not found: {jsonline_file}")
        
        # Check if saved LoRA checkpoint exists
        peft_checkpoint = checkpoint_dir if (checkpoint_dir / "adapter_config.json").exists() else None
        if peft_checkpoint:
            print(f" Found saved LoRA checkpoint at {peft_checkpoint}. Extracting embeddings with fine-tuned model...")
        else:
            print(f" Extracting embeddings using base model '{args.model_name_or_path}'...")

        extract_item_embeddings_from_llm(
            model_name_or_path=args.model_name_or_path,
            jsonline_file=jsonline_file,
            output_path=extracted_path,
            peft_path=peft_checkpoint,
            pool_type=args.pool_type,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        # Also trigger user barycenter calculation
        from knowlight.datasets.data_loader import KnowLightDataLoader
        from knowlight.teachers.llmemb.llmemb_teacher import LLMEmbTeacher
        data_loader = KnowLightDataLoader(dataset_name=args.dataset)
        data_loader.load_data()
        teacher = LLMEmbTeacher(dataset_name=args.dataset)
        teacher.load_item_embeddings(num_items=data_loader.num_items)
        teacher.load_user_embeddings(num_users=data_loader.num_users, train_dict=data_loader.train_dict)
    else:
        print("\nPlease specify either --do_train or --do_predict.")
        print("Example:")
        print("  python knowlight/teachers/llmemb/train_scft.py --dataset movielens --do_train --model_name_or_path Qwen/Qwen2.5-0.5B")


if __name__ == "__main__":
    main()
