import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Suppress expected library warnings to clean up log output
warnings.filterwarnings("ignore", category=UserWarning, module="peft.*")
warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*Both `max_new_tokens`.*")

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.paths import PROMPTS_DIR, CACHE_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class LLMIntentExtractor:
    """
    Extracts high-level latent user intents, taste profiles, and preferred genres
    from 1-hop and 2-hop graph history context using instruction-tuned LLMs (e.g. Qwen3.5-4B, LLaMA 3).
    Supports multi-GPU chunking and distributed HPC execution (Leonardo CINECA).
    """
    tokenizer: Any
    model: Any
    model_dir: Path
    model_tag: str

    def __init__(
        self,
        model_name_or_path: str = "models/Qwen3.5-4B",
        device: Optional[Union[str, torch.device]] = None,
        quantization: Optional[str] = None,
        batch_size: int = 16
    ):
        self.model_name_or_path = model_name_or_path
        if isinstance(device, str):
            self.device = torch.device(device)
        elif device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.quantization = quantization
        self.batch_size = batch_size

        self._resolve_model_path()
        self._load_model()

    def _resolve_model_path(self):
        target = Path(self.model_name_or_path)
        if target.exists() and target.is_dir():
            self.model_dir = target
            self.model_tag = target.name
        else:
            cand = MODELS_DIR / target.name
            if cand.exists() and cand.is_dir():
                self.model_dir = cand
                self.model_tag = cand.name
            else:
                self.model_dir = target
                self.model_tag = target.name.replace("/", "_")

    def _load_model(self):
        logging.info(f"[LLM Intent Extractor] Loading tokenizer and model from '{self.model_dir}' on {self.device}...")
        is_local = self.model_dir.exists() and self.model_dir.is_dir()
        model_id = str(self.model_dir) if is_local else self.model_name_or_path

        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=is_local
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure left-padding for causal LM batched generation
        self.tokenizer.padding_side = "left"

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map=str(self.device) if "cuda" in str(self.device) else "auto",
                trust_remote_code=True,
                local_files_only=is_local
            )
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map=str(self.device) if "cuda" in str(self.device) else "auto",
                trust_remote_code=True,
                local_files_only=is_local
            )
        else:
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            # For multi-GPU chunking, place model on the assigned device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=compute_dtype,
                trust_remote_code=True,
                local_files_only=is_local
            ).to(self.device)

        self.model.eval()
        logging.info(f"[LLM Intent Extractor] Model successfully loaded on {self.device}.")

    def _format_chat_prompt(self, user_context_prompt: str) -> str:
        system_instruction = (
            "You are an expert recommender system profiling assistant. "
            "Analyze the user's item interaction history and collaborative graph neighborhood context across items and communities. "
            "Synthesize their underlying latent tastes, category/domain preferences, and thematic tendencies in 2-3 concise, clear natural language sentences. "
            "Do not list individual item titles, product names, or raw metadata tags. Provide only the high-level qualitative user profile synthesis."
        )

        user_content = (
            f"User Interaction History & Graph Neighborhood:\n"
            f"{user_context_prompt}\n\n"
            f"Instruction: Based on the interaction history and collaborative community context above, synthesize this user's underlying latent preferences, preferred categories/attributes, and behavioral patterns in 2-3 concise, natural language sentences."
        )

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt_str + "This user "
        else:
            return (
                f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\nThis user "
            )

    @staticmethod
    def _clean_intent_text(text: str) -> str:
        """
        Strips internal thinking blocks, CoT scratchpads, bullet lists, prompt echoes,
        disclaimers/notes, and incomplete trailing sentences across all domains.
        """
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>")[-1].strip()
        if "<think>" in cleaned:
            cleaned = cleaned.split("<think>")[0].strip()

        # 1. Remove instruction header / echo artifacts
        for marker in ["### Assistant:", "### Assistant", "### Response:", "###", "assistant\n", "Assistant:", "Assistant", "<|im_start|>assistant"]:
            if cleaned.startswith(marker):
                cleaned = cleaned[len(marker):].strip()
        cleaned = cleaned.split("###")[0].strip()

        # 2. Remove trailing disclaimers, notes, and rewrite duplicates
        disclaimer_patterns = [
            r"\n+(?:please\s+)?note(?:\s+that)?[:\s\(\[].*$",
            r"\n+\(note[:\s].*?\)$",
            r"\n+\(since\s+there\s+is\s+no\s+specific.*?\)$",
            r"\n+(?:here\s+is\s+a\s+rewritten|here\s+are\s+(?:two|three|\d+)\s+(?:concise|short)).*$"
        ]
        for dp in disclaimer_patterns:
            cleaned = re.sub(dp, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

        # 3. If there is an explicit summary/conclusion at the end (e.g. Qwen CoT finishing with Overall/In summary)
        summary_match = re.search(r"\n+(?:in\s+summary|in\s+conclusion|overall|synthesized\s+profile)[:,\s]+(this\s+user.*)", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if summary_match and len(summary_match.group(1).strip()) > 50:
            cleaned = summary_match.group(1).strip()

        # 4. Remove prompt-echo artifacts if model repeated instruction
        echo_patterns = [
            r"^please (provide|synthesize|analyze).*?(\n\n|\.\s+)",
            r"^based on (the|this) (provided )?(data|user|information|analysis).*?:\s*",
            r"^here is (the|a) (concise |high-level |qualitative )?(synthesized )?(user )?profile.*?:?\s*",
            r"^the user'?s (underlying )?(latent )?(tastes|profile).*?can be summarized as follows:?\s*"
        ]
        for pat in echo_patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

        # 5. Filter out scratchpad / bullet lines (e.g. "* Item 1", "**User 102:**", "Let's analyze...")
        metadata_prefixes = (
            "title:", "genres:", "genre:", "category:", "categories:", "brand:", "price:",
            "res:", "item:", "description:", "stars:", "review:", "rating:", "user ",
            "direct preferences:", "collaborative filtering", "name:", "city:", "state:",
            "similar users", "indirectly interacted", "indirect interactions", "space opera;",
            "fantasy|sci-fi", "rama|fantasy", "action|adventure", "drama|thriller"
        )
        metadata_contains = (
            " | genres:", " | geners:", " | genes:", " | category:", " | brand:", " | description:",
            " | price:", " | rating:", " | stars:", "interacted with:"
        )

        lines = cleaned.split("\n")
        valid_lines = []
        for line in lines:
            l_str = line.strip()
            l_lower = l_str.lower()
            if not l_str:
                continue
            if l_str.startswith("*") or l_str.startswith("-") or l_str.startswith("•"):
                continue
            if re.match(r"^\*\*\s*user\s+\d+.*?\*\*", l_lower) or re.match(r"^user\s+\d+:", l_lower):
                continue
            if "let's analyze" in l_lower or "let's look at" in l_lower:
                continue
            if any(l_lower.startswith(prefix) for prefix in metadata_prefixes):
                continue
            if any(marker in l_lower for marker in metadata_contains):
                continue
            if l_str.endswith("| Genes") or l_str.endswith("| Geners") or l_str.endswith("| Genres"):
                continue
            valid_lines.append(l_str)

        cleaned = " ".join(valid_lines).strip()

        # 6. Remove leftover boilerplate intro
        cleaned = re.sub(r"^(?:this\s+user\s+\d+\s+has\s+a\s+diverse\s+but\s+specific\s+set\s+of\s+interactions\.\s*)", "", cleaned, flags=re.IGNORECASE).strip()

        # 7. Normalize start
        if cleaned and not (cleaned.lower().startswith("this user") or cleaned.lower().startswith("the user")):
            cleaned = "This user " + cleaned

        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        # 8. Truncate incomplete trailing sentences (if cut off mid-sentence by token limit)
        last_punct = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
        if last_punct > 40:
            cleaned = cleaned[:last_punct + 1].strip()

        return cleaned

    def generate_intent_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05
    ) -> List[str]:
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        formatted_prompts = [self._format_chat_prompt(p) for p in prompts]

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0.0 else False,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        input_len = inputs.input_ids.shape[1]
        results = []
        for out in output_ids:
            gen_tokens = out[input_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            # Because prompt ended with "This user ", re-attach it to decoded completion
            if not text.startswith("This user") and not text.startswith("The user"):
                text = "This user " + text
            cleaned = self._clean_intent_text(text)
            results.append(cleaned)

        return results

    def extract_text_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512
    ) -> np.ndarray:
        orig_pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]  # (B, seq_len, dim)
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            all_embs.append(pooled.cpu().to(torch.float32).numpy())

        self.tokenizer.padding_side = orig_pad_side
        return np.vstack(all_embs).astype(np.float32)

    def extract_dataset_user_intents(
        self,
        dataset_name: str,
        output_dir: Optional[Path] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        extract_embeddings: bool = True,
        chunk_id: Optional[int] = None,
        num_chunks: Optional[int] = None
    ) -> Path:
        """
        Extracts User Intent CoT reasoning from {dataset_name}_user_intent_prompts.jsonl.
        Supports multi-GPU chunked parallel execution via chunk_id and num_chunks.
        """
        prompt_file = PROMPTS_DIR / f"{dataset_name}_user_intent_prompts.jsonl"
        if not prompt_file.exists():
            logging.info(f"User intent prompt file not found at {prompt_file}. Generating now...")
            from knowlight.prompts.prompt_builder import generate_user_intent_prompts
            generate_user_intent_prompts(dataset_name)

        out_dir = output_dir or (CACHE_DIR / self.model_tag / dataset_name / "intents")
        out_dir.mkdir(parents=True, exist_ok=True)

        is_chunked = chunk_id is not None and num_chunks is not None
        chunk_suffix = f"_chunk_{chunk_id}_of_{num_chunks}" if is_chunked else ""
        chunk_info = f" (Chunk {chunk_id + 1}/{num_chunks})" if is_chunked else ""

        json_path = out_dir / f"user_intents{chunk_suffix}.json"
        temp_json_path = out_dir / f"user_intents_temp{chunk_suffix}.json"

        user_intents = {}
        prior_completed_uids = set()
        prior_data = {}

        # 1. Recover any previously extracted users from global checkpoint if not chunked
        global_temp_path = out_dir / "user_intents_temp.json"
        global_final_path = out_dir / "user_intents.json"
        
        def is_valid_intent(val: Any) -> bool:
            if not isinstance(val, str):
                return False
            v = val.strip()
            if len(v) < 50:
                return False
            if v.lower().startswith("please provide") or v.lower().startswith("instruction:"):
                return False
            if v.endswith("| Genes") or v.endswith("| Geners") or v.endswith("| Genres"):
                return False
            return True

        if global_temp_path.exists():
            try:
                with open(global_temp_path, "r", encoding="utf-8") as f:
                    tdata = json.load(f)
                    for k, v in tdata.items():
                        if is_valid_intent(v):
                            prior_data[str(k)] = v
                            prior_completed_uids.add(str(k))
            except Exception:
                pass

        if global_final_path.exists():
            try:
                with open(global_final_path, "r", encoding="utf-8") as f:
                    fdata = json.load(f)
                    for k, v in fdata.items():
                        if is_valid_intent(v):
                            prior_data[str(k)] = v
                            prior_completed_uids.add(str(k))
            except Exception:
                pass

        # 2. Recover from chunk-specific temp file if resuming this chunk
        if is_chunked and temp_json_path.exists() and not json_path.exists():
            try:
                with open(temp_json_path, "r", encoding="utf-8") as f:
                    c_temp = json.load(f)
                    user_intents = {str(k): v for k, v in c_temp.items() if is_valid_intent(v)}
                logging.info(f"[{self.model_tag}]{chunk_info} Resuming intent extraction from chunk temp checkpoint ({len(user_intents)} valid users loaded).")
            except Exception as e:
                logging.warning(f"[{self.model_tag}]{chunk_info} Failed loading chunk temp checkpoint ({e}). Starting fresh.")
                user_intents = {}
        elif not is_chunked and prior_data:
            user_intents.update(prior_data)

        samples = []
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        if is_chunked:
            # 1. Deterministic assignment: each chunk owns the fixed subset from original dataset where idx % num_chunks == chunk_id
            chunk_all_samples = [s for idx, s in enumerate(samples) if idx % num_chunks == chunk_id]
            # 2. Filter out users that this chunk or prior runs have already completed
            pending_samples = [
                s for s in chunk_all_samples
                if str(s["user_id"]) not in prior_completed_uids and str(s["user_id"]) not in user_intents
            ]
            logging.info(
                f"[{self.model_tag}]{chunk_info} Assigned total {len(chunk_all_samples)} users for this chunk. "
                f"Remaining to compute: {len(pending_samples)} ({len(user_intents)} already completed in temp checkpoint)."
            )
        else:
            pending_samples = [
                s for s in samples
                if str(s["user_id"]) not in prior_completed_uids and str(s["user_id"]) not in user_intents
            ]
            logging.info(f"[{self.model_tag}] Assigned {len(pending_samples)} pending users out of {len(samples)} total ({len(user_intents)} in checkpoint).")

        # Dynamic Length Bucketing: Sort pending samples by length to minimize padding waste
        pending_samples.sort(key=lambda s: len(s["prompt"]))

        total_pending = len(pending_samples)
        for i in range(0, total_pending, self.batch_size):
            batch = pending_samples[i:i + self.batch_size]
            batch_prompts = [s["prompt"] for s in batch]
            batch_uids = [str(s["user_id"]) for s in batch]

            intents = self.generate_intent_batch(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

            for uid, cot in zip(batch_uids, intents):
                user_intents[uid] = cot

            if i % (self.batch_size * 5) == 0:
                torch.cuda.empty_cache()

            processed = len(user_intents)
            if (i // self.batch_size) % 5 == 0 or processed == (len(user_intents) + total_pending):
                logging.info(f"[{self.model_tag}]{chunk_info} Processed {len(user_intents)} users ({len(user_intents)} saved in temp checkpoint).")
                with open(temp_json_path, "w", encoding="utf-8") as f:
                    json.dump(user_intents, f, indent=2, ensure_ascii=False)

        # Save final complete chunk JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(user_intents, f, indent=2, ensure_ascii=False)

        if temp_json_path.exists():
            try:
                temp_json_path.unlink()
            except Exception:
                pass

        logging.info(f"[{self.model_tag}]{chunk_info} Successfully saved {len(user_intents)} User Intents to: {json_path}")

        # If single GPU execution and extract_embeddings requested, extract full embeddings now
        if not is_chunked and extract_embeddings:
            self.extract_and_save_dense_embeddings(dataset_name, out_dir=out_dir)

        return json_path

    def extract_and_save_dense_embeddings(self, dataset_name: str, out_dir: Optional[Path] = None) -> Path:
        """
        Extracts dense text embeddings from the full merged user_intents.json and saves user_intent_text_emb.npy.
        """
        target_dir = out_dir or (CACHE_DIR / self.model_tag / dataset_name / "intents")
        json_path = target_dir / "user_intents.json"
        npy_path = target_dir / "user_intent_text_emb.npy"

        if not json_path.exists():
            raise FileNotFoundError(f"Cannot extract embeddings: {json_path} does not exist.")

        with open(json_path, "r", encoding="utf-8") as f:
            user_intents = json.load(f)

        logging.info(f"[{self.model_tag}] Extracting dense text embeddings for {len(user_intents)} users from {json_path.name}...")
        user_ids = list(user_intents.keys())
        intent_texts = [user_intents[uid] for uid in user_ids]

        raw_embeddings = self.extract_text_embeddings(intent_texts, batch_size=self.batch_size * 2)
        emb_dim = raw_embeddings.shape[1]

        # Map user_ids to structured index matrix
        prompt_file = PROMPTS_DIR / f"{dataset_name}_user_intent_prompts.jsonl"
        mapped = False

        if prompt_file.exists():
            try:
                ordered_uids = []
                with open(prompt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            ordered_uids.append(str(json.loads(line).get("user_id", "")))
                if ordered_uids:
                    uid_to_idx = {uid: idx for idx, uid in enumerate(ordered_uids)}
                    num_total_users = max(len(ordered_uids), len(user_ids))
                    indexed_embeddings = np.zeros((num_total_users, emb_dim), dtype=np.float32)
                    for uid, emb in zip(user_ids, raw_embeddings):
                        idx = uid_to_idx.get(str(uid))
                        if idx is not None and idx < num_total_users:
                            indexed_embeddings[idx] = emb
                    mapped = True
                    logging.info(f"[{self.model_tag}] Aligned {len(user_ids)} User Intent embeddings with prompt order ({indexed_embeddings.shape}).")
            except Exception as e:
                logging.warning(f"[{self.model_tag}] Failed alignment via prompt file ({e}). Falling back.")

        if not mapped:
            try:
                int_uids = [int(u) for u in user_ids]
                max_uid = max(int_uids)
                indexed_embeddings = np.zeros((max_uid + 1, emb_dim), dtype=np.float32)
                for uid_int, emb in zip(int_uids, raw_embeddings):
                    indexed_embeddings[uid_int] = emb
            except (ValueError, TypeError):
                indexed_embeddings = raw_embeddings

        np.save(npy_path, indexed_embeddings)
        logging.info(f"[{self.model_tag}] Saved indexed User Intent text embeddings ({indexed_embeddings.shape}) to: {npy_path}")
        return npy_path

    @classmethod
    def merge_dataset_chunks(
        cls,
        model_name_or_path: str,
        dataset_name: str,
        output_dir: Optional[Path] = None,
        extract_embeddings: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 32
    ) -> Path:
        """
        Merges parallel GPU chunk files (user_intents_chunk_*.json) into the single final user_intents.json
        and extracts the unified user_intent_text_emb.npy embedding matrix.
        """
        model_tag = Path(model_name_or_path).name.replace("/", "_")
        out_dir = output_dir or (CACHE_DIR / model_tag / dataset_name / "intents")
        final_json = out_dir / "user_intents.json"

        chunk_files = sorted(out_dir.glob("user_intents_chunk_*.json"))
        temp_files = sorted(out_dir.glob("user_intents_temp_chunk_*.json"))
        global_temp = out_dir / "user_intents_temp.json"

        all_sources = chunk_files + temp_files
        if global_temp.exists():
            all_sources.append(global_temp)

        if not all_sources and not final_json.exists():
            raise FileNotFoundError(f"No chunk files or checkpoints found to merge in {out_dir}")

        merged_intents = {}

        # 1. Load existing final if present
        if final_json.exists():
            try:
                with open(final_json, "r", encoding="utf-8") as f:
                    merged_intents.update(json.load(f))
                logging.info(f"[Merge] Loaded {len(merged_intents)} existing intents from {final_json.name}")
            except Exception:
                pass

        # 2. Merge all chunk files
        for cf in all_sources:
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    merged_intents.update(cdata)
                    logging.info(f"[Merge] Merged {len(cdata)} intents from {cf.name}")
            except Exception as e:
                logging.warning(f"[Merge] Failed reading {cf.name}: {e}")

        # 3. Clean and normalize all merged intents (strips CoT, notes, and incomplete sentence ends)
        cleaned_intents = {}
        for uid, raw_text in merged_intents.items():
            if isinstance(raw_text, str):
                c_text = cls._clean_intent_text(raw_text)
                if len(c_text) >= 40:
                    cleaned_intents[str(uid)] = c_text

        # Check total dataset coverage
        prompt_file = PROMPTS_DIR / f"{dataset_name}_user_intent_prompts.jsonl"
        if prompt_file.exists():
            total_expected = sum(1 for line in open(prompt_file, "r", encoding="utf-8") if line.strip())
            logging.info(f"[Merge] Total unique users merged: {len(cleaned_intents)}/{total_expected} ({len(cleaned_intents) / max(total_expected, 1) * 100:.1f}%).")

        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(cleaned_intents, f, indent=2, ensure_ascii=False)

        logging.info(f"[Merge] Successfully saved final merged User Intents ({len(cleaned_intents)} users) to: {final_json}")

        # Clean up chunk temp files
        for tf in temp_files:
            try:
                tf.unlink()
            except Exception:
                pass

        if extract_embeddings:
            logging.info(f"[Merge] Initializing LLM teacher to extract dense text embeddings on merged data...")
            extractor = cls(
                model_name_or_path=model_name_or_path,
                device=device,
                batch_size=batch_size
            )
            extractor.extract_and_save_dense_embeddings(dataset_name, out_dir=out_dir)

        return final_json


def main():
    parser = argparse.ArgumentParser(description="Extract and Merge User Intent Profiles and Embeddings across Multi-GPU (HPC / Leonardo)")
    parser.add_argument("--model", type=str, default="models/Qwen3.5-4B", help="Model name or path in models/")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name (e.g. movielens, amazon_beauty, amazon_cd_2014)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for intent description and reasoning (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling (default: 0.9)")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty (default: 1.05)")
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "4bit", "8bit"], help="Quantization")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cuda:0, cuda:1, cpu)")
    parser.add_argument("--chunk_id", type=int, default=None, help="GPU chunk index (0 to num_chunks-1)")
    parser.add_argument("--num_chunks", type=int, default=None, help="Total number of parallel GPU chunks")
    parser.add_argument("--merge", action="store_true", help="Merge chunk files into final user_intents.json and extract embeddings")
    parser.add_argument("--no_embeddings", action="store_true", help="Skip extracting dense embeddings during merge/extraction")

    args = parser.parse_args()

    if args.merge:
        LLMIntentExtractor.merge_dataset_chunks(
            model_name_or_path=args.model,
            dataset_name=args.dataset,
            extract_embeddings=not args.no_embeddings,
            device=args.device,
            batch_size=args.batch_size
        )
        return

    quant = None if args.quantization == "none" else args.quantization
    extractor = LLMIntentExtractor(
        model_name_or_path=args.model,
        device=args.device,
        quantization=quant,
        batch_size=args.batch_size
    )

    extractor.extract_dataset_user_intents(
        dataset_name=args.dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        extract_embeddings=not args.no_embeddings,
        chunk_id=args.chunk_id,
        num_chunks=args.num_chunks
    )


if __name__ == "__main__":
    main()
