import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import json
import re
import numpy as np
import torch
from typing import List, Dict, Any, Optional

from knowlight.teachers.base_teacher import BaseTeacher
from knowlight.prompts.prompt_templates import PromptTemplates


class LLMTeacher(BaseTeacher):
    """
    LLM Semantic Teacher for Recommendation Graph Instruction Tuning, Feature Extraction, and Online Inference.
    Supports any Causal LLM architecture (e.g. Qwen, LLaMA, Mistral).
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config.get("llm", {})
        self.device = device
        self.use_mock = self.config.get("use_mock", True)
        self.llm_dim = self.config.get("hidden_size", 4096)  # Dynamic fallback hidden state dimension

        self.model = None
        self.tokenizer = None

        if not self.use_mock:
            self._init_real_model()

    def _init_real_model(self):
        """Initialize real HuggingFace model or Fine-Tuned Checkpoint."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from knowlight.paths import MODELS_DIR
            from pathlib import Path
            model_name = self.config.get("model_name_or_path", "Qwen/Qwen3.5-4B")

            # Resolve local model directory if present
            target_path = Path(model_name)
            if not target_path.exists():
                candidate_models = MODELS_DIR / model_name
                if candidate_models.exists():
                    target_path = candidate_models
                else:
                    repo_name = Path(model_name).name
                    candidate_repo = MODELS_DIR / repo_name
                    if candidate_repo.exists():
                        target_path = candidate_repo

            is_local = target_path.exists() and target_path.is_dir()
            model_identifier = str(target_path) if is_local else model_name

            print(f"[LLM Teacher] Loading tokenizer and model from: {model_identifier} (local_files_only={is_local})")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_identifier,
                local_files_only=is_local,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            quantization = self.config.get("quantization", None)
            if quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_identifier,
                    quantization_config=quant_config,
                    device_map=self.config.get("device_map", "auto"),
                    local_files_only=is_local,
                    trust_remote_code=True
                )
            else:
                compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_identifier,
                    torch_dtype=compute_dtype,
                    device_map=self.config.get("device_map", "auto"),
                    local_files_only=is_local,
                    trust_remote_code=True
                )
            
            # Dynamically update llm_dim from the actual loaded model config
            if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
                self.llm_dim = self.model.config.hidden_size

            self.model.eval()
            print(f"[LLM Teacher] Model loaded successfully (hidden_size={self.llm_dim}).")
        except Exception as e:
            print(f"[LLM Teacher Warning] Failed to load real model weights ({e}). Falling back to Mock Mode.")
            self.use_mock = True

    def extract_features(self, inputs: List[str]) -> torch.Tensor:
        """BaseTeacher compliance alias for extract_hidden_states."""
        return self.extract_hidden_states(inputs)

    def predict_ranking(self, user_profile: Any, candidate_items: List[Any]) -> Dict[str, float]:
        """BaseTeacher compliance alias for predict_candidate_ranking."""
        return self.predict_candidate_ranking(
            user_attributes=str(user_profile),
            user_history="User history",
            neighbors_1="Neighbors 1",
            neighbors_2="Neighbors 2",
            candidate_items=candidate_items
        )

    def extract_hidden_states(self, prompts: List[str]) -> torch.Tensor:
        """
        Extracts last-token hidden states for a batch of prompts.
        Used by MultiHopFeatureExtractor in Step 5.
        """
        if self.use_mock:
            batch_size = len(prompts)
            embeddings = []
            for prompt in prompts:
                seed = abs(hash(prompt)) % (2**32)
                rng = np.random.RandomState(seed)
                emb = rng.randn(self.llm_dim).astype(np.float32)
                embeddings.append(emb)
            return torch.tensor(np.array(embeddings), dtype=torch.float32, device=self.device)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 1024)
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]  # Shape: (B, seq_len, llm_dim)
            
            # Extract representation of last valid token before padding
            attention_mask = inputs["attention_mask"]
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden_state.size(0), device=self.device)
            last_token_embeds = last_hidden_state[batch_indices, seq_lengths]

        return last_token_embeds

    def predict_candidate_ranking(
        self,
        user_attributes: str,
        user_history: str,
        neighbors_1: str,
        neighbors_2: str,
        candidate_items: List[Any]
    ) -> Dict[str, float]:
        """
        Online Inference: Candidate Ranking Prediction.
        Evaluates relevance scores {item_id: score} for candidate items given multi-hop context.
        """
        cand_str = ", ".join([str(c) for c in candidate_items])
        prompt = PromptTemplates.build_ranking_prompt(
            user_attributes=user_attributes,
            user_history=user_history,
            neighbors_1=neighbors_1,
            neighbors_2=neighbors_2,
            candidate_items=cand_str
        )

        if self.use_mock:
            scores = {}
            for idx, item_id in enumerate(candidate_items):
                score = float(np.round(1.0 / (idx + 1.5) + np.random.uniform(0.01, 0.1), 4))
                scores[str(item_id)] = min(score, 1.0)
            return scores

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._parse_json_dict(text)

    def predict_edge_affinity(
        self,
        entity_a_type: str = "USER",
        entity_a_text: str = "User Profile",
        entity_a_meta: str = "User History",
        entity_b_type: str = "ITEM",
        entity_b_text: str = "Item Attributes",
        entity_b_meta: str = "Item Metadata",
        a_neighbors_1: str = "None",
        b_neighbors_1: str = "None",
        a_neighbors_2: str = "None",
        b_neighbors_2: str = "None"
    ) -> Dict[str, float]:
        """
        Online Inference: Structural Edge Affinity Prediction (Link Prediction).
        Returns: {"structural_affinity": float}
        """
        prompt = PromptTemplates.build_structural_affinity_prompt(
            entity_a_type=entity_a_type,
            entity_a_text=entity_a_text,
            entity_a_meta=entity_a_meta,
            entity_b_type=entity_b_type,
            entity_b_text=entity_b_text,
            entity_b_meta=entity_b_meta,
            a_neighbors_1=a_neighbors_1,
            b_neighbors_1=b_neighbors_1,
            a_neighbors_2=a_neighbors_2,
            b_neighbors_2=b_neighbors_2
        )

        if self.use_mock:
            seed = abs(hash(prompt)) % (2**32)
            rng = np.random.RandomState(seed)
            return {
                "structural_affinity": float(np.round(rng.uniform(0.3, 0.95), 4))
            }

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._parse_json_dict(text)

    def _parse_json_dict(self, text: str) -> dict:
        """Extract JSON dictionary from model output text using regex."""
        try:
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}


