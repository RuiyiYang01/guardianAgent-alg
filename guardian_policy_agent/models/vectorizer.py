# guardian_policy_agent/models/vectorizer.py
import torch
import hashlib
from typing import List, Dict, Any, Optional

# === 更新后的词表 ===
VOCAB_DATA_CATEGORIES = [
    # 基础类型
    "Location", "Contact", "DeviceID", "IPAddress", "Demographic",
    "Health", "Financial", "Biometric", "Content", "BrowsingHistory",
    "AppUsage", "Files", "Camera", "Microphone", "Clipboard",

    # 新增 (适配 Guardian Capture)
    "Credentials",    # 对应 password
    "Authentication", # 对应 auth tokens
    "InputText",      # 通用文本输入

    # 新增 (适配 Network Event)
    "cookies", "identifiers", "online_identifiers"
]

VOCAB_ACTIONS = [
    "Collect", "Share", "Use", "Store", "Transfer", "Control", "Process",
    # 对应 guardian_capture 的 kind
    "paste", "selection", "input",
    # 对应 guardian_event 的 method/type
    "xmlhttprequest", "script", "sub_frame"
]

VOCAB_PURPOSES = [
    "Advertising", "Analytics", "Functionality", "Security",
    "Personalization", "Legal", "Marketing", "Unknown"
]

class SimpleFeatureEncoder:
    def __init__(self):
        self.data_map = {k: i for i, k in enumerate(VOCAB_DATA_CATEGORIES)}
        self.action_map = {k: i for i, k in enumerate(VOCAB_ACTIONS)}
        self.purpose_map = {k: i for i, k in enumerate(VOCAB_PURPOSES)}

        self.data_dim = len(VOCAB_DATA_CATEGORIES)
        self.action_dim = len(VOCAB_ACTIONS)
        self.purpose_dim = len(VOCAB_PURPOSES)

        self.input_dim = self.data_dim + self.action_dim + self.purpose_dim

    def _to_multihot(self, items: List[str], mapping: Dict[str, int], dim: int) -> torch.Tensor:
        vec = torch.zeros(dim, dtype=torch.float32)
        if not items:
            return vec

        for item in items:
            key = str(item).strip()
            # 1. 尝试直接匹配
            idx = mapping.get(key)
            # 2. 尝试不区分大小写匹配
            if idx is None:
                for map_k, map_v in mapping.items():
                    if map_k.lower() == key.lower():
                        idx = map_v
                        break
            # 3. 尝试包含匹配 (模糊)
            if idx is None:
                for map_k, map_v in mapping.items():
                    if map_k.lower() in key.lower():
                        idx = map_v
                        break

            if idx is not None:
                vec[idx] = 1.0
        return vec

    def vectorize(self, obj: Dict[str, Any]) -> torch.Tensor:
        d_vec = self._to_multihot(obj.get("data_categories", []), self.data_map, self.data_dim)
        a_vec = self._to_multihot(obj.get("actions", []), self.action_map, self.action_dim)
        p_vec = self._to_multihot(obj.get("purposes", []), self.purpose_map, self.purpose_dim)

        return torch.cat([d_vec, a_vec, p_vec])


class SentenceFeatureEncoder:
    """
    Encode behavior/policy dicts as dense 384-dim vectors using a frozen
    sentence transformer (all-MiniLM-L6-v2).

    Replaces the 42-dim multi-hot encoding with semantically rich embeddings,
    preserving the same vectorize() interface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 50000):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        self.input_dim = self.model.get_sentence_embedding_dimension()
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_size = cache_size
        # Compatibility with code that checks for data_map
        self.data_map = {k: i for i, k in enumerate(VOCAB_DATA_CATEGORIES)}
        self.action_map = {k: i for i, k in enumerate(VOCAB_ACTIONS)}
        self.purpose_map = {k: i for i, k in enumerate(VOCAB_PURPOSES)}

    def _dict_to_text(self, obj: Dict[str, Any]) -> str:
        """Convert a structured dict to a natural language description.
        If a 'raw_text' key is present, use it directly for richer semantics."""
        raw = obj.get("raw_text")
        if raw:
            return raw
        parts = []
        cats = obj.get("data_categories", [])
        if cats:
            parts.append("Data: " + ", ".join(str(c) for c in cats))
        acts = obj.get("actions", [])
        if acts:
            parts.append("Actions: " + ", ".join(str(a) for a in acts))
        purps = obj.get("purposes", [])
        if purps:
            parts.append("Purposes: " + ", ".join(str(p) for p in purps))
        return ". ".join(parts) if parts else "empty"

    def vectorize(self, obj: Dict[str, Any]) -> torch.Tensor:
        text = self._dict_to_text(obj)
        # Cache by text hash
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        vec = self.model.encode(text, convert_to_tensor=True).cpu().float()
        if len(self._cache) < self._cache_size:
            self._cache[key] = vec
        return vec

    def vectorize_batch(self, objs: List[Dict[str, Any]]) -> torch.Tensor:
        """Batch encode for training efficiency."""
        texts = [self._dict_to_text(o) for o in objs]
        vecs = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return vecs.cpu().float()