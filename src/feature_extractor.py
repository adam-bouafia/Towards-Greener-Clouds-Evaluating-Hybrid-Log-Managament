# src/feature_extractor.py
from __future__ import annotations

import os, hashlib, threading, shelve
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ---- OpenVINO toggle (optional, GPU on Iris Xe) ----
USE_OV = os.environ.get("EMBED_BACKEND", "").lower() == "openvino"
if USE_OV:
    try:
        from optimum.intel.openvino import OVModelForFeatureExtraction
    except Exception as _e:
        print(f"[EMBED] OpenVINO requested but import failed ({_e}); falling back to PyTorch.")
        USE_OV = False

# ---- Cache + model config (env-tunable) ----
CACHE_DIR = os.environ.get("EMBED_CACHE_DIR", "./.cache")
CACHE_DB = os.path.join(CACHE_DIR, "embeddings.db")
CACHE_MEM_CAP = int(os.environ.get("EMBED_CACHE_MAX", "50000"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "teoogherghi/Log-Analysis-Model-DistilBert")
EMBED_TRUNC = int(os.environ.get("EMBED_TRUNC", "256"))  # max tokens

# namespace cache by model+trunc so swapping either doesn't reuse stale vectors
_CACHE_NS = f"{EMBED_MODEL}|trunc={EMBED_TRUNC}"

def _norm_key(text: str) -> str:
    """Normalize whitespace, include cache namespace, and hash -> stable key."""
    t = " ".join((text or "").split())
    return hashlib.sha1((_CACHE_NS + "\n" + t).encode("utf-8")).hexdigest()

class _DiskLRUCache:
    """
    Shelve-backed LRU cache:
      - in-memory LRU for hot items
      - persistent shelve DB for cold items
    Stores/returns np.float32 vectors. Thread-safe for our usage.
    """
    def __init__(self, path=CACHE_DB, mem_cap=CACHE_MEM_CAP):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = shelve.open(path, flag="c", writeback=False)
        self.mem, self.order, self.mem_cap = {}, [], int(mem_cap)
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.mem:
                try: self.order.remove(key)
                except ValueError: pass
                self.order.append(key)
                return self.mem[key]
            if key in self.db:
                arr = self.db[key]
                self._mem_put(key, arr)
                return arr
            return None

    def set(self, key: str, value: np.ndarray):
        with self.lock:
            value = np.asarray(value, dtype=np.float32)
            self._mem_put(key, value)
            self.db[key] = value
            self.db.sync()

    def _mem_put(self, key: str, value: np.ndarray):
        # avoid duplicate entries in LRU order
        if key in self.mem:
            try: self.order.remove(key)
            except ValueError: pass
        self.mem[key] = value
        self.order.append(key)
        if len(self.mem) > self.mem_cap:
            try:
                old = self.order.pop(0)
                self.mem.pop(old, None)
            except IndexError:
                pass

    def close(self):
        try: self.db.close()
        except Exception: pass

    def __del__(self):
        self.close()

def _pick_device():
    # CUDA likely absent on Iris Xe, but keep for portability.
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LogFeatureExtractor:
    """
    768-d (or model hidden size) embedding with:
      - OpenVINO GPU/CPU (if EMBED_BACKEND=openvino), otherwise PyTorch
      - LRU + disk cache keyed by SHA1(content + model + truncation)
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.cache = _DiskLRUCache()

        if USE_OV:
            device = os.environ.get("EMBED_DEVICE", "GPU")  # "GPU" or "CPU"
            try:
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    EMBED_MODEL, export=True, device=device
                )
                self._backend = "ov"
                print(f"[EMBED] Using OpenVINO on {device}")
            except Exception as e:
                print(f"[EMBED] OpenVINO init failed ({e}); falling back to PyTorch.")
                self._init_torch()
        else:
            self._init_torch()

        # pick hidden size from config when available (OV + Torch both expose .config)
        self._dim = int(getattr(getattr(self, "model", None), "config", None).hidden_size
                        if getattr(getattr(self, "model", None), "config", None) is not None
                        else 768)

    def _init_torch(self):
        self.model = AutoModel.from_pretrained(EMBED_MODEL)
        self.model.eval()
        self.device = _pick_device()
        self.model.to(self.device)
        self._backend = "torch"
        print(f"[EMBED] Using PyTorch on {self.device.type}")

    def get_embedding(self, text: str) -> np.ndarray:
        key = _norm_key(text)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        if not text:
            emb = np.zeros((self._dim,), dtype=np.float32)
            self.cache.set(key, emb)
            return emb

        toks = self.tokenizer(text, truncation=True, max_length=EMBED_TRUNC,
                              padding=False, return_tensors="pt")

        if self._backend == "ov":
            with torch.no_grad():
                out = self.model(**toks)  # OV returns torch tensors too
                reps = out.last_hidden_state.mean(dim=1)
                emb = reps.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            with torch.no_grad():
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = self.model(**toks)
                reps = out.last_hidden_state.mean(dim=1)
                emb = reps.squeeze(0).cpu().numpy().astype(np.float32)

        self.cache.set(key, emb)
        return emb

    def close(self):
        self.cache.close()
