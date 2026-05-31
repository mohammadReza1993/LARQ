#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import gc
import math
import json
import time
import random
import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    GPTQConfig,
)


# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def move_batch_to_device_nonblocking(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def move_batch_to_cpu(batch):
    return {k: v.to("cpu", non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def maybe_to_device(x, device, dtype=None):
    if x is None:
        return None
    return x.to(device, dtype=dtype, non_blocking=True) if dtype else x.to(device, non_blocking=True)


def clear_feature_store(store):
    if hasattr(store, "clear"):
        store.clear()


def cleanup_model(model):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_").replace(" ", "_")


def ensure_gptq_model_dir(args):
    if args.quant_backend != "gptq":
        return
    safe_name = sanitize_model_id(args.model_id)
    if args.gptq_model_dir is None or str(args.gptq_model_dir).strip() == "":
        args.gptq_model_dir = os.path.join(args.gptq_root_dir, safe_name)
    os.makedirs(args.gptq_model_dir, exist_ok=True)


def append_run_result(results_file: str, row: dict):
    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    fieldnames = [
        "model_id", "arch_backend", "quant_backend",
        "attn_adapter_kind", "mlp_adapter_kind", "attn_r", "mlp_r",
        "trainable_parameters",
        "teacher_test_ppl", "student_test_ppl", "trained_adapter_model_test_ppl",
        "teacher_mmlu_acc", "student_mmlu_acc", "trained_mmlu_acc", "mmlu_n_examples",
        "teacher_hellaswag_acc", "student_hellaswag_acc", "trained_hellaswag_acc",
        "hellaswag_n_examples",
        "gptq_model_dir", "output_dir",
        "logical_total_params", "detected_bnb_4bit_layers",
        "detected_bnb_8bit_layers", "memory_footprint_bytes",
    ]
    file_exists = os.path.isfile(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def safe_get_model_memory_footprint(model: nn.Module) -> Optional[int]:
    try:
        if hasattr(model, "get_memory_footprint"):
            return int(model.get_memory_footprint())
    except Exception:
        pass
    return None


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unavailable"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


def print_gpu_memory(prefix: str = ""):
    if not torch.cuda.is_available():
        return
    cur = torch.cuda.memory_allocated() / (1024 ** 3)
    mx  = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"{prefix}GPU allocated: {cur:.2f} GB | max: {mx:.2f} GB")


def inspect_quantization(model: nn.Module, quant_backend: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "quant_backend": quant_backend,
        "bnb_linear4bit_count": 0,
        "bnb_linear8bit_count": 0,
        "gptq_like_modules_count": 0,
        "sample_quant_modules": [],
        "memory_footprint_bytes": safe_get_model_memory_footprint(model),
    }
    try:
        import bitsandbytes as bnb
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                info["bnb_linear4bit_count"] += 1
                if len(info["sample_quant_modules"]) < 12:
                    info["sample_quant_modules"].append((name, type(module).__name__))
            elif isinstance(module, bnb.nn.Linear8bitLt):
                info["bnb_linear8bit_count"] += 1
                if len(info["sample_quant_modules"]) < 12:
                    info["sample_quant_modules"].append((name, type(module).__name__))
    except Exception:
        pass
    for name, module in model.named_modules():
        joined = type(module).__name__.lower() + " " + type(module).__module__.lower()
        if "gptq" in joined or "quantlinear" in joined or "qlinear" in joined:
            info["gptq_like_modules_count"] += 1
            if len(info["sample_quant_modules"]) < 12:
                info["sample_quant_modules"].append((name, type(module).__name__))
    return info


def print_quantization_report(model: nn.Module, quant_backend: str):
    info = inspect_quantization(model, quant_backend)
    print("\n=== Quantization inspection ===")
    print(f"Backend: {quant_backend}")
    if quant_backend == "bnb":
        print(f"Detected BnB Linear4bit layers : {info['bnb_linear4bit_count']}")
        print(f"Detected BnB Linear8bitLt layers: {info['bnb_linear8bit_count']}")
        if info["bnb_linear4bit_count"] == 0 and info["bnb_linear8bit_count"] == 0:
            print("WARNING: No bitsandbytes quantized linear layers detected.")
        else:
            print("BnB quantized modules detected successfully.")
    elif quant_backend == "gptq":
        print(f"Detected GPTQ-like modules: {info['gptq_like_modules_count']}")
    print(f"Memory footprint: {human_bytes(info['memory_footprint_bytes'])}")
    if info["sample_quant_modules"]:
        print("Sample quantized modules:")
        for name, mtype in info["sample_quant_modules"][:12]:
            print(f"  - {name}: {mtype}")
    print("===============================\n")
    return info


# =========================================================
# Architecture backends
# =========================================================

SUPPORTED_BACKENDS = ("llama", "mistral", "qwen2")
MODEL_TYPE_ALIASES  = {
    "llama":     "llama",
    "mistral":   "mistral",
    "qwen2":     "qwen2",
    "qwen2_moe": "qwen2",
}


@dataclass
class DecoderBackend:
    name: str

    def get_body(self, model):
        if not hasattr(model, "model"):
            raise AttributeError(f"{model.__class__.__name__} does not expose `.model`.")
        return model.model

    def get_layers(self, model) -> nn.ModuleList:
        body = self.get_body(model)
        if not hasattr(body, "layers"):
            raise AttributeError(f"{body.__class__.__name__} does not expose `.layers`.")
        return body.layers

    def set_layers(self, model, layers):
        self.get_body(model).layers = layers

    def get_hidden_size(self, model) -> int:
        cfg = getattr(model, "config", None)
        if cfg is None or not hasattr(cfg, "hidden_size"):
            raise AttributeError(f"{model.__class__.__name__} does not expose config.hidden_size.")
        return int(cfg.hidden_size)

    def get_embed_tokens(self, model):
        return getattr(self.get_body(model), "embed_tokens", None)

    def get_model_input_device(self, model) -> torch.device:
        embed = self.get_embed_tokens(model)
        if embed is not None and hasattr(embed, "weight"):
            return embed.weight.device
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
        return next(model.parameters()).device

    def detect_layer_layout(self, layer):
        required = ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]
        missing  = [n for n in required if not hasattr(layer, n)]
        if missing:
            raise AttributeError(
                f"Backend '{self.name}': {layer.__class__.__name__} missing {missing}.")

    def get_pre_attn_norm(self, layer):  return layer.input_layernorm
    def get_attn_module(self, layer):    return layer.self_attn
    def get_post_attn_norm(self, layer): return layer.post_attention_layernorm
    def get_mlp_module(self, layer):     return layer.mlp


BACKENDS = {name: DecoderBackend(name=name) for name in SUPPORTED_BACKENDS}


def infer_backend_name_from_model_id(model_id: str) -> str:
    cfg        = AutoConfig.from_pretrained(model_id)
    model_type = getattr(cfg, "model_type", None)
    if model_type in MODEL_TYPE_ALIASES:
        return MODEL_TYPE_ALIASES[model_type]
    raise ValueError(f"Unsupported model_type '{model_type}' for '{model_id}'.")


def resolve_backend(model_id: str, backend_name: str) -> DecoderBackend:
    if backend_name == "auto":
        backend_name = infer_backend_name_from_model_id(model_id)
    if backend_name not in BACKENDS:
        raise ValueError(f"Unsupported backend '{backend_name}'.")
    return BACKENDS[backend_name]


def get_model_input_device(model, backend=None) -> torch.device:
    if backend is not None:
        return backend.get_model_input_device(model)
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    return next(model.parameters()).device


# =========================================================
# Text dataloaders (WikiText-2 / WikiText-103 / C4)
# =========================================================

def build_text_dataloaders(
    model_name, dataset_name="wikitext2", block_size=512,
    train_batch_size=2, eval_batch_size=2, num_workers=4,
    c4_train_samples="train[:1%]", c4_val_ratio=0.1, c4_test_ratio=0.1, seed=42,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    nonempty = lambda e, col: e[col] is not None and len(e[col].strip()) > 0

    if dataset_name == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")
        col = "text"
        train_split    = raw["train"].filter(lambda e: nonempty(e, col))
        val_split      = raw["validation"].filter(lambda e: nonempty(e, col))
        raw_test_split = raw["test"].filter(lambda e: nonempty(e, col))

    elif dataset_name == "wikitext103":
        raw = load_dataset("wikitext", "wikitext-103-raw-v1")
        col = "text"
        train_split    = raw["train"].filter(lambda e: nonempty(e, col))
        val_split      = raw["validation"].filter(lambda e: nonempty(e, col))
        raw_test_split = raw["test"].filter(lambda e: nonempty(e, col))

    elif dataset_name == "c4":
        raw_train = load_dataset("allenai/c4", "en", split=c4_train_samples)
        col       = "text"
        raw_train = raw_train.filter(lambda e: nonempty(e, col))
        s1        = raw_train.train_test_split(test_size=c4_test_ratio, seed=seed)
        raw_test_split = s1["test"]
        s2        = s1["train"].train_test_split(
            test_size=c4_val_ratio / (1.0 - c4_test_ratio), seed=seed)
        train_split = s2["train"]
        val_split   = s2["test"]
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    def tokenize_fn(examples):
        return tokenizer(examples[col])

    def group_texts(examples):
        cat    = {k: sum(examples[k], []) for k in examples}
        total  = (len(cat["input_ids"]) // block_size) * block_size
        result = {k: [t[i:i+block_size] for i in range(0, total, block_size)]
                  for k, t in cat.items()}
        result["labels"] = [x[:] for x in result["input_ids"]]
        return result

    lm_train = train_split.map(tokenize_fn, batched=True,
                                remove_columns=train_split.column_names).map(
        group_texts, batched=True)
    lm_val   = val_split.map(tokenize_fn, batched=True,
                              remove_columns=val_split.column_names).map(
        group_texts, batched=True)

    collator     = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(lm_train, batch_size=train_batch_size, shuffle=True,
                              collate_fn=collator, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(lm_val,   batch_size=eval_batch_size,  shuffle=False,
                              collate_fn=collator, num_workers=num_workers, pin_memory=True)
    return tokenizer, train_loader, val_loader, raw_test_split, col


# =========================================================
# MC dataloaders (MMLU / HellaSwag / mmlu+hellaswag)
# =========================================================

def _mmlu_to_text(example) -> str:
    """
    Format MMLU as MC question-answer text, matching the evaluation prompt
    format used in evaluate_mmlu (question + choices + Answer: X).
    """
    choices_str  = "\n".join(f"{l}. {c}"
                             for l, c in zip(["A","B","C","D"], example["choices"]))
    answer_label = ["A","B","C","D"][example["answer"]]
    return (f"Question: {example['question']}\n"
            f"{choices_str}\n"
            f"Answer: {answer_label}")


def _hellaswag_to_text(example) -> str:
    """
    Format HellaSwag as plain sentence completion matching the evaluation
    format used in evaluate_hellaswag: activity_label + ctx + correct ending.

    This is critical for training/eval consistency. The previous MC format
    (Context: ...\nA. ...\nAnswer: B) was mismatched with evaluation which
    scores log P(full ending continuation | context), causing degradation
    when training on HellaSwag only. Using plain completion ensures the
    adapter learns corrections aligned with how the model is scored at eval.
    """
    activity       = example.get("activity_label", "").strip()
    ctx            = example["ctx"].strip()
    context        = f"{activity}: {ctx}" if activity else ctx
    correct_ending = example["endings"][int(example["label"])].strip()
    return f"{context} {correct_ending}"


def build_mc_dataloaders(
    model_name, dataset_name="mmlu", block_size=256,
    train_batch_size=2, eval_batch_size=2, num_workers=4,
    mmlu_subject="all", mmlu_split="test",
    hellaswag_eval_split="validation",
    mc_train_max_samples=None,
    seed=42,
):
    """
    Build train/val DataLoaders for MMLU, HellaSwag, or both combined.

    mc_train_max_samples caps the number of training examples loaded per
    dataset BEFORE building the DataLoader, so len(train_loader) is accurate
    and the LR scheduler is correctly sized.

    The raw eval splits are returned in their original HF schema for
    log-likelihood scoring in evaluate_mmlu / evaluate_hellaswag.

    Training text formats:
      MMLU      : MC question-answer format matching evaluate_mmlu prompt
      HellaSwag : plain sentence completion format matching evaluate_hellaswag
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    TEXT = "text"
    raw_eval_splits = {}
    all_train_texts: List[str] = []
    all_val_texts:   List[str] = []

    # ---- MMLU ----
    if dataset_name in ("mmlu", "mmlu+hellaswag"):
        print(f"Loading MMLU (subject={mmlu_subject})...")
        mmlu_raw       = load_dataset("cais/mmlu", mmlu_subject)
        mmlu_train_raw = mmlu_raw["auxiliary_train"]
        mmlu_val_raw   = mmlu_raw["validation"]

        if mc_train_max_samples is not None:
            n = min(mc_train_max_samples, len(mmlu_train_raw))
            mmlu_train_raw = mmlu_train_raw.select(range(n))
            print(f"  MMLU train capped at {n:,} examples "
                  f"(full={len(mmlu_raw['auxiliary_train']):,})")
        else:
            print(f"  MMLU train: {len(mmlu_train_raw):,} examples (full dataset)")

        raw_eval_splits["mmlu"] = mmlu_raw[mmlu_split]
        all_train_texts.extend([_mmlu_to_text(e) for e in mmlu_train_raw])
        all_val_texts.extend(  [_mmlu_to_text(e) for e in mmlu_val_raw])

    # ---- HellaSwag ----
    if dataset_name in ("hellaswag", "mmlu+hellaswag"):
        print("Loading HellaSwag...")
        hs_raw       = load_dataset("Rowan/hellaswag")
        hs_train_raw = hs_raw["train"]
        hs_val_raw   = hs_raw[hellaswag_eval_split].filter(
            lambda e: e["label"] in ["0","1","2","3"])

        if mc_train_max_samples is not None:
            n = min(mc_train_max_samples, len(hs_train_raw))
            hs_train_raw = hs_train_raw.select(range(n))
            print(f"  HellaSwag train capped at {n:,} examples "
                  f"(full={len(hs_raw['train']):,})")
        else:
            print(f"  HellaSwag train: {len(hs_train_raw):,} examples (full dataset)")

        raw_eval_splits["hellaswag"] = hs_val_raw
        all_train_texts.extend([_hellaswag_to_text(e) for e in hs_train_raw])
        all_val_texts.extend(  [_hellaswag_to_text(e) for e in hs_val_raw])

    if not all_train_texts:
        raise ValueError(f"Unsupported MC dataset_name: '{dataset_name}'.")

    random.seed(seed)
    random.shuffle(all_train_texts)
    random.shuffle(all_val_texts)

    print(f"Total train texts: {len(all_train_texts):,} | "
          f"val texts: {len(all_val_texts):,}")

    from datasets import Dataset as HFDataset
    train_hf = HFDataset.from_dict({TEXT: all_train_texts})
    val_hf   = HFDataset.from_dict({TEXT: all_val_texts})

    def tokenize_fn(examples):
        return tokenizer(examples[TEXT])

    def group_texts(examples):
        cat    = {k: sum(examples[k], []) for k in examples}
        total  = (len(cat["input_ids"]) // block_size) * block_size
        result = {k: [t[i:i+block_size] for i in range(0, total, block_size)]
                  for k, t in cat.items()}
        result["labels"] = [x[:] for x in result["input_ids"]]
        return result

    lm_train = train_hf.map(tokenize_fn, batched=True, remove_columns=[TEXT]).map(
        group_texts, batched=True, desc="Grouping MC train texts")
    lm_val   = val_hf.map(tokenize_fn,   batched=True, remove_columns=[TEXT]).map(
        group_texts, batched=True, desc="Grouping MC val texts")

    print(f"Train blocks: {len(lm_train):,} | Val blocks: {len(lm_val):,}")

    collator     = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_loader = DataLoader(lm_train, batch_size=train_batch_size, shuffle=True,
                              collate_fn=collator, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(lm_val,   batch_size=eval_batch_size,  shuffle=False,
                              collate_fn=collator, num_workers=num_workers, pin_memory=True)
    return tokenizer, train_loader, val_loader, raw_eval_splits, TEXT


# =========================================================
# Dataloader router
# =========================================================

_MC_DATASETS   = {"mmlu", "hellaswag", "mmlu+hellaswag"}
_TEXT_DATASETS = {"wikitext2", "wikitext103", "c4"}


def build_dataloaders(args, model_name: str):
    if args.dataset_name in _MC_DATASETS:
        return build_mc_dataloaders(
            model_name=model_name,
            dataset_name=args.dataset_name,
            block_size=args.block_size,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            mmlu_subject=args.mmlu_subject,
            mmlu_split=args.mmlu_split,
            hellaswag_eval_split=args.hellaswag_eval_split,
            mc_train_max_samples=(
                args.mc_train_max_samples if args.mc_train_max_samples > 0 else None),
            seed=args.seed,
        )
    return build_text_dataloaders(
        model_name=model_name,
        dataset_name=args.dataset_name,
        block_size=args.block_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        c4_train_samples=args.c4_train_samples,
        c4_val_ratio=args.c4_val_ratio,
        c4_test_ratio=args.c4_test_ratio,
        seed=args.seed,
    )


# =========================================================
# Adapters
# =========================================================

class Gate(nn.Module):
    def __init__(self, init: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return self.alpha * x


class IdentityAdapter(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class MLPTokenAdapter(nn.Module):
    def __init__(self, dim, kind="lowrank", r=8, act="gelu",
                 gate_init=0.1, dtype=torch.float32):
        super().__init__()
        self.kind = kind
        if kind == "affine":
            self.scale = nn.Parameter(torch.zeros(dim, dtype=dtype))
            self.bias  = nn.Parameter(torch.zeros(dim, dtype=dtype))
            self.low1 = self.low2 = self.mid = None
        else:
            self.scale = self.bias = None
            self.low1 = nn.Linear(dim, r, bias=True, dtype=dtype)
            self.low2 = nn.Linear(r, dim, bias=True, dtype=dtype)
            nn.init.kaiming_uniform_(self.low1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.low1.bias)
            nn.init.zeros_(self.low2.weight)
            nn.init.zeros_(self.low2.bias)
            if kind == "bottleneck":
                self.mid = nn.Linear(r, r, bias=True, dtype=dtype)
                nn.init.zeros_(self.mid.weight)
                nn.init.zeros_(self.mid.bias)
            else:
                self.mid = None
        self.act  = (nn.GELU() if act == "gelu" else
                     nn.SiLU() if act == "silu" else nn.Identity())
        self.gate = Gate(gate_init)

    def forward(self, x):
        dtype = x.dtype
        y = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        if self.kind == "affine":
            out = y * self.scale + self.bias
        elif self.kind == "lowrank":
            out = self.low2(self.low1(y))
        else:
            out = self.low2(self.act(self.mid(self.act(self.low1(y)))))
        return self.gate(out).to(dtype)


def build_branch_adapter(dim, kind, r, gate_init, dtype) -> nn.Module:
    if kind == "none":
        return IdentityAdapter()
    if kind not in ("affine", "lowrank", "bottleneck"):
        raise ValueError(f"Unsupported adapter kind: {kind}")
    return MLPTokenAdapter(dim=dim, kind=kind, r=r, gate_init=gate_init, dtype=dtype)


class WrappedDecoderLayer(nn.Module):
    def __init__(self, decoder_layer, backend, hidden_size,
                 attn_adapter_kind="lowrank", mlp_adapter_kind="lowrank",
                 attn_r=8, mlp_r=8, gate_init=0.1, adapter_dtype=torch.float32):
        super().__init__()
        self.layer   = decoder_layer
        self.backend = backend
        self.backend.detect_layer_layout(decoder_layer)
        self.attn_adapter = build_branch_adapter(
            hidden_size, attn_adapter_kind, attn_r, gate_init, adapter_dtype)
        self.mlp_adapter  = build_branch_adapter(
            hidden_size, mlp_adapter_kind,  mlp_r,  gate_init, adapter_dtype)

    def _call_self_attn(self, u, **kwargs):
        attn = self.backend.get_attn_module(self.layer)
        trials = [
            dict(hidden_states=u, **kwargs),
            {"hidden_states": u,
             "attention_mask":    kwargs.get("attention_mask"),
             "position_ids":      kwargs.get("position_ids"),
             "past_key_value":    kwargs.get("past_key_value"),
             "output_attentions": kwargs.get("output_attentions", False),
             "use_cache":         kwargs.get("use_cache", False),
             "cache_position":    kwargs.get("cache_position"),
             "position_embeddings": kwargs.get("position_embeddings")},
            {"hidden_states": u,
             "attention_mask":    kwargs.get("attention_mask"),
             "position_ids":      kwargs.get("position_ids"),
             "past_key_value":    kwargs.get("past_key_value"),
             "output_attentions": kwargs.get("output_attentions", False),
             "use_cache":         kwargs.get("use_cache", False)},
            {"hidden_states":  u,
             "attention_mask": kwargs.get("attention_mask"),
             "position_ids":   kwargs.get("position_ids")},
        ]
        last_err = None
        for ck in trials:
            try:
                return attn(**ck)
            except TypeError as e:
                last_err = e
        raise last_err

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        residual = hidden_states
        u        = self.backend.get_pre_attn_norm(self.layer)(hidden_states)
        attn_out = self._call_self_attn(
            u, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs)

        attn_tail   = attn_out[1:] if isinstance(attn_out, tuple) else ()
        attn_output = attn_out[0]  if isinstance(attn_out, tuple) else attn_out

        ap = list(self.attn_adapter.parameters())
        if ap and ap[0].device != u.device:
            self.attn_adapter = self.attn_adapter.to(u.device)
        hidden_states = residual + attn_output + self.attn_adapter(u)

        residual = hidden_states
        v        = self.backend.get_post_attn_norm(self.layer)(hidden_states)
        mp = list(self.mlp_adapter.parameters())
        if mp and mp[0].device != v.device:
            self.mlp_adapter = self.mlp_adapter.to(v.device)
        hidden_states = (residual
                         + self.backend.get_mlp_module(self.layer)(v)
                         + self.mlp_adapter(v))

        if output_attentions and use_cache:
            return (hidden_states,
                    attn_tail[1] if len(attn_tail) > 1 else None,
                    attn_tail[0] if len(attn_tail) > 0 else None)
        elif output_attentions:
            return (hidden_states, attn_tail[0] if attn_tail else None)
        elif use_cache:
            return (hidden_states, attn_tail[0] if attn_tail else None)
        return hidden_states


def add_decoder_block_adapters(model, backend, attn_r=8, mlp_r=8,
                                attn_adapter_kind="lowrank", mlp_adapter_kind="lowrank",
                                gate_init=0.1, freeze_backbone=True,
                                adapter_dtype=torch.float32) -> List[nn.Parameter]:
    hidden_size    = backend.get_hidden_size(model)
    layers         = backend.get_layers(model)
    new_layers     = []
    adapter_params: List[nn.Parameter] = []

    for layer in layers:
        dev     = next(layer.parameters()).device
        wrapped = WrappedDecoderLayer(
            decoder_layer=layer, backend=backend, hidden_size=hidden_size,
            attn_adapter_kind=attn_adapter_kind, mlp_adapter_kind=mlp_adapter_kind,
            attn_r=attn_r, mlp_r=mlp_r, gate_init=gate_init, adapter_dtype=adapter_dtype,
        ).to(dev)
        new_layers.append(wrapped)
        for name, p in wrapped.named_parameters():
            if "attn_adapter" in name or "mlp_adapter" in name:
                adapter_params.append(p)

    backend.set_layers(model, nn.ModuleList(new_layers))
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
    for p in adapter_params:
        p.requires_grad_(True)
    return adapter_params


# =========================================================
# Feature Matching hooks
# =========================================================

class FeatureHookStore:
    def __init__(self):
        self.features: Dict[int, torch.Tensor] = {}
        self.handles = []

    def clear(self):  self.features.clear()

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            self.features[idx] = output[0] if isinstance(output, tuple) else output
        return hook


def register_layer_output_hooks(model, backend, layer_ids) -> FeatureHookStore:
    store  = FeatureHookStore()
    layers = backend.get_layers(model)
    for idx in [i for i in layer_ids if 0 <= i < len(layers)]:
        store.handles.append(layers[idx].register_forward_hook(store._make_hook(idx)))
    return store


# =========================================================
# Losses
# =========================================================

def kd_loss_causal_lm(student_logits, teacher_logits, labels, temperature=2.0):
    s = student_logits[:, :-1, :].contiguous()
    t = teacher_logits[:, :-1, :].contiguous()
    y = labels[:, 1:].contiguous()
    valid = y.ne(-100)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
    s, t  = s[valid], t[valid]
    log_p = F.log_softmax(s / temperature, dim=-1)
    p_t   = F.softmax(t / temperature, dim=-1)
    return F.kl_div(log_p, p_t, reduction="batchmean") * (temperature ** 2)


# =========================================================
# Evaluation — perplexity (text datasets)
# =========================================================

@torch.no_grad()
def evaluate_perplexity(model, data_loader, device, max_batches=None):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for step, batch in enumerate(data_loader):
        if max_batches is not None and step >= max_batches:
            break
        batch  = move_batch_to_device(batch, device)
        out    = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       labels=batch["labels"], use_cache=False, return_dict=True)
        valid  = (batch["labels"][:, 1:] != -100).sum().item()
        total_nll    += out.loss.float().item() * valid
        total_tokens += valid
        del out, batch
    return math.exp(total_nll / max(total_tokens, 1))


@torch.no_grad()
def evaluate_perplexity_standard(model, tokenizer, raw_text_dataset,
                                  text_column="text", max_length=None,
                                  stride=512, backend=None):
    model.eval()
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", None)
        if max_length is None or max_length > 4096:
            max_length = 2048

    text      = "\n\n".join(raw_text_dataset[text_column])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(
        get_model_input_device(model, backend))
    seq_len   = input_ids.size(1)
    nll_sum, n_tokens, prev_end = 0.0, 0, 0

    for begin_loc in range(0, seq_len, stride):
        end_loc  = min(begin_loc + max_length, seq_len)
        trg_len  = end_loc - prev_end
        ids      = input_ids[:, begin_loc:end_loc]
        tgt      = ids.clone()
        tgt[:, :-trg_len] = -100
        out      = model(input_ids=ids, labels=tgt, use_cache=False, return_dict=True)
        nll_sum += out.loss.float().item() * trg_len
        n_tokens += trg_len
        del out, ids, tgt
        prev_end = end_loc
        if end_loc == seq_len:
            break
    return math.exp(nll_sum / max(n_tokens, 1))


# =========================================================
# Evaluation — MMLU (n-shot, log-likelihood, lm-eval-harness exact match)
# =========================================================

def _resolve_label_token_ids(tokenizer, labels=("A","B","C","D")) -> List[int]:
    ids = []
    for label in labels:
        space_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        bare_ids  = tokenizer.encode(label,        add_special_tokens=False)
        if len(space_ids) == 1:
            ids.append(space_ids[0])
        elif len(bare_ids) == 1:
            ids.append(bare_ids[0])
        else:
            ctx_ids = tokenizer.encode(f"Answer: {label}", add_special_tokens=False)
            ids.append(ctx_ids[-1])
    return ids


@torch.no_grad()
def evaluate_mmlu(
    model: nn.Module,
    tokenizer,
    dataset,
    device: torch.device,
    backend: Optional[DecoderBackend] = None,
    max_samples: Optional[int] = None,
    num_fewshot: int = 5,
) -> Dict[str, float]:
    """
    n-shot MMLU evaluation matching lm-evaluation-harness exactly.

    Few-shot examples come from the MMLU dev split (5 per subject).
    Prompt format per lm-eval default template:
      description : "The following are multiple choice questions (with answers) about {subject}.\n\n"
      doc_to_text : "{question}\nA. {c0}\nB. {c1}\nC. {c2}\nD. {c3}\nAnswer:"
      doc_to_choice: [" A", " B", " C", " D"]  (space-prefixed for LLaMA SP tokenizer)
    """
    model.eval()
    input_device = get_model_input_device(model, backend)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    LABELS = ["A", "B", "C", "D"]

    label_token_ids = _resolve_label_token_ids(tokenizer, LABELS)
    print("MMLU label token IDs:")
    for l, tid in zip(LABELS, label_token_ids):
        print(f"  '{l}' -> token_id={tid} -> decoded='{tokenizer.decode([tid])}'")

    # Load dev split for few-shot examples
    few_shot_by_subject: Dict[str, List] = {}
    if num_fewshot > 0:
        try:
            dev_data = load_dataset("cais/mmlu", "all", split="dev")
            for ex in dev_data:
                subj = ex.get("subject", "")
                if subj not in few_shot_by_subject:
                    few_shot_by_subject[subj] = []
                if len(few_shot_by_subject[subj]) < num_fewshot:
                    few_shot_by_subject[subj].append(ex)
            print(f"Loaded few-shot dev examples for "
                  f"{len(few_shot_by_subject)} subjects.")
        except Exception as e:
            print(f"Warning: could not load MMLU dev split: {e}")

    def format_example(ex: dict, include_answer: bool) -> str:
        choices_str = "\n".join(f"{l}. {c}" for l, c in zip(LABELS, ex["choices"]))
        text = f"{ex['question'].strip()}\n{choices_str}\nAnswer:"
        if include_answer:
            text += f" {LABELS[int(ex['answer'])]}\n\n"
        return text

    def build_prompt(example: dict) -> str:
        subject = example.get("subject", "")
        header  = (
            f"The following are multiple choice questions (with answers) "
            f"about {subject.replace('_', ' ')}.\n\n"
            if subject else
            "The following are multiple choice questions (with answers).\n\n"
        )
        shots = few_shot_by_subject.get(subject, [])
        if not shots and few_shot_by_subject:
            shots = next(iter(few_shot_by_subject.values()))
        shots = shots[:num_fewshot]
        few_shot_block = "".join(format_example(s, include_answer=True) for s in shots)
        test_block     = format_example(example, include_answer=False)
        return header + few_shot_block + test_block

    correct, total = 0, 0
    for example in dataset:
        gold    = int(example["answer"])
        context = build_prompt(example)
        ctx_ids = tokenizer.encode(
            context, add_special_tokens=True, return_tensors="pt").to(input_device)

        out       = model(input_ids=ctx_ids, use_cache=False, return_dict=True)
        logits    = out.logits[0, -1, :].float()
        log_probs = F.log_softmax(logits, dim=-1)
        scores     = [log_probs[tid].item() for tid in label_token_ids]
        prediction = int(torch.tensor(scores).argmax().item())

        if prediction == gold:
            correct += 1
        total += 1
        del out, ctx_ids

    return {"accuracy": 100.0 * correct / max(total, 1),
            "n_correct": correct, "n_examples": total}


# =========================================================
# Evaluation — HellaSwag (0-shot, log-likelihood over full endings)
# =========================================================

@torch.no_grad()
def _score_continuation(model, ctx_ids, cont_ids, device) -> float:
    full_ids  = torch.cat([ctx_ids, cont_ids], dim=1).to(device)
    cont_len  = cont_ids.size(1)
    out       = model(input_ids=full_ids, use_cache=False, return_dict=True)
    logits    = out.logits[0, :-1, :].float()
    target    = full_ids[0, 1:]
    start     = logits.size(0) - cont_len
    log_probs = F.log_softmax(logits[start:], dim=-1)
    token_lps = log_probs[torch.arange(cont_len, device=device), target[start:]]
    del out, full_ids
    return token_lps.mean().item()


@torch.no_grad()
def evaluate_hellaswag(
    model: nn.Module,
    tokenizer,
    dataset,
    device: torch.device,
    backend: Optional[DecoderBackend] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    0-shot HellaSwag evaluation using log-likelihood scoring over full endings.
    Context = activity_label + ": " + ctx  (matches _hellaswag_to_text training format)
    Each ending is scored as a continuation of the context.
    """
    model.eval()
    input_device = get_model_input_device(model, backend)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct, total = 0, 0
    for example in dataset:
        activity = example.get("activity_label", "").strip()
        ctx      = example["ctx"].strip()
        context  = f"{activity}: {ctx}" if activity else ctx
        gold     = int(example["label"])
        endings  = example["endings"]

        ctx_ids = tokenizer.encode(
            context, add_special_tokens=True, return_tensors="pt")

        scores = []
        for ending in endings:
            cont_ids = tokenizer.encode(
                " " + ending.strip(), add_special_tokens=False, return_tensors="pt")
            scores.append(_score_continuation(model, ctx_ids, cont_ids, input_device))

        if int(torch.tensor(scores).argmax().item()) == gold:
            correct += 1
        total += 1

    return {"accuracy": 100.0 * correct / max(total, 1),
            "n_correct": correct, "n_examples": total}


# =========================================================
# MC eval dispatcher
# =========================================================

def evaluate_mc_datasets(
    model, tokenizer, raw_eval_splits, device, backend,
    mc_eval_max_samples, num_fewshot=5, label="",
) -> Dict[str, Dict]:
    results = {}

    if "mmlu" in raw_eval_splits:
        print(f"{label} MMLU evaluation ({num_fewshot}-shot, log-likelihood)...")
        results["mmlu"] = evaluate_mmlu(
            model, tokenizer, raw_eval_splits["mmlu"],
            device=device, backend=backend,
            max_samples=mc_eval_max_samples,
            num_fewshot=num_fewshot)
        r = results["mmlu"]
        print(f"  MMLU   acc={r['accuracy']:.2f}%  ({r['n_correct']}/{r['n_examples']})")

    if "hellaswag" in raw_eval_splits:
        print(f"{label} HellaSwag evaluation (0-shot, log-likelihood)...")
        results["hellaswag"] = evaluate_hellaswag(
            model, tokenizer, raw_eval_splits["hellaswag"],
            device=device, backend=backend,
            max_samples=mc_eval_max_samples)
        r = results["hellaswag"]
        print(f"  HellaSwag acc={r['accuracy']:.2f}%  ({r['n_correct']}/{r['n_examples']})")

    return results


# =========================================================
# Saving
# =========================================================

def extract_trainable_state_dict(model):
    return {name: p.detach().cpu()
            for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(output_dir, model, tokenizer, args, epoch, val_metric, is_best=False):
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {"epoch": epoch, "val_metric": val_metric,
            "trainable_state_dict": extract_trainable_state_dict(model),
            "args": vars(args)}
    torch.save(ckpt, os.path.join(output_dir, "last_adapter_checkpoint.pt"))
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    if is_best:
        torch.save(ckpt, os.path.join(output_dir, "best_adapter_checkpoint.pt"))


# =========================================================
# Model Loading
# =========================================================

def load_teacher_model_gpu_eval(model_id, compute_dtype, tokenizer, is_mc=False):
    # For MC datasets (MMLU/HellaSwag), load teacher on CPU to avoid OOM
    # when switching between large models in a sweep. MC eval runs forward
    # passes on CPU which is slower but avoids the memory issue entirely.
    # For text datasets (perplexity), GPU is needed for reasonable speed.
    if is_mc:
        print("Loading teacher on CPU for MC evaluation (avoids OOM for large models)...")
        device = {"": "cpu"}
    else:
        print("Loading teacher on GPU for perplexity evaluation...")
        device = {"": 0}
    t = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=compute_dtype, device_map=device, low_cpu_mem_usage=True)
    t.config.use_cache = False
    t.config.pad_token_id = tokenizer.pad_token_id
    t.eval()
    for p in t.parameters(): p.requires_grad_(False)
    return t


def load_teacher_model_cpu_train(model_id, compute_dtype, tokenizer):
    t = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=compute_dtype, device_map={"": "cpu"}, low_cpu_mem_usage=True)
    t.config.use_cache = False
    t.config.pad_token_id = tokenizer.pad_token_id
    t.eval()
    for p in t.parameters(): p.requires_grad_(False)
    return t


def quantize_and_save_gptq_model(model_id, tokenizer, args):
    print(f"Quantizing {model_id} with GPTQ -> {args.gptq_model_dir}")
    gptq_cfg = GPTQConfig(
        bits=args.gptq_bits, dataset=args.gptq_dataset, tokenizer=tokenizer,
        group_size=args.gptq_group_size, desc_act=args.gptq_desc_act)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=gptq_cfg)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    os.makedirs(args.gptq_model_dir, exist_ok=True)
    model.save_pretrained(args.gptq_model_dir)
    tokenizer.save_pretrained(args.gptq_model_dir)
    print(f"Saved GPTQ model to: {args.gptq_model_dir}")
    return model


def load_student_model(model_id, tokenizer, compute_dtype, args):
    if args.quant_backend == "bnb":
        print(f"Loading BnB student (4bit={args.load_in_4bit}, type={args.quant_type})...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit, bnb_4bit_quant_type=args.quant_type,
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
        student = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg,
            torch_dtype=compute_dtype, device_map={"": 0})
    elif args.quant_backend == "gptq":
        ensure_gptq_model_dir(args)
        has_saved = (
            os.path.isdir(args.gptq_model_dir) and
            (os.path.isfile(os.path.join(args.gptq_model_dir, "config.json")) or
             os.path.isfile(os.path.join(args.gptq_model_dir, "quantize_config.json"))))
        if not has_saved or args.gptq_quantize_only:
            cleanup_model(quantize_and_save_gptq_model(model_id, tokenizer, args))
            if args.gptq_quantize_only:
                print("GPTQ done. Re-run without --gptq_quantize_only to train.")
                return None
        print(f"Loading GPTQ student from: {args.gptq_model_dir}")
        student = AutoModelForCausalLM.from_pretrained(
            args.gptq_model_dir, device_map={"": 0}, torch_dtype=compute_dtype)
    else:
        raise ValueError(f"Unsupported quant backend: {args.quant_backend}")

    student.config.use_cache = False
    student.config.pad_token_id = tokenizer.pad_token_id
    return student


# =========================================================
# Training
# =========================================================

def train_one_epoch(teacher, student, train_loader, optimizer, scheduler, scaler,
                    teacher_hooks, student_hooks, fm_layers, device, epoch, args):
    teacher.eval(); student.train()
    running   = {"loss": 0.0, "kd": 0.0, "ce": 0.0, "fm": 0.0}
    num_steps = 0
    optimizer.zero_grad(set_to_none=True)

    autocast_dtype    = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    trainable_params  = get_trainable_params(student)
    kd_chunk_size     = getattr(args, "kd_chunk_size", 0)
    empty_cache_steps = getattr(args, "empty_cache_steps", 0)
    log_interval      = getattr(args, "log_every", 50)
    max_steps         = getattr(args, "max_steps_per_epoch", 0)

    for step, batch in enumerate(train_loader):

        if max_steps > 0 and step >= max_steps:
            print(f"  Reached max_steps_per_epoch={max_steps}, stopping epoch early.")
            break

        clear_feature_store(teacher_hooks)
        clear_feature_store(student_hooks)

        batch_t = move_batch_to_cpu(batch)
        with torch.no_grad():
            t_out = teacher(input_ids=batch_t["input_ids"],
                            attention_mask=batch_t["attention_mask"],
                            labels=None, use_cache=False, return_dict=True)
        teacher_logits = maybe_to_device(t_out.logits, device, dtype=torch.float32)
        teacher_feats  = {}
        if args.alpha_fm > 0:
            for idx in fm_layers:
                feat = teacher_hooks.features.get(idx)
                if feat is not None:
                    teacher_feats[idx] = feat.to(
                        device, dtype=torch.float32, non_blocking=True)
        del t_out, batch_t

        batch_s = move_batch_to_device_nonblocking(batch, device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            s_out = student(input_ids=batch_s["input_ids"],
                            attention_mask=batch_s["attention_mask"],
                            labels=None, use_cache=False, return_dict=True)

            if args.alpha_kd > 0:
                if kd_chunk_size > 0:
                    vocab = s_out.logits.size(-1)
                    kd_accum, nch = 0.0, 0
                    for start in range(0, vocab, kd_chunk_size):
                        end       = min(start + kd_chunk_size, vocab)
                        kd_accum += kd_loss_causal_lm(
                            s_out.logits[..., start:end],
                            teacher_logits[..., start:end],
                            batch_s["labels"], args.temperature)
                        nch += 1
                    loss_kd = kd_accum / max(nch, 1)
                else:
                    loss_kd = kd_loss_causal_lm(
                        s_out.logits, teacher_logits,
                        batch_s["labels"], args.temperature)
            else:
                loss_kd = torch.zeros((), device=device, dtype=torch.float32)

            if args.alpha_fm > 0:
                loss_fm, valid_fm = torch.zeros((), device=device, dtype=torch.float32), 0
                for idx in fm_layers:
                    sf = student_hooks.features.get(idx)
                    tf = teacher_feats.get(idx)
                    if sf is not None and tf is not None:
                        loss_fm  = loss_fm + F.mse_loss(sf.float(), tf.float())
                        valid_fm += 1
                if valid_fm > 0:
                    loss_fm = loss_fm / valid_fm
            else:
                loss_fm = torch.zeros((), device=device, dtype=torch.float32)

            loss_ce    = torch.zeros((), device=device, dtype=torch.float32)
            total_loss = args.alpha_kd * loss_kd + args.alpha_fm * loss_fm
            loss       = total_loss / args.grad_accum

        if scaler is not None and args.use_grad_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        del batch_s, teacher_logits, teacher_feats, s_out

        if (step + 1) % args.grad_accum == 0:
            if args.max_grad_norm > 0:
                if scaler is not None and args.use_grad_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            if scaler is not None and args.use_grad_scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running["loss"] += total_loss.detach().float().item()
        running["kd"]   += loss_kd.detach().float().item()
        running["ce"]   += loss_ce.detach().float().item()
        running["fm"]   += loss_fm.detach().float().item()
        num_steps       += 1

        if empty_cache_steps > 0 and (step + 1) % empty_cache_steps == 0:
            gc.collect(); torch.cuda.empty_cache()

        if (step + 1) % log_interval == 0:
            cur = torch.cuda.memory_allocated(device) / (1024 ** 3)
            mx  = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            cap = f"/{max_steps}" if max_steps > 0 else f"/{len(train_loader)}"
            print(f"[Epoch {epoch} Step {step+1}{cap}] "
                  f"loss={running['loss']/num_steps:.4f} "
                  f"kd={running['kd']/num_steps:.4f} "
                  f"fm={running['fm']/num_steps:.6f} "
                  f"gpu={cur:.2f}GB max={mx:.2f}GB")

    d = max(num_steps, 1)
    return {k: v / d for k, v in running.items()}


# =========================================================
# Args
# =========================================================

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--model_id",     type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--output_dir",   type=str, default="./runs/kd_fm")
    ap.add_argument("--results_file", type=str, default="./results/summary_results.csv")
    ap.add_argument("--seed",         type=int, default=42)

    ap.add_argument("--arch_backend", type=str, default="auto",
                    choices=["auto", "llama", "mistral", "qwen2"])

    ap.add_argument(
        "--dataset_name", type=str, default="mmlu+hellaswag",
        choices=["wikitext2", "wikitext103", "c4", "mmlu", "hellaswag", "mmlu+hellaswag"],
        help="wikitext2/wikitext103/c4 -> perplexity. "
             "mmlu/hellaswag/mmlu+hellaswag -> log-likelihood accuracy.")
    ap.add_argument("--block_size",       type=int, default=256)
    ap.add_argument("--train_batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size",  type=int, default=2)
    ap.add_argument("--num_workers",      type=int, default=4)

    # C4-only
    ap.add_argument("--c4_train_samples", type=str,   default="train[:1%]")
    ap.add_argument("--c4_val_ratio",     type=float, default=0.1)
    ap.add_argument("--c4_test_ratio",    type=float, default=0.1)

    # MMLU
    ap.add_argument("--mmlu_subject",     type=str, default="all")
    ap.add_argument("--mmlu_split",       type=str, default="test",
                    choices=["test", "validation"])
    ap.add_argument("--mmlu_num_fewshot", type=int, default=5)

    # HellaSwag
    ap.add_argument("--hellaswag_eval_split", type=str, default="validation",
                    choices=["validation", "train"])

    # MC data size control
    ap.add_argument("--mc_train_max_samples", type=int, default=5000,
                    help="Max training examples per MC dataset. 0 = full dataset.")
    ap.add_argument("--mc_eval_max_samples",  type=int, default=500,
                    help="Max examples per MC dataset for eval. 0 = full dataset.")
    ap.add_argument("--max_steps_per_epoch",  type=int, default=0,
                    help="Hard-stop each epoch after N steps. 0 = no cap.")

    ap.add_argument("--epochs",        type=int,   default=3)
    ap.add_argument("--lr",            type=float, default=1e-3)
    ap.add_argument("--weight_decay",  type=float, default=0.0)
    ap.add_argument("--warmup_ratio",  type=float, default=0.03)
    ap.add_argument("--grad_accum",    type=int,   default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_every",     type=int,   default=50)

    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha_kd",    type=float, default=1.0)
    ap.add_argument("--alpha_fm",    type=float, default=1e-3)
    ap.add_argument("--alpha_ce",    type=float, default=0.0)

    ap.add_argument("--attn_adapter_kind", type=str, default="lowrank",
                    choices=["none", "affine", "lowrank", "bottleneck"])
    ap.add_argument("--mlp_adapter_kind",  type=str, default="lowrank",
                    choices=["none", "affine", "lowrank", "bottleneck"])
    ap.add_argument("--attn_r",    type=int,   default=8)
    ap.add_argument("--mlp_r",     type=int,   default=8)
    ap.add_argument("--gate_init", type=float, default=0.1)

    ap.add_argument("--quant_backend", type=str, default="bnb", choices=["bnb", "gptq"])
    ap.add_argument("--load_in_4bit",
                    type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
                    default=True)
    ap.add_argument("--quant_type",    type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--compute_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--gptq_bits",          type=int,  default=4)
    ap.add_argument("--gptq_dataset",       type=str,  default="wikitext2")
    ap.add_argument("--gptq_group_size",    type=int,  default=128)
    ap.add_argument("--gptq_desc_act",      action="store_true", default=False)
    ap.add_argument("--gptq_root_dir",      type=str,  default="./gptq_students")
    ap.add_argument("--gptq_model_dir",     type=str,  default="")
    ap.add_argument("--gptq_quantize_only", action="store_true", default=False)

    ap.add_argument("--amp_dtype",       type=str,  default="bf16",
                    choices=["bf16", "fp16"])
    ap.add_argument("--use_grad_scaler", action="store_true", default=False)

    ap.add_argument("--eval_max_batches", type=int, default=None)
    ap.add_argument("--ppl_max_length",   type=int, default=512)
    ap.add_argument("--ppl_stride",       type=int, default=256)

    ap.add_argument("--kd_chunk_size",     type=int, default=0)
    ap.add_argument("--empty_cache_steps", type=int, default=0)

    return ap.parse_args()


# =========================================================
# Result table helpers
# =========================================================

def _print_mc_comparison_table(teacher_mc, student_mc, trained_mc):
    datasets = sorted(set(
        list(teacher_mc.keys()) + list(student_mc.keys()) + list(trained_mc.keys())))
    print("\n=== Log-likelihood accuracy comparison ===")
    print(f"{'Dataset':<14} {'Teacher':>10} {'Student':>10} {'Trained':>10}")
    print("-" * 48)
    for ds in datasets:
        t = teacher_mc.get(ds, {}).get("accuracy", float("nan"))
        s = student_mc.get(ds, {}).get("accuracy", float("nan"))
        f = trained_mc.get(ds, {}).get("accuracy", float("nan"))
        print(f"{ds:<14} {t:>9.2f}% {s:>9.2f}% {f:>9.2f}%")
    print("=" * 48)


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    ensure_gptq_model_dir(args)

    assert torch.cuda.is_available(), "This script requires a CUDA GPU."
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7")

    backend = resolve_backend(args.model_id, args.arch_backend)
    args.arch_backend = backend.name
    print(f"Resolved architecture backend: {backend.name}")

    device        = torch.device("cuda:0")
    compute_dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
    is_mc         = args.dataset_name in _MC_DATASETS
    mc_max        = args.mc_eval_max_samples if args.mc_eval_max_samples > 0 else None

    if is_mc:
        cap = args.mc_train_max_samples
        print(f"\nTraining data config:")
        print(f"  dataset              : {args.dataset_name}")
        print(f"  mc_train_max_samples : {cap if cap > 0 else 'full dataset'}")
        print(f"  max_steps_per_epoch  : {args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else 'no cap'}")
        print(f"  epochs               : {args.epochs}\n")

    print(f"Building dataloaders: {args.dataset_name}")
    tokenizer, train_loader, val_loader, raw_eval_data, text_column = build_dataloaders(
        args, model_name=args.model_id)

    # ---- Teacher eval ----
    teacher = load_teacher_model_gpu_eval(
        args.model_id, compute_dtype, tokenizer, is_mc=is_mc)

    if is_mc:
        teacher_mc = evaluate_mc_datasets(
            teacher, tokenizer, raw_eval_data,
            device=device, backend=backend,
            mc_eval_max_samples=mc_max,
            num_fewshot=args.mmlu_num_fewshot,
            label="Teacher")
        teacher_test_ppl = None
    else:
        print("Teacher test perplexity...")
        teacher_test_ppl = evaluate_perplexity_standard(
            teacher, tokenizer, raw_eval_data, text_column=text_column,
            max_length=args.ppl_max_length, stride=args.ppl_stride, backend=backend)
        teacher_mc = {}
        print(f"Teacher test ppl: {teacher_test_ppl:.4f}")

    cleanup_model(teacher); teacher = None
    print_gpu_memory(prefix="[After teacher cleanup] ")

    # ---- Load student + inject adapters ----
    print("Loading student...")
    student = load_student_model(args.model_id, tokenizer, compute_dtype, args)
    if student is None:
        return

    student.config.use_cache = False
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    quant_info = print_quantization_report(student, args.quant_backend)

    print(f"Injecting adapters (backend={backend.name})...")
    adapter_params = add_decoder_block_adapters(
        student, backend=backend, attn_r=args.attn_r, mlp_r=args.mlp_r,
        attn_adapter_kind=args.attn_adapter_kind, mlp_adapter_kind=args.mlp_adapter_kind,
        gate_init=args.gate_init, freeze_backbone=True, adapter_dtype=torch.float32)

    student_layers = backend.get_layers(student)
    for i in range(min(2, len(student_layers))):
        ap_ = list(student_layers[i].attn_adapter.parameters())
        mp_ = list(student_layers[i].mlp_adapter.parameters())
        print(f"layer {i}  "
              f"attn={type(student_layers[i].attn_adapter).__name__}"
              f"@{ap_[0].device if ap_ else 'none'}  "
              f"mlp={type(student_layers[i].mlp_adapter).__name__}"
              f"@{mp_[0].device if mp_ else 'none'}")

    total_params_count     = count_total_params(student)
    trainable_params_count = count_trainable_params(student)
    print(f"\n=== Parameter summary ===")
    print(f"Logical total    : {total_params_count:,}")
    print(f"Trainable        : {trainable_params_count:,}  ({len(adapter_params)} tensors)")
    if args.quant_backend == "bnb":
        print(f"BnB 4-bit layers : {quant_info['bnb_linear4bit_count']}")
        print(f"Memory footprint : {human_bytes(quant_info['memory_footprint_bytes'])}")
    print("=========================\n")

    student_n_layers = len(backend.get_layers(student))

    # ---- Initial student eval ----
    if is_mc:
        student_mc = evaluate_mc_datasets(
            student, tokenizer, raw_eval_data,
            device=device, backend=backend,
            mc_eval_max_samples=mc_max,
            num_fewshot=args.mmlu_num_fewshot,
            label="Student (quantized)")
        init_test_ppl = None
    else:
        print("Initial student test perplexity...")
        init_test_ppl = evaluate_perplexity_standard(
            student, tokenizer, raw_eval_data, text_column=text_column,
            max_length=args.ppl_max_length, stride=args.ppl_stride, backend=backend)
        student_mc = {}
        print(f"Initial quantized student test ppl: {init_test_ppl:.4f}")

    # ---- Load teacher on CPU for training ----
    print("Loading teacher on CPU for KD training...")
    teacher          = load_teacher_model_cpu_train(args.model_id, compute_dtype, tokenizer)
    teacher_n_layers = len(backend.get_layers(teacher))
    shared_n_layers  = min(student_n_layers, teacher_n_layers)
    print(f"Student layers: {student_n_layers}  Teacher layers: {teacher_n_layers}")

    fm_layers = (sorted({0, shared_n_layers // 4, shared_n_layers // 2,
                          (3 * shared_n_layers) // 4, shared_n_layers - 1})
                 if shared_n_layers >= 8 else list(range(shared_n_layers)))
    print(f"FM layers: {fm_layers}")

    teacher_hooks = register_layer_output_hooks(teacher, backend, fm_layers)
    student_hooks = register_layer_output_hooks(student, backend, fm_layers)

    effective_steps_per_epoch = (
        min(args.max_steps_per_epoch, len(train_loader))
        if args.max_steps_per_epoch > 0
        else len(train_loader)
    )
    total_steps  = args.epochs * math.ceil(effective_steps_per_epoch / args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_steps)
    print(f"Effective steps/epoch: {effective_steps_per_epoch} | "
          f"Total optimizer steps: {total_steps} | Warmup: {warmup_steps}")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = (torch.cuda.amp.GradScaler()
                 if args.use_grad_scaler and args.amp_dtype == "fp16" else None)

    best_val_ppl = float("inf")
    init_val_ppl = None

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        stats = train_one_epoch(
            teacher=teacher, student=student, train_loader=train_loader,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            teacher_hooks=teacher_hooks, student_hooks=student_hooks,
            fm_layers=fm_layers, device=device, epoch=epoch, args=args)

        val_ppl = evaluate_perplexity(
            student, val_loader, device, max_batches=args.eval_max_batches)
        if init_val_ppl is None:
            init_val_ppl = val_ppl
        is_best = val_ppl < best_val_ppl
        if is_best:
            best_val_ppl = val_ppl

        save_checkpoint(args.output_dir, student, tokenizer, args, epoch, val_ppl, is_best)
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"loss={stats['loss']:.4f} kd={stats['kd']:.4f} "
              f"fm={stats['fm']:.6f} val_ppl={val_ppl:.4f} "
              f"time={(time.time()-t0)/60:.2f}min")

    # ---- Final eval ----
    if is_mc:
        trained_mc = evaluate_mc_datasets(
            student, tokenizer, raw_eval_data,
            device=device, backend=backend,
            mc_eval_max_samples=mc_max,
            num_fewshot=args.mmlu_num_fewshot,
            label="Trained adapter model")
        final_test_ppl = None
        _print_mc_comparison_table(teacher_mc, student_mc, trained_mc)
    else:
        print("Final test perplexity...")
        final_test_ppl = evaluate_perplexity_standard(
            student, tokenizer, raw_eval_data, text_column=text_column,
            max_length=args.ppl_max_length, stride=args.ppl_stride, backend=backend)
        trained_mc = {}
        print("\n=== Same-split comparison on TEST set ===")
        print(f"Teacher test ppl                  : {teacher_test_ppl:.4f}")
        print(f"Initial quantized student test ppl: {init_test_ppl:.4f}")
        print(f"Trained adapter model test ppl    : {final_test_ppl:.4f}")

    print(f"\nVal summary — init: {init_val_ppl:.4f}  best: {best_val_ppl:.4f}")

    # ---- Save CSV ----
    def _acc(mc, ds): return f"{mc[ds]['accuracy']:.2f}" if ds in mc else ""
    def _n(mc, ds):   return mc[ds]["n_examples"]         if ds in mc else ""

    append_run_result(
        results_file=args.results_file,
        row={
            "model_id":            args.model_id,
            "arch_backend":        args.arch_backend,
            "quant_backend":       args.quant_backend,
            "attn_adapter_kind":   args.attn_adapter_kind,
            "mlp_adapter_kind":    args.mlp_adapter_kind,
            "attn_r":              args.attn_r,
            "mlp_r":               args.mlp_r,
            "trainable_parameters": trainable_params_count,
            "teacher_test_ppl":
                f"{teacher_test_ppl:.6f}" if teacher_test_ppl is not None else "",
            "student_test_ppl":
                f"{init_test_ppl:.6f}" if init_test_ppl is not None else "",
            "trained_adapter_model_test_ppl":
                f"{final_test_ppl:.6f}" if final_test_ppl is not None else "",
            "teacher_mmlu_acc":       _acc(teacher_mc, "mmlu"),
            "student_mmlu_acc":       _acc(student_mc, "mmlu"),
            "trained_mmlu_acc":       _acc(trained_mc, "mmlu"),
            "mmlu_n_examples":        _n(trained_mc,   "mmlu"),
            "teacher_hellaswag_acc":  _acc(teacher_mc, "hellaswag"),
            "student_hellaswag_acc":  _acc(student_mc, "hellaswag"),
            "trained_hellaswag_acc":  _acc(trained_mc, "hellaswag"),
            "hellaswag_n_examples":   _n(trained_mc,   "hellaswag"),
            "gptq_model_dir":
                args.gptq_model_dir if args.quant_backend == "gptq" else "",
            "output_dir":           args.output_dir,
            "logical_total_params": total_params_count,
            "detected_bnb_4bit_layers":  quant_info["bnb_linear4bit_count"],
            "detected_bnb_8bit_layers":  quant_info["bnb_linear8bit_count"],
            "memory_footprint_bytes":    quant_info["memory_footprint_bytes"] or "",
        },
    )

    if is_mc:
        metric_str = " ".join(
            f"{ds}_teacher={_acc(teacher_mc,ds)} "
            f"{ds}_student={_acc(student_mc,ds)} "
            f"{ds}_trained={_acc(trained_mc,ds)}"
            for ds in sorted(set(
                list(teacher_mc) + list(student_mc) + list(trained_mc)))
        )
    else:
        metric_str = (f"teacher_ppl={teacher_test_ppl:.4f} "
                      f"student_ppl={init_test_ppl:.4f} "
                      f"trained_ppl={final_test_ppl:.4f}")

    print(f"\nSaved summary to: {args.results_file}")
    print(f"[SUMMARY] model={args.model_id} backend={args.quant_backend} "
          f"attn={args.attn_adapter_kind} mlp={args.mlp_adapter_kind} "
          f"trainable={trainable_params_count} {metric_str}")

    teacher_hooks.remove(); student_hooks.remove()
    cleanup_model(teacher); cleanup_model(student)


if __name__ == "__main__":
    main()