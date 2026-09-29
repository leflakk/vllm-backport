# SPDX-License-Identifier: Apache-2.0
"""Per-tensor layout of an ExLlamaV3 (EXL3) checkpoint, mapped onto the module
paths of vLLM's glm5next model.

ExLlamaV3 writes a ``tensor_storage`` ledger into ``quantization_config.json``
(one entry per checkpoint module with the stored tensor names and shapes);
``config.json`` only carries the summary. The trellis shape encodes the
bitrate (``[in/16, out/16, 16*K]``) and the codebook is the name of the int32
marker tensor (``mcg`` or ``mul1``).

Checkpoint modules are matched to served modules by their ``layers.N.<path>``
suffix (or ``lm_head``), so the result is independent of the multimodal
wrapper prefix (``language_model.model.``) and of the MTP name rewrite.

This module deliberately has no torch/vLLM imports at module scope so the
mapping can be unit-tested without a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CODEBOOKS = ("mcg", "mul1")

_LAYER_KEY_RE = re.compile(r"(?:^|\.)(layers\.\d+\..+)$")
_LAYER_IDX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$"
)

# checkpoint module suffix -> (served module suffix, shard ids it fills)
_DENSE_FORK: dict[str, tuple[str, tuple[int, ...]]] = {
    "mlp.gate_proj": ("mlp.gate_up_proj", (0,)),
    "mlp.up_proj": ("mlp.gate_up_proj", (1,)),
    "mlp.shared_experts.gate_proj": ("mlp.shared_experts.gate_up_proj", (0,)),
    "mlp.shared_experts.up_proj": ("mlp.shared_experts.gate_up_proj", (1,)),
    "self_attn.q_a_proj": ("self_attn.fused_qkv_a_proj", (0,)),
    "self_attn.kv_a_proj_with_mqa": ("self_attn.fused_qkv_a_proj", (1,)),
    # KDA: BF16-source packs store q/k/v separately, native ExLlamaV3 quants
    # store one fused qkv_proj (split at load, see glm5next load_weights).
    "self_attn.q_proj": ("self_attn.in_proj_qkvbfg_a", (0,)),
    "self_attn.k_proj": ("self_attn.in_proj_qkvbfg_a", (1,)),
    "self_attn.v_proj": ("self_attn.in_proj_qkvbfg_a", (2,)),
    "self_attn.qkv_proj": ("self_attn.in_proj_qkvbfg_a", (0, 1, 2)),
}

# Served modules built from several checkpoint modules: total shard count and
# the shards that stay BF16 when only part of the module is EXL3.
_FUSED_SHARDS: dict[str, tuple[int, tuple[int, ...]]] = {
    "mlp.gate_up_proj": (2, ()),
    "mlp.shared_experts.gate_up_proj": (2, ()),
    "self_attn.fused_qkv_a_proj": (2, ()),
    # b_proj, f_a_proj, g_a_proj are never quantized by ExLlamaV3.
    "self_attn.in_proj_qkvbfg_a": (6, (3, 4, 5)),
}


def served_key(prefix: str) -> str | None:
    """Normalize a checkpoint or vLLM module path.

    Returns ``layers.N.<path>`` or ``lm_head``; ``None`` for anything else
    (embeddings, vision tower, hyper-connection heads).
    """
    head = prefix.split(".", 2)[:2]
    if "visual" in head:
        return None
    m = _LAYER_KEY_RE.search(prefix)
    if m:
        return m.group(1)
    if prefix == "lm_head" or prefix.endswith(".lm_head"):
        return "lm_head"
    return None


def layer_index(prefix: str) -> int | None:
    m = _LAYER_IDX_RE.search(prefix)
    return int(m.group(1)) if m else None


@dataclass
class Exl3ExpertSpec:
    k_gate: int
    k_up: int
    k_down: int
    codebook: str

    @property
    def is_mcg(self) -> bool:
        return self.codebook == "mcg"


@dataclass
class Exl3Layout:
    # served key -> {"bits": K, "codebook": cb, "bf16_shards": [...]}
    dense: dict[str, dict[str, Any]] = field(default_factory=dict)
    # layer index -> routed expert spec
    experts: dict[int, Exl3ExpertSpec] = field(default_factory=dict)

    def dense_for(self, prefix: str) -> dict[str, Any] | None:
        key = served_key(prefix)
        return self.dense.get(key) if key is not None else None

    def experts_for(self, prefix: str) -> Exl3ExpertSpec | None:
        idx = layer_index(prefix)
        return self.experts.get(idx) if idx is not None else None


def _module_quant(name: str, entry: dict[str, Any]) -> tuple[int, str] | None:
    """(K, codebook) of one tensor_storage entry, None when not EXL3."""
    stored = entry.get("stored_tensors") or {}
    trellis = [v for k, v in stored.items() if k.endswith(".trellis")]
    if not trellis:
        return None
    words = int(trellis[0]["shape"][-1])
    if words % 16:
        raise ValueError(
            f"EXL3 {name}: half-integer bitrate ({words / 16}) is not supported"
        )
    codebooks = [cb for cb in CODEBOOKS if any(k.endswith("." + cb) for k in stored)]
    if len(codebooks) != 1:
        raise ValueError(f"EXL3 {name}: expected one codebook marker, got {codebooks}")
    return words // 16, codebooks[0]


def build_exl3_layout(tensor_storage: dict[str, dict[str, Any]]) -> Exl3Layout:
    """Map an ExLlamaV3 ``tensor_storage`` ledger onto glm5next module paths."""
    layout = Exl3Layout()
    expert_bits: dict[int, dict[str, set[int]]] = {}
    expert_cb: dict[int, set[str]] = {}
    served: dict[str, dict[str, Any]] = {}

    for name, entry in tensor_storage.items():
        q = _module_quant(name, entry)
        if q is None:
            continue
        k, cb = q
        m = _EXPERT_RE.search(name)
        if m:
            idx, proj = int(m.group(1)), m.group(3)
            expert_bits.setdefault(idx, {"gate": set(), "up": set(), "down": set()})
            expert_bits[idx][proj].add(k)
            expert_cb.setdefault(idx, set()).add(cb)
            continue
        key = served_key(name)
        if key is None:
            continue  # vision tower: served BF16-only / --language-model-only
        base, _, rest = key.partition(".")
        if key != "lm_head":
            # key = "layers.N.<rest>"
            n, _, rest = rest.partition(".")
            base = f"{base}.{n}."
        else:
            base, rest = "", "lm_head"
        target, shards = _DENSE_FORK.get(rest, (rest, (0,)))
        rec = served.setdefault(
            base + target,
            {"bits": set(), "codebook": set(), "shards": set(), "src": []},
        )
        rec["bits"].add(k)
        rec["codebook"].add(cb)
        rec["shards"].update(shards)
        rec["src"].append(name)

    for key, rec in served.items():
        if len(rec["bits"]) != 1 or len(rec["codebook"]) != 1:
            raise ValueError(
                f"EXL3 served module {key} mixes bitrates/codebooks "
                f"{sorted(rec['bits'])}/{sorted(rec['codebook'])} across {rec['src']}"
            )
        rest = key.split(".", 2)[2] if key.startswith("layers.") else key
        bf16_shards: list[int] = []
        if rest in _FUSED_SHARDS:
            n_shards, allowed_bf16 = _FUSED_SHARDS[rest]
            missing = sorted(set(range(n_shards)) - rec["shards"])
            if any(s not in allowed_bf16 for s in missing):
                raise ValueError(
                    f"EXL3 served module {key}: shards {missing} are not EXL3 "
                    f"in the checkpoint (sources {rec['src']})"
                )
            bf16_shards = missing
        layout.dense[key] = {
            "bits": next(iter(rec["bits"])),
            "codebook": next(iter(rec["codebook"])),
            "bf16_shards": bf16_shards,
        }

    for idx, bits in expert_bits.items():
        if any(len(v) != 1 for v in bits.values()) or len(expert_cb[idx]) != 1:
            ks = {p: sorted(v) for p, v in bits.items()}
            raise NotImplementedError(
                f"EXL3 layer {idx}: routed experts mix bitrates/codebooks "
                f"({ks}, {sorted(expert_cb[idx])}); only per-layer uniform "
                "expert K is supported"
            )
        layout.experts[idx] = Exl3ExpertSpec(
            k_gate=next(iter(bits["gate"])),
            k_up=next(iter(bits["up"])),
            k_down=next(iter(bits["down"])),
            codebook=next(iter(expert_cb[idx])),
        )
    return layout


def load_exl3_layout(model: str, revision: str | None) -> Exl3Layout | None:
    """Read ``quantization_config.json`` next to the checkpoint.

    Returns None when the file or its ``tensor_storage`` ledger is absent.
    """
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    qc = get_hf_file_to_dict("quantization_config.json", model, revision)
    if not qc or not qc.get("tensor_storage"):
        return None
    return build_exl3_layout(qc["tensor_storage"])


def exl3_or_none(quant_config: Any) -> Any:
    """`quant_config` if it is EXL3, else None.

    glm5next builds its MLA/KDA projections unquantized because FP8/NVFP4
    checkpoints keep them BF16; native ExLlamaV3 quants store them as EXL3.
    Exl3Config still returns the unquantized method for every module its
    layout does not declare, so passing it through is safe.
    """
    if quant_config is not None and quant_config.get_name() == "exl3":
        return quant_config
    return None
