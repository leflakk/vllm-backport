# SPDX-License-Identifier: Apache-2.0
"""GPU checks of the EXL3 path against real checkpoint tensors (one GPU).

Needs CUDA, the exllamav3_ext extension and a local EXL3 checkpoint:

    EXL3_TEST_CHECKPOINT=/models/GLM-5.3-Flash-exl3-4.05bpw \\
        pytest tests/quantization/test_exl3_gpu.py -v

TP=8 is checked by building every rank's shard on the single GPU and
comparing the recombined result with the unsharded module, which is what
vLLM's column split / row split + all-reduce does across GPUs.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

CKPT = os.environ.get("EXL3_TEST_CHECKPOINT")
TP = 8
N_EXPERTS = 32  # subset of one layer's experts, keeps the test light
MOE_LAYER = int(os.environ.get("EXL3_TEST_MOE_LAYER", "3"))
PREFIX = "model.language_model.layers"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not CKPT, reason="set EXL3_TEST_CHECKPOINT"),
]


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-6)).item()


@pytest.fixture(scope="module")
def exl3():
    pytest.importorskip("exllamav3_ext")
    from vllm.model_executor.layers.quantization import exl3 as mod

    return mod


@pytest.fixture(scope="module")
def ckpt():
    from safetensors import safe_open

    root = Path(CKPT)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    handles = {}

    def get(name: str) -> torch.Tensor:
        f = weight_map[name]
        if f not in handles:
            handles[f] = safe_open(root / f, framework="pt", device="cuda")
        return handles[f].get_tensor(name)

    def module(name: str) -> dict[str, torch.Tensor]:
        out = {}
        for s in ("trellis", "suh", "svh", "mcg", "mul1"):
            if f"{name}.{s}" in weight_map:
                out[s] = get(f"{name}.{s}")
        return out

    return module


def _linear(exl3, parts: dict[str, torch.Tensor]):
    cb = {k: parts[k].reshape(1) for k in ("mcg", "mul1") if k in parts}
    return exl3.make_linear_exl3(parts["trellis"], parts["suh"], parts["svh"], **cb)


def test_extension_has_cubin_for_this_gpu(exl3):
    import exllamav3_ext

    assert hasattr(exllamav3_ext, "exl3_moe")
    assert exl3.exl3_moe_api(exllamav3_ext.exl3_moe) == "v1"
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump not on PATH")
    major, minor = torch.cuda.get_device_capability()
    listing = subprocess.run(
        [cuobjdump, "--list-elf", exllamav3_ext.__file__],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f"sm_{major}{minor}" in listing, listing[:2000]


def _moe_state(exl3, ckpt, tp_rank: int, tp_size: int) -> torch.nn.Module:
    """Minimal RoutedExperts stand-in, as Exl3MoEMethod builds it."""
    layer = torch.nn.Module()
    inners = []
    cb = None
    for e in range(N_EXPERTS):
        pack = {}
        for proj in ("gate", "up", "down"):
            parts = ckpt(f"{PREFIX}.{MOE_LAYER}.mlp.experts.{e}.{proj}_proj")
            shard = exl3.shard_exl3_row if proj == "down" else exl3.shard_exl3_col
            parts = {
                k: (
                    shard(v, k, tp_rank, tp_size)
                    if k in ("trellis", "suh", "svh")
                    else v
                )
                for k, v in parts.items()
            }
            cb = "mcg" if "mcg" in parts else "mul1"
            pack[proj] = _linear(exl3, parts)
        inners.append(pack)
    gate0, down0 = inners[0]["gate"], inners[0]["down"]
    layer.w13_trellis = torch.nn.Parameter(gate0.trellis, requires_grad=False)
    layer._exl3_hidden_size = gate0.in_features
    layer._exl3_intermediate_local = gate0.out_features
    k_gate, k_down = int(gate0.K), int(down0.K)
    layer._exl3_bits = k_gate
    layer._exl3_moe_spec = (k_gate, k_gate, k_down, cb == "mcg")
    layer._exl3_inners = inners
    exl3.build_exl3_fused_state(layer, inners)
    return layer


def _moe_reference(exl3, ckpt, x, ids, w, limit):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in ids.unique().tolist():
        lin = {
            p: _linear(exl3, ckpt(f"{PREFIX}.{MOE_LAYER}.mlp.experts.{e}.{p}_proj"))
            for p in ("gate", "up", "down")
        }
        wt = {p: lin[p].get_weight_tensor().float() for p in lin}  # [in, out]
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok].float()
        g, u = h @ wt["gate"], h @ wt["up"]
        a = F.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)
        out.index_add_(0, tok, (a @ wt["down"]) * w[tok, slot, None].float())
    return out


@pytest.mark.parametrize("tokens", [1, 8, 200])
def test_moe_fused_matches_loop_reference_and_tp8(exl3, ckpt, tokens):
    torch.manual_seed(0)
    limit = 10.0
    hidden = 4096
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    ids = torch.stack(
        [torch.randperm(N_EXPERTS, device="cuda")[:8] for _ in range(tokens)]
    )
    w = torch.rand(tokens, 8, device="cuda").softmax(-1)

    full = _moe_state(exl3, ckpt, 0, 1)
    fused = exl3.apply_exl3_experts(x, ids, w, full, limit=limit, fused=True)
    loop = exl3.apply_exl3_experts(x, ids, w, full, limit=limit, fused=False)
    ref = _moe_reference(exl3, ckpt, x, ids, w, limit)
    assert _rel_err(fused, loop) < 2e-2
    assert _rel_err(fused, ref) < 3e-2

    tp_sum = torch.zeros_like(fused, dtype=torch.float32)
    for r in range(TP):
        rank = _moe_state(exl3, ckpt, r, TP)
        tp_sum += exl3.apply_exl3_experts(x, ids, w, rank, limit=limit).float()
        del rank
    assert _rel_err(tp_sum, fused) < 3e-2


def test_kda_qkv_split_column_parallel_tp8(exl3, ckpt):
    from vllm.models.glm5next.nvidia.model import _split_fused_kda_weights

    base = f"{PREFIX}.0.self_attn."
    fused = ckpt(base + "qkv_proj")
    x = torch.randn(4, 4096, device="cuda", dtype=torch.float16)
    y_full = _linear(exl3, fused).forward(x, {}, out_dtype=torch.float32)

    parts: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in _split_fused_kda_weights(
        [(base + "qkv_proj." + s, v) for s, v in fused.items()]
    ):
        mod, suffix = name.rsplit(".", 2)[-2:]
        parts.setdefault(mod, {})[suffix] = t
    width = y_full.shape[1] // 3
    for i, mod in enumerate(("q_proj", "k_proj", "v_proj")):
        for r in range(TP):
            shard = {
                k: exl3.shard_exl3_col(v, k, r, TP) if k != "mul1" else v
                for k, v in parts[mod].items()
            }
            y = _linear(exl3, shard).forward(x, {}, out_dtype=torch.float32)
            lo = i * width + r * width // TP
            assert _rel_err(y, y_full[:, lo : lo + width // TP]) < 1e-2, (mod, r)


def test_o_proj_row_parallel_tp8(exl3, ckpt):
    parts = ckpt(f"{PREFIX}.3.self_attn.o_proj")
    lin = _linear(exl3, parts)
    x = torch.randn(4, lin.in_features, device="cuda", dtype=torch.float16)
    y_full = lin.forward(x, {}, out_dtype=torch.float32)
    step = lin.in_features // TP
    y = torch.zeros_like(y_full)
    for r in range(TP):
        shard = {
            k: exl3.shard_exl3_row(v, k, r, TP) if k in ("trellis", "suh") else v
            for k, v in parts.items()
        }
        xs = x[:, r * step : (r + 1) * step].contiguous()
        y += _linear(exl3, shard).forward(xs, {}, out_dtype=torch.float32)
    assert _rel_err(y, y_full) < 1e-2


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_lm_head_aligned_window_tp8(exl3, ckpt, rank):
    """TP=8 lm_head shards (19360 rows) are not 128-aligned: each rank runs
    the covering 128-aligned window as EXL3 and crops, like Exl3LinearMethod."""
    parts = ckpt("lm_head")
    lin = _linear(exl3, parts)
    x = torch.randn(2, lin.in_features, device="cuda", dtype=torch.float16)
    y_full = lin.forward(x, {}, out_dtype=torch.float32)
    rows = lin.out_features // TP
    lo = rank * rows
    w_lo, w_hi = lo // 128 * 128, min(-(-(lo + rows) // 128) * 128, lin.out_features)
    window = dict(parts)
    window["trellis"] = parts["trellis"][:, w_lo // 16 : w_hi // 16].contiguous()
    window["svh"] = parts["svh"][w_lo:w_hi].contiguous()
    y = _linear(exl3, window).forward(x, {}, out_dtype=torch.float32)
    y = y[:, lo - w_lo : lo - w_lo + rows]
    assert _rel_err(y, y_full[:, lo : lo + rows]) < 1e-2
