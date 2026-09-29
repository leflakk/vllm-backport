# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the EXL3 checkpoint layout and glm5next weight remapping.

GPU checks against real checkpoint tensors live in test_exl3_gpu.py.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3_layout import (
    build_exl3_layout,
    served_key,
)


def _exl3(name: str, in_f: int, out_f: int, k: int, cb: str = "mul1") -> dict:
    """tensor_storage entry as written by ExLlamaV3."""
    return {
        "stored_tensors": {
            f"{name}.trellis": {"shape": [in_f // 16, out_f // 16, 16 * k]},
            f"{name}.suh": {"shape": [in_f]},
            f"{name}.svh": {"shape": [out_f]},
            f"{name}.{cb}": {"shape": []},
        }
    }


def _bf16(name: str) -> dict:
    return {"stored_tensors": {f"{name}.weight": {"shape": [8, 8]}}}


def _turboderp_like_storage() -> dict:
    """One dense-MLP KDA layer, one MLA MoE layer and the head, named like
    turboderp/GLM-5.3-Flash-exl3 (model.language_model.* keys)."""
    p = "model.language_model.layers"
    ts = {
        f"{p}.0.self_attn.qkv_proj": _exl3("q", 4096, 24576, 6),
        f"{p}.0.self_attn.b_proj": _bf16("b"),
        f"{p}.0.self_attn.o_proj": _exl3("o", 8192, 4096, 6),
        f"{p}.0.mlp.gate_proj": _exl3("g", 4096, 12288, 5),
        f"{p}.0.mlp.up_proj": _exl3("u", 4096, 12288, 5),
        f"{p}.0.mlp.down_proj": _exl3("d", 12288, 4096, 5),
        f"{p}.3.self_attn.q_a_proj": _exl3("qa", 4096, 1536, 6),
        f"{p}.3.self_attn.kv_a_proj_with_mqa": _exl3("kva", 4096, 512, 6),
        f"{p}.3.self_attn.q_b_proj": _exl3("qb", 1536, 16384, 6),
        f"{p}.3.self_attn.indexer.wq_b": _exl3("wq", 1536, 4096, 6),
        f"{p}.3.mlp.shared_experts.gate_proj": _exl3("sg", 4096, 2048, 6),
        f"{p}.3.mlp.shared_experts.up_proj": _exl3("su", 4096, 2048, 6),
        f"{p}.3.mlp.shared_experts.down_proj": _exl3("sd", 2048, 4096, 6),
        "lm_head": _exl3("lm", 4096, 154880, 6),
        "model.visual.blocks.0.attn.q_proj": _exl3("v", 1024, 1024, 6),
        "model.language_model.embed_tokens": _bf16("e"),
    }
    for e in range(4):
        for proj, (i, o) in {
            "gate": (4096, 2048),
            "up": (4096, 2048),
            "down": (2048, 4096),
        }.items():
            ts[f"{p}.3.mlp.experts.{e}.{proj}_proj"] = _exl3("x", i, o, 4)
    return ts


def test_served_key_is_wrapper_independent():
    ckpt = "model.language_model.layers.3.self_attn.o_proj"
    runtime = "language_model.model.layers.3.self_attn.o_proj"
    assert served_key(ckpt) == served_key(runtime) == "layers.3.self_attn.o_proj"
    assert served_key("language_model.lm_head") == "lm_head"
    assert served_key("visual.blocks.0.attn.qkv") is None
    assert served_key("model.language_model.embed_tokens") is None


def test_layout_maps_native_exllamav3_quant():
    layout = build_exl3_layout(_turboderp_like_storage())

    kda_in = layout.dense["layers.0.self_attn.in_proj_qkvbfg_a"]
    assert kda_in == {"bits": 6, "codebook": "mul1", "bf16_shards": [3, 4, 5]}
    assert layout.dense["layers.0.mlp.gate_up_proj"]["bits"] == 5
    assert layout.dense["layers.3.self_attn.fused_qkv_a_proj"]["bf16_shards"] == []
    assert "layers.3.mlp.shared_experts.gate_up_proj" in layout.dense
    assert "layers.3.self_attn.indexer.wq_b" in layout.dense
    assert layout.dense["lm_head"]["bits"] == 6
    assert not any("visual" in k or "embed" in k for k in layout.dense)

    spec = layout.experts_for("language_model.model.layers.3.mlp.experts")
    assert (spec.k_gate, spec.k_up, spec.k_down, spec.codebook) == (4, 4, 4, "mul1")
    assert layout.experts_for("language_model.model.layers.0.mlp.experts") is None


def test_layout_rejects_mixed_expert_bitrates():
    ts = _turboderp_like_storage()
    ts["model.language_model.layers.3.mlp.experts.1.down_proj"] = _exl3(
        "x", 2048, 4096, 3
    )
    with pytest.raises(NotImplementedError, match="layer 3"):
        build_exl3_layout(ts)


def test_layout_rejects_partially_quantized_fused_module():
    ts = _turboderp_like_storage()
    del ts["model.language_model.layers.3.self_attn.kv_a_proj_with_mqa"]
    with pytest.raises(ValueError, match="fused_qkv_a_proj"):
        build_exl3_layout(ts)


def test_split_fused_kda_weights():
    from vllm.models.glm5next.nvidia.model import _split_fused_kda_weights

    trellis = torch.arange(2 * 6 * 4, dtype=torch.int16).reshape(2, 6, 4)
    svh = torch.arange(96, dtype=torch.float16)
    suh = torch.ones(32, dtype=torch.float16)
    conv = torch.arange(96 * 4, dtype=torch.float32).reshape(96, 1, 4)
    other = torch.zeros(3)
    base = "layers.0.self_attn."
    out = dict(
        _split_fused_kda_weights(
            [
                (base + "qkv_proj.trellis", trellis),
                (base + "qkv_proj.svh", svh),
                (base + "qkv_proj.suh", suh),
                (base + "conv1d.weight", conv),
                (base + "o_proj.trellis", other),
            ]
        )
    )

    assert torch.equal(out[base + "k_proj.trellis"], trellis[:, 2:4])
    assert torch.equal(out[base + "v_proj.svh"], svh[64:])
    assert out[base + "q_proj.suh"] is suh
    assert torch.equal(out[base + "k_conv1d.weight"], conv[32:64])
    assert out[base + "o_proj.trellis"] is other
    assert not any("qkv_proj" in k or ".conv1d." in k for k in out)


def test_exl3_is_registered_and_detected():
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    assert "exl3" in QUANTIZATION_METHODS
    assert get_quantization_config("exl3") is Exl3Config
    hf_cfg = {"quant_method": "exl3", "bits": 4.05, "codebook": "mul1"}
    assert Exl3Config.override_quantization_method(hf_cfg, None) == "exl3"

    cfg = Exl3Config.from_config(hf_cfg)
    assert cfg.bits is None and cfg.codebook == "mul1"
    assert Exl3Config.from_config({"quant_method": "exl3", "bits": 4}).bits == 4
