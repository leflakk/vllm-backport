# EXL3 (ExLlamaV3 trellis quants)

`--quantization exl3` serves [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
EXL3 checkpoints of GLM-5.3-Flash (`glm5next`) on Ampere and newer (sm_80+),
with vLLM's batching, prefix caching and OpenAI API. The method is detected
automatically from `config.json` (`quant_method: exl3`).

It is a port of the EXL3 overlay of
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(`overlay/exl3.py` @ `94ae731f`), extended to load native ExLlamaV3 quants such
as [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3).

## What runs where

| Part | turboderp 4.05bpw | Kernel |
| --- | --- | --- |
| Routed experts | EXL3 K4, mul1 codebook | fused `exl3_moe` (LinearEXL3 loop with `EXL3_FUSED_MOE=0`) |
| Shared experts, dense MLP, MLA `q_a/kv_a/q_b/o`, indexer `wq_b`, KDA `qkv`/`o` | EXL3 K5/K6, mul1 | LinearEXL3 custom op |
| `lm_head` | EXL3 K6 | LinearEXL3; at TP=8 (shard not 128-aligned) each rank runs the covering 128-aligned window and crops |
| KDA `b/f/g` projections, `kv_b_proj`, norms, embeddings | FP16/BF16 | native vLLM |

The per-module bitrate/codebook is read from the checkpoint's
`quantization_config.json` (`tensor_storage`), see
`vllm/model_executor/layers/quantization/exl3_layout.py`. The fused KDA
`qkv_proj` / `conv1d` tensors of native quants are split into q/k/v at load.

## Requirements and limits

- ExLlamaV3 **>= 1.4.5** (mul1 experts in `exl3_moe`); the images pin `v1.5.3`.
- Per-layer uniform expert bitrate. Checkpoints that mix K3/K4 across the
  experts of a layer (e.g. `satgeze/...-TR3-3.5bpw`) are rejected at load.
- Every TP shard of an EXL3 matrix must be 128-aligned (Hadamard blocks):
  TP 1/2/4/8 work for GLM-5.3-Flash (the TP=8 `lm_head` computes 96 extra rows).
- turboderp's quants ship **no MTP layer**: serve without speculative decoding.
- The vision tower is EXL3 too and is not supported yet: use
  `--language-model-only`.

## Build (RTX 3090, sm_86)

Quick path, on top of the published `lazymio/vllm-backport:latest-sm86` image
(compiles only the ExLlamaV3 kernels, checks that the base matches this branch):

```bash
docker build -f docker/Dockerfile.exl3-overlay \
    --build-arg max_jobs=16 -t vllm-backport:exl3-sm86 .
```

From source (full vLLM build, several hours):

```bash
docker build -f docker/Dockerfile --target vllm-openai \
    --build-arg torch_cuda_arch_list='8.6' \
    --build-arg exl3_cuda_arch_list='8.6' \
    --build-arg max_jobs=16 --build-arg nvcc_threads=2 \
    -t vllm-backport:exl3-sm86 .
```

## Serve (8x RTX 3090)

```bash
hf download turboderp/GLM-5.3-Flash-exl3 --revision 4.05bpw \
    --local-dir /models/GLM-5.3-Flash-exl3-4.05bpw

docker run --gpus all --ipc=host -p 8000:8000 \
    -v /models:/models vllm-backport:exl3-sm86 \
    /models/GLM-5.3-Flash-exl3-4.05bpw \
    --served-model-name GLM-5.3-Flash-EXL3 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --kv-cache-dtype fp8 \
    --language-model-only \
    --enforce-eager \
    --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
```

Memory is tight with the 4.05bpw quant at TP=8: about 21 GiB per GPU is
allocated once the weights are loaded, leaving ~2 GiB per 3090 for KV cache,
activations and CUDA graphs. Raise `--gpu-memory-utilization` (0.97) and keep
the context short, or use the 3.05bpw branch (~116 GiB) for headroom. Once
this works, drop `--enforce-eager`, then raise `--max-model-len`.

Load logs to look for: `EXL3 layout: 203 dense EXL3 modules, 42 routed-expert
layers`, `EXL3 mul1 trellis engaged for routed experts: K=4/4/4 ...
fused_moe=exl3_moe` and `[dense-exl3] 203 EXL3-dense modules loaded`.

## Tests

```bash
# CPU: layout mapping, q/k/v split, registration
pytest tests/quantization/test_exl3.py
# One GPU, real tensors: fused MoE vs loop vs dequantized reference, TP=8
# column/row splits, lm_head dequant, sm_86 cubin present
EXL3_TEST_CHECKPOINT=/models/GLM-5.3-Flash-exl3-4.05bpw \
    pytest tests/quantization/test_exl3_gpu.py -v
# Serving benchmark (structured / prose decode, C1 and C4)
BENCH_BASE_URL=http://127.0.0.1:8000 BENCH_MODEL=GLM-5.3-Flash-EXL3 \
    python benchmarks/exl3/bench_decode.py --phase c1 --out c1.json --structured
```

Environment knobs inherited from the port: `EXL3_FUSED_MOE=0` (LinearEXL3
loop instead of `exl3_moe`), `EXL3_TEMP_ROWS_FUSED` (fused kernel rows per
expert, default 128), `EXL3_MOE_ROW_TILE=1` (row-tiled prefill),
`GLM53_DENSE_EXL3_PREFILL_BF16` (keep BF16 copies of dense EXL3 modules for
large prefill batches, costs VRAM).
