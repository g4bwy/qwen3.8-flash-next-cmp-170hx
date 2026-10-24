#!/usr/bin/env python3
"""CPU resolution test: does the base tree accept ct_config output and route each
module family to the intended scheme/method? No GPU, no weights."""
import json
import sys

sys.path.insert(0, __import__("os").environ.get("VLLM_TREE", "/home/gab/test/wt-oct4"))
sys.path.insert(0, "/home/gab/test/scratch/w8a16")

import torch  # noqa: E402
from ct_config import build_ct_config  # noqa: E402

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E402
    CompressedTensorsConfig)

SRC = (sys.argv[2] if len(sys.argv) > 2 else
       "/home/gab/models/Qwen/Qwen3.8-Flash-Next-FP8")
official = json.load(open(f"{SRC}/config.json"))
mods = official["quantization_config"]["modules_to_not_convert"]
from ct_config import CONVERT_RE  # noqa: E402
if len(sys.argv) > 1:
    cfg = json.load(open(sys.argv[1]))["quantization_config"]
    held = cfg["ignore"]
    print(f"resolving real config file: {sys.argv[1]} ({len(held)} ignore)")
    qc = CompressedTensorsConfig.from_config(cfg)
else:
    held = [m for m in mods if not CONVERT_RE.match(m + ".weight")]
    print(f"ignore entries: {len(held)} (of {len(mods)})")
    qc = CompressedTensorsConfig.from_config(build_ct_config(held))
# The runtime fusion is a property of the model, not of this test. vLLM expands a
# fused module through the model class's packed_modules_mapping, and
# should_ignore_layer raises when that expansion mixes ignored and quantized
# shards. Hardcoding the mapping here once hid exactly that failure (upstream
# folded self_attn.indexer.index_qk_proj into qkv_proj), so read it from the model.
from vllm.models.qwen4_exp.nvidia.model import (  # noqa: E402
    Qwen4ExpForConditionalGeneration,
)

qc.packed_modules_mapping = dict(
    Qwen4ExpForConditionalGeneration.packed_modules_mapping
)
print("packed_modules_mapping[qkv_proj]:",
      qc.packed_modules_mapping.get("qkv_proj"))
print("config parsed ok")


def fake(cls):
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    return obj


# mirror hf_to_vllm_mapper: runtime names after "model.language_model." ->
# "model." (and mtp keeps its own namespace)
def rt(ckpt_name: str) -> str:
    return ckpt_name.replace("model.language_model.", "model.", 1)


cases = [
    ("MergedColumn gate", fake(MergedColumnParallelLinear),
     rt("model.language_model.layers.0.mlp.shared_expert.gate_proj"), "int8"),
    ("gate_up fused", fake(MergedColumnParallelLinear),
     "model.layers.0.mlp.shared_expert.gate_up_proj", "int8"),
    ("RowProj out_proj", fake(RowParallelLinear),
     rt("model.language_model.layers.0.linear_attn.out_proj"), "int8"),
    ("QKV fused", fake(QKVParallelLinear),
     "model.layers.3.self_attn.qkv_proj", "int8"),
    ("o_proj", fake(RowParallelLinear),
     rt("model.language_model.layers.3.self_attn.o_proj"), "int8"),
    ("MoE experts", fake(RoutedExperts),
     rt("model.language_model.layers.0.mlp.experts"), "fp8"),
    ("MoE draft", fake(RoutedExperts),
     "mtp.layers.0.mlp.experts", "fp8"),
    ("draft qkv bf16", fake(QKVParallelLinear),
     "mtp.layers.0.self_attn.qkv_proj", None),
    ("draft shared bf16", fake(MergedColumnParallelLinear),
     "mtp.layers.0.mlp.shared_expert.gate_up_proj", None),
    ("draft dense bf16", fake(RowParallelLinear),
     "mtp.layers.0.self_attn.o_proj", None),
    ("held router", fake(MergedColumnParallelLinear),
     rt("model.language_model.layers.0.mlp.gate"), None),
    ("held hc", fake(MergedColumnParallelLinear),
     rt("model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up"), None),
    ("held gdn in_proj", fake(MergedColumnParallelLinear),
     rt("model.language_model.layers.0.linear_attn.in_proj_qkv"), None),
    ("indexer int8 (fused with qkv)", fake(MergedColumnParallelLinear),
     rt("model.language_model.layers.3.self_attn.indexer.index_qk_proj"), "int8"),
    # The fused runtime module must resolve as a unit: if the indexer shard and
    # q/k/v disagree, should_ignore_layer raises instead of returning.
    ("qkv fused resolves whole", fake(QKVParallelLinear),
     "model.layers.3.self_attn.qkv_proj", "int8"),
]

fails = 0
for name, layer, prefix, expect in cases:
    try:
        if expect == "fp8":
            sd = qc.get_scheme_dict(layer, prefix)
            w = None if not sd else sd["weights"]
            ok = bool(w) and w.type == "float" and w.strategy == "block"
            print(f"{name:18s} scheme-dict: type={getattr(w,'type',None)} "
                  f"bits={getattr(w,'num_bits',None)} strat={getattr(w,'strategy',None)} "
                  f"act={None if not sd or sd.get('input_activations') is None else sd['input_activations'].strategy} "
                  f"{'PASS' if ok else 'CHECK'}")
            fails += 0 if ok else 1
            continue
        m = qc.get_quant_method(layer, prefix)
    except Exception as exc:  # noqa: BLE001
        print(f"{name:18s} EXC {type(exc).__name__}: {exc}")
        fails += 1
        continue
    cls = type(m).__name__ if m else "None(unquantized)"
    scheme = getattr(layer, "scheme", None)
    scls = type(scheme).__name__ if scheme else "-"
    ok = "?"
    if expect == "int8":
        ok = "WNA16" in scls or "WNa16" in scls
    elif expect is None:
        ok = m is None or "Unquantized" in cls
    print(f"{name:18s} method={cls:38s} scheme={scls:32s} {'PASS' if ok else 'CHECK'}")
    fails += 0 if ok else 1

# --- PLE embedding method selection (needs patched base tree) ---
ple_prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
try:
    from vllm.models.qwen4_exp.common.ngram_embedding import (
        Qwen4ExpPLEEmbeddingMethod)
    m = Qwen4ExpPLEEmbeddingMethod.from_quant_config(qc, ple_prefix, None)
    ok = type(m).__name__ == "Qwen4ExpPLEFp8EmbeddingMethod"
    print(f"PLE ngram       method={type(m).__name__} {'PASS' if ok else 'CHECK'}")
    fails += 0 if ok else 1
except NotImplementedError as exc:
    print(f"PLE ngram       RAISED {type(exc).__name__}  CHECK")
    fails += 1

sys.exit(1 if fails else 0)
