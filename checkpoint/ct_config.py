#!/usr/bin/env python3
"""Compressed-tensors mixed config for the W8A16-dense checkpoint.

group_0: routed experts (main + mtp draft) keep the official fp8 e4m3
         block-128 bytes; FusedMoE dispatch + block weight strategy.
group_1: 228 main-body dense modules plus the 12 QSA indexer projections the
         fused qkv_proj folds in become int8 symmetric group-32, bf16
         activations (weight-only W8A16). A fused module resolves as one unit,
         so the indexer cannot stay bf16 while q/k/v go int8.
ignore:  the official release's remaining modules_to_not_convert.
PLE ngram fp8 is selected by text_config.ple_embedding_dtype in the model
code, before any quant-config type check, so it needs no group.
"""
import re

CONVERT_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:linear_attn\.out_proj"
    r"|self_attn\.(?:q|k|v|o)_proj"
    # vLLM folds the QSA indexer into the fused qkv_proj module, and a fused
    # module must carry one scheme for all its shards, so the indexer travels
    # with q/k/v. It stays int8 weight-only, the same recipe as its shards.
    r"|self_attn\.indexer\.index_qk_proj"
    r"|mlp\.shared_expert\.(?:gate|up|down)_proj)\.weight$"
)

INT8_TARGETS = [
    # Runtime prefixes are post-hf_to_vllm_mapper ("model.layers.N.*"); the
    # checkpoint spelling "model.language_model.layers.N.*" also contains
    # "model.layers.", so these match either naming. The mtp draft layers
    # ("mtp.layers.N.*") never contain "model.layers." and stay bf16.
    # Fused runtime names (qkv_proj, gate_up_proj) plus component fallbacks.
    r"re:.*model\.layers\.\d+\.linear_attn\.out_proj$",
    r"re:.*model\.layers\.\d+\.self_attn\.qkv_proj$",
    r"re:.*model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
    r"re:.*model\.layers\.\d+\.self_attn\.indexer\.index_qk_proj$",
    r"re:.*model\.layers\.\d+\.mlp\.shared_expert\.gate_up_proj$",
    r"re:.*model\.layers\.\d+\.mlp\.shared_expert\.(gate|up|down)_proj$",
]


def _wa(**over):
    d = dict(num_bits=None, type=None, strategy=None, symmetric=True,
             dynamic=False, group_size=None, block_structure=None,
             actorder=None, observer="minmax", observer_kwargs={})
    d.update(over)
    return d


def build_ct_config(held_modules: list[str]) -> dict:
    ignore = []
    for m in held_modules:
        ignore.append(m)
        if m.startswith("model.language_model."):
            ignore.append("model." + m[len("model.language_model."):])
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "global_compression_ratio": None,
        "kv_cache_scheme": None,
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": ["RoutedExperts", r"re:.*ple_embedding\.ngram_embedding$"],
                "weights": _wa(num_bits=8, type="float", strategy="block",
                               block_structure=[128, 128]),
                "input_activations": _wa(
                    num_bits=8, type="float", strategy="token", dynamic=True,
                    observer=None),
                "output_activations": None,
            },
            "group_1": {
                "format": "pack-quantized",
                "targets": INT8_TARGETS,
                "weights": _wa(num_bits=8, type="int", strategy="group",
                               group_size=32),
                "input_activations": None,
                "output_activations": None,
            },
        },
        "ignore": ignore,
        "sparsity_config": {},
        "transform_config": {},
    }
