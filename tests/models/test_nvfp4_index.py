from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec, _validate_nvfp4_expert_index


SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^layers\.(?P<layer>\d+)\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj|gate_alias)\."
        r"(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={
        "gate_proj": "gate", "up_proj": "up", "down_proj": "down", "gate_alias": "gate",
    },
    layer_to_bank=lambda layer, config: layer,
    desc="test NVFP4 experts",
)
CONFIG = SimpleNamespace(num_layers=1, num_experts=2)


def _index():
    return {
        f"layers.0.experts.{expert}.{proj}.{kind}": "model.safetensors"
        for expert in range(CONFIG.num_experts)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale", "weight_scale_2")
    }


def test_nvfp4_index_rejects_missing_tensor_before_bank_allocation():
    index = _index()
    del index["layers.0.experts.1.down_proj.weight_scale_2"]

    with pytest.raises(ValueError, match=r"missing NVFP4 expert slot \(0, 1, 'down', 'weight_scale_2'\)"):
        _validate_nvfp4_expert_index(index, CONFIG, SPEC)


def test_nvfp4_index_rejects_alias_that_would_overwrite_an_expert_slot():
    index = _index()
    index["layers.0.experts.0.gate_alias.weight"] = "other.safetensors"

    with pytest.raises(ValueError, match="duplicate NVFP4 expert slot"):
        _validate_nvfp4_expert_index(index, CONFIG, SPEC)


def test_nvfp4_index_accepts_complete_expert_layout():
    _validate_nvfp4_expert_index(_index(), CONFIG, SPEC)
