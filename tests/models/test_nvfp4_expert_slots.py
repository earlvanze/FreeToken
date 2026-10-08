import json
import re
from types import SimpleNamespace

import pytest
from freetoken.models import nvfp4_banks

SPEC = nvfp4_banks.Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^layers\.(?P<layer>\d+)\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj|gate_alias)\."
        r"(?P<kind>weight|weight_scale|weight_scale_2|packed)$"
    ),
    proj_to_role={
        "gate_proj": "gate", "up_proj": "up", "down_proj": "down", "gate_alias": "gate",
    },
    layer_to_bank=lambda layer, config: layer,
    desc="test NVFP4 experts",
    kind_map={"packed": "weight"},
)
CONFIG = SimpleNamespace(num_layers=1, num_experts=2)


def _weight_map():
    return {
        f"layers.0.experts.{expert}.{proj}.{kind}": "model.safetensors"
        for expert in range(CONFIG.num_experts)
        for proj in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale", "weight_scale_2")
    }


def _read_index(tmp_path, monkeypatch, weight_map):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    monkeypatch.setattr(nvfp4_banks, "download_hf_weight", lambda _: str(tmp_path))
    return nvfp4_banks.iter_nvfp4_expert_pieces("unused", CONFIG, SPEC, primary=False)


def test_nvfp4_index_accepts_complete_layout(tmp_path, monkeypatch):
    assert _read_index(tmp_path, monkeypatch, _weight_map()) is not None


def test_nvfp4_index_rejects_alias_masking_missing_slot(tmp_path, monkeypatch):
    weight_map = _weight_map()
    del weight_map["layers.0.experts.1.down_proj.weight_scale_2"]
    weight_map["layers.0.experts.0.gate_alias.packed"] = "model.safetensors"
    with pytest.raises(ValueError, match="duplicate NVFP4 expert slot"):
        _read_index(tmp_path, monkeypatch, weight_map)


def test_nvfp4_index_rejects_out_of_range_expert(tmp_path, monkeypatch):
    weight_map = _weight_map()
    del weight_map["layers.0.experts.1.down_proj.weight_scale_2"]
    weight_map["layers.0.experts.2.down_proj.weight_scale_2"] = "model.safetensors"
    with pytest.raises(ValueError, match="expert 2.*outside"):
        _read_index(tmp_path, monkeypatch, weight_map)


def test_nvfp4_index_reports_missing_slot(tmp_path, monkeypatch):
    weight_map = _weight_map()
    del weight_map["layers.0.experts.1.down_proj.weight_scale_2"]
    expected = r"missing NVFP4 expert slot \(0, 1, 'down_global'\)"
    with pytest.raises(ValueError, match=expected):
        _read_index(tmp_path, monkeypatch, weight_map)
