# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.fixture
def aux_runner(monkeypatch):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(model="test", multimodal_config=None)
    runner.ascend_config = SimpleNamespace(mix_placement=False)
    runner.eplb_enable = False
    runner.vllm_config = SimpleNamespace(quant_config=None)
    runner.drafter = None
    runner.lora_config = None
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.use_aux_hidden_state_outputs = True
    runner.speculative_config = SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_config=SimpleNamespace(architectures=["DSparkDraftModel"]),
    )
    runner._eagle3_uses_aux_hidden_state = lambda: False
    runner._get_eagle3_aux_layers_from_config = lambda: (41, 42, 43)
    inner = SimpleNamespace(start_layer=33, end_layer=43, make_empty_intermediate_tensors=object())
    model = SimpleNamespace(
        model=inner,
        named_parameters=lambda: (),
        set_aux_hidden_state_layers=Mock(),
        get_eagle3_default_aux_hidden_state_layers=lambda: (41, 42, 43),
    )
    runner.get_model = lambda: model
    pp_group = SimpleNamespace(world_size=4, is_last_rank=True)
    propagate = Mock(return_value=True)
    prefix = "vllm_ascend.worker.model_runner_v1"
    monkeypatch.setattr(f"{prefix}.get_model", lambda **kwargs: model)
    monkeypatch.setattr(f"{prefix}.get_pp_group", lambda: pp_group)
    monkeypatch.setattr(f"{prefix}.DeviceMemoryProfiler", lambda: nullcontext(SimpleNamespace(consumed_memory=0)))
    monkeypatch.setattr(f"{prefix}.get_offloader", Mock())
    monkeypatch.setattr(f"{prefix}.supports_multimodal_pruning", lambda model: False)
    monkeypatch.setattr(f"{prefix}.set_moe_quant_mismatch", lambda *args: None)
    monkeypatch.setattr("vllm.model_executor.models.interfaces.supports_eagle3", lambda model: True)
    monkeypatch.setattr("vllm_ascend.patch.worker.patch_eagle3_pp_aux.patch_eagle3_pp_aux_propagation", propagate)
    return runner, model, pp_group, propagate


def test_dspark_last_stage_collects_aux_locally(aux_runner):
    runner, model, _, propagate = aux_runner
    runner.load_model()
    model.set_aux_hidden_state_layers.assert_called_once_with((41, 42, 43))
    propagate.assert_not_called()


@pytest.mark.parametrize("layers", [(33, 42, 43), (41, 42, 44)])
def test_dspark_rejects_aux_outside_last_stage_before_setting_layers(aux_runner, layers):
    runner, model, _, propagate = aux_runner
    runner._get_eagle3_aux_layers_from_config = lambda: layers
    with pytest.raises(ValueError, match="last stage"):
        runner.load_model()
    model.set_aux_hidden_state_layers.assert_not_called()
    propagate.assert_not_called()


def test_dspark_nonlast_stage_skips_aux(aux_runner):
    runner, model, pp_group, propagate = aux_runner
    pp_group.is_last_rank = False
    runner.load_model()
    model.set_aux_hidden_state_layers.assert_not_called()
    propagate.assert_not_called()


def test_pp1_keeps_aux_configuration(aux_runner):
    runner, model, pp_group, propagate = aux_runner
    pp_group.world_size = 1
    runner.load_model()
    model.set_aux_hidden_state_layers.assert_called_once_with((41, 42, 43))
    propagate.assert_not_called()


@pytest.mark.parametrize("pp_size", [1, 4])
def test_preserves_dspark_capture_mode_hook(aux_runner, pp_size):
    runner, model, pp_group, _ = aux_runner
    pp_group.world_size = pp_size
    model.set_dspark_aux_capture_materialized = Mock()
    runner._draft_uses_qwen3_gqa_dspark = lambda: True
    runner.load_model()
    model.set_dspark_aux_capture_materialized.assert_called_once_with(True)


def test_eagle3_keeps_cross_stage_aux_propagation(aux_runner):
    runner, model, _, propagate = aux_runner
    runner.speculative_config.use_dspark = lambda: False
    runner._eagle3_uses_aux_hidden_state = lambda: True
    runner._get_eagle3_aux_layers_from_config = lambda: (2, 20, 42)
    runner.load_model()
    model.set_aux_hidden_state_layers.assert_called_once_with((2, 20, 42))
    propagate.assert_called_once_with(model.model)
    assert model.make_empty_intermediate_tensors is model.model.make_empty_intermediate_tensors
