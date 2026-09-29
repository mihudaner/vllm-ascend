import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.eplb.eplb_updator import EplbUpdator


class TestEplbUpdatorComputeAndSetMoeLoad(unittest.TestCase):
    def setUp(self):
        # ====================== 1. Mock environment ======================
        self.rank = 0
        self.world_size = 4
        self.device = torch.device("cpu")

        # mock dist
        p1 = patch("torch.distributed.get_rank", return_value=self.rank)
        p2 = patch("torch.distributed.get_world_size", return_value=self.world_size)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

        # ====================== 2. Mock comm group ======================
        self.mock_comm_group = MagicMock()

        def mock_all_gather(tensor, dim):
            gathered = torch.cat([tensor for _ in range(self.world_size)], dim=dim)
            return gathered

        self.mock_comm_group.all_gather = mock_all_gather

        p3 = patch("vllm_ascend.eplb.eplb_updator.get_dynamic_eplb_group", return_value=self.mock_comm_group)
        self.addCleanup(p3.stop)
        p3.start()

        # mock _PP in vllm.distributed.parallel_state (PP+EPLB support)
        # Patching the variable directly so that even the real get_pp_group()
        # (already imported into eplb_updator's namespace) reads a non-None _PP.
        self.mock_pp = MagicMock()
        self.mock_pp.rank_in_group = 0
        p4 = patch("vllm.distributed.parallel_state._PP", self.mock_pp)
        self.addCleanup(p4.stop)
        p4.start()

        publisher = patch("vllm_ascend.eplb.eplb_updator.AsyncHeatPublisher")
        self.addCleanup(publisher.stop)
        self.publisher_factory = publisher.start()

        # ====================== 3. Mock EplbUpdator ======================
        self.eplb_config = MagicMock()
        self.loader = MagicMock()
        self.eplb_process = MagicMock()
        self.process = MagicMock()
        self.eplb_process.shared_dict = {}

        self.updator = EplbUpdator(
            eplb_config=self.eplb_config, loader=self.loader, eplb_process=self.eplb_process, process=self.process
        )

        # ====================== 4. Mock adaptor ======================
        self.adaptor = MagicMock()
        self.adaptor.num_moe_layers = 4
        self.adaptor.num_dense_layers = 2
        self.mock_local_load = torch.randn(58, 100, 8, device=self.device)
        self.adaptor.get_rank_expert_workload.return_value = self.mock_local_load

        self.updator.set_adaptor(self.adaptor)

    def test_compute_and_set_moe_load_normal(self):
        self.updator.multi_stage = False

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (58, self.world_size, 100, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")
        self.assertEqual(moe_load.shape[1], self.world_size)

    def test_compute_and_set_moe_load_multi_stage(self):
        self.updator.multi_stage = True

        moe_load = self.updator.compute_and_set_moe_load()

        self.assertEqual(moe_load.shape, (100, 58, self.world_size, 8))
        self.assertTrue("moe_load" in self.updator.shared_dict)
        self.assertEqual(moe_load.device.type, "cpu")

    def test_warmup_reuses_publisher_without_waking_worker(self):
        self.updator.multi_stage = False
        self.updator.compute_and_set_moe_load()
        self.updator.compute_and_set_moe_load()
        self.publisher_factory.assert_called_once()
        self.eplb_process.planner_q.put.assert_not_called()

    def test_heat_step_enqueues_without_early_wakeup(self):
        self.updator.expert_heat_collection_interval = 600
        self.updator.algorithm_execution_interval = 50
        self.updator.cur_iterations = 599
        self.updator.multi_stage = True
        self.updator._heat_publisher = MagicMock()
        self.updator.forward_end()
        args = self.updator._heat_publisher.submit.call_args.args
        torch.testing.assert_close(args[1], self.mock_local_load.unsqueeze(1))
        self.assertEqual(args[0].shape, (58, self.world_size, 100, 8))
        self.assertTrue(args[2])
        self.eplb_process.planner_q.put.assert_not_called()
        self.assertEqual(self.updator.cur_iterations, 600)

    def test_publish_before_wakeup(self):
        payload = torch.ones(3)
        self.eplb_process.planner_q.put.side_effect = lambda _: self.assertIs(
            self.updator.shared_dict["moe_load"], payload
        )
        self.updator._publish_moe_load(payload)
        self.eplb_process.planner_q.put.assert_called_once_with(1)

    def test_update_boundary_wait_and_error_propagation(self):
        self.updator.expert_heat_collection_interval = 600
        self.updator.algorithm_execution_interval = 50
        self.updator.cur_iterations = 649
        publisher = self.updator._heat_publisher = MagicMock()
        order = []
        publisher.finish_before_update.side_effect = lambda: order.append("finish")
        self.eplb_process.block_update_q.get.side_effect = lambda: order.append("get")
        self.updator.forward_before()
        self.assertEqual(order, ["finish", "get"])
        self.eplb_process.block_update_q.get.reset_mock()
        publisher.check_error.side_effect = RuntimeError("D2H failed")
        with self.assertRaisesRegex(RuntimeError, "D2H failed"):
            self.updator.forward_before()
        self.eplb_process.block_update_q.get.assert_not_called()

    def test_shutdown_drains_publication_before_process_termination(self):
        publisher = self.updator._heat_publisher = MagicMock()
        order = []
        publisher.close.side_effect = lambda: order.append("close")
        self.process.terminate.side_effect = lambda: order.append("terminate")
        self.updator.shutdown()
        self.assertEqual(order, ["close", "terminate"])


if __name__ == "__main__":
    unittest.main()
