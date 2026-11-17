import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from cosmosr import trainer
from cosmosr.args import get_args
from cosmosr.cli import validate_checkpoint
from cosmosr.config import build_command
from cosmosr.data import FieldDataset
from cosmosr.models import WDistLoss
from cosmosr.models.styled_srsgan import G, D
from test_launcher import tiny_config, write_case


torch.set_num_threads(1)
WIDTH = {"chan_base": 8, "chan_min": 4, "chan_max": 8}


class ModuleFacade(torch.nn.Module):
    """Expose the DDP module attribute for CPU regression checks."""

    def __init__(self, module, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class DiagnosticMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backpropagated = False

    def forward(self, pred, target):
        loss = (pred - target).square().mean()

        def mark(gradient):
            self.backpropagated = True
            return gradient

        loss.register_hook(mark)
        return loss


class TrainingTests(unittest.TestCase):
    def test_dataset_preserves_disk_fields_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_case(root)
            lr = root / "LR_selected_100_train"
            hr = root / "HR_selected_100_train"
            raw = np.load(lr / "case_0/disp.npy").copy()
            dataset = FieldDataset(
                [str(lr / "*/disp.npy"), str(lr / "*/vel.npy")],
                [str(hr / "*/disp.npy"), str(hr / "*/vel.npy")],
                style_pattern=str(lr / "*/style.npy"),
                in_norms=["cosmology.dis", "cosmology.vel"],
                tgt_norms=["cosmology.dis", "cosmology.vel"],
                crop=8,
                in_pad=3,
                tgt_pad=0,
                scale_factor=2,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["input"].shape), (6, 14, 14, 14))
            self.assertEqual(tuple(item["target"].shape), (6, 16, 16, 16))
            np.testing.assert_array_equal(np.load(lr / "case_0/disp.npy"), raw)
            np.testing.assert_allclose(
                item["input"][:3, 3:11, 3:11, 3:11], raw / 60000, rtol=1e-6
            )

    def run_step(self, epoch, gp_interval=1):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_case(root)
            config = tiny_config()
            config["gp_interval"] = gp_interval
            args = get_args(build_command(config, root)[3:])
            args.in_chan = args.out_chan = [3, 3]
            args.style_size = 6
            loader = DataLoader(
                FieldDataset(
                    args.train_in_patterns,
                    args.train_tgt_patterns,
                    style_pattern=args.train_style_pattern,
                    in_norms=args.in_norms,
                    tgt_norms=args.tgt_norms,
                    crop=8,
                    in_pad=3,
                    tgt_pad=0,
                    scale_factor=2,
                ),
                batch_size=1,
            )
            generator = ModuleFacade(G(6, 6, 6, **WIDTH))
            critic = ModuleFacade(D(12, 1, 6, **WIDTH))
            before_g = copy.deepcopy(generator.state_dict())
            before_d = copy.deepcopy(critic.state_dict())
            opt_g = torch.optim.Adam(
                generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
            )
            opt_d = torch.optim.Adam(
                critic.parameters(), lr=args.adv_lr, betas=(0.0, 0.99)
            )
            sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g)
            sched_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d)
            mse = DiagnosticMSE()
            logger = Mock()
            cwd = Path.cwd()
            try:
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{root / 'gloo-store'}",
                    rank=0,
                    world_size=1,
                )
                os.chdir(root)
                with patch.object(
                    trainer, "plt_slices", return_value=Mock()
                ), patch.object(
                    trainer,
                    "wgan_grad_penalty",
                    wraps=trainer.wgan_grad_penalty,
                ) as gp, contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    loss = trainer.train(
                        epoch,
                        loader,
                        generator,
                        mse,
                        opt_g,
                        sched_g,
                        critic,
                        WDistLoss(),
                        opt_d,
                        sched_d,
                        logger,
                        torch.device("cpu"),
                        args,
                    )
                gp_calls = gp.call_count
            finally:
                os.chdir(cwd)
                if dist.is_initialized():
                    dist.destroy_process_group()
            self.assertTrue(torch.isfinite(loss).all())
            self.assertTrue(
                any(
                    not torch.equal(v, before_g[k])
                    for k, v in generator.state_dict().items()
                )
            )
            changed_d = any(
                not torch.equal(v, before_d[k]) for k, v in critic.state_dict().items()
            )
            saved_files = list(root.glob("state_*_b*.pt"))
            self.assertEqual(len(saved_files), 1)
            saved = torch.load(saved_files[0], map_location="cpu", weights_only=False)
            self.assertTrue(
                {"model", "adv_model", "optimizer", "adv_optimizer", "rng"}.issubset(
                    saved
                )
            )
            self.assertEqual(
                set(saved),
                {
                    "epoch",
                    "model",
                    "optimizer",
                    "scheduler",
                    "rng",
                    "min_loss",
                    "adv_model",
                    "adv_optimizer",
                    "adv_scheduler",
                },
            )
            restored = G(6, 6, 6, **WIDTH)
            restored.load_state_dict(saved["model"], strict=True)
            for key, value in generator.module.state_dict().items():
                torch.testing.assert_close(
                    restored.state_dict()[key], value, rtol=0, atol=0
                )
            return mse.backpropagated, changed_d, gp_calls

    def test_adversarial_step_updates_both_models_without_mse_backward(self):
        mse_backward, changed_d, gp_calls = self.run_step(2)
        self.assertFalse(mse_backward)
        self.assertTrue(changed_d)
        self.assertEqual(gp_calls, 1)

    def test_initial_epoch_retains_mse_warmup(self):
        mse_backward, changed_d, gp_calls = self.run_step(0)
        self.assertTrue(mse_backward)
        self.assertFalse(changed_d)
        self.assertEqual(gp_calls, 0)

    def test_gradient_penalty_keeps_lazy_interval(self):
        self.assertEqual(self.run_step(14, gp_interval=16)[2], 0)
        self.assertEqual(self.run_step(15, gp_interval=16)[2], 1)

    def test_worker_initialization_and_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_case(root)
            config = tiny_config()
            config.update(epochs=1, save_interval=0)
            args = get_args(build_command(config, root)[3:])
            args.gpus_per_node = args.world_size = args.nodes = 1
            original_import = trainer.import_attr

            def import_attr(name, *modules, **kwargs):
                if name == "styled_srsgan.G":
                    return lambda *a, **kw: G(*a, **kw, **WIDTH)
                if name == "styled_srsgan.D":
                    return lambda *a, **kw: D(*a, **kw, **WIDTH)
                return original_import(name, *modules, **kwargs)

            def init_group(*args):
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{root / 'gloo-store'}",
                    rank=0,
                    world_size=1,
                )

            cwd = Path.cwd()
            cpu = torch.device("cpu")
            torch_proxy = SimpleNamespace(**vars(torch))
            torch_proxy.device = lambda *a, **kw: cpu
            torch_proxy.cuda = SimpleNamespace(**vars(torch.cuda))
            torch_proxy.cuda.set_device = Mock()
            try:
                os.chdir(root)
                with patch.object(
                    trainer, "import_attr", side_effect=import_attr
                ), patch.object(
                    trainer,
                    "dist_init",
                    side_effect=init_group,
                ), patch.object(
                    trainer, "DistributedDataParallel", ModuleFacade
                ), patch.object(
                    trainer, "torch", torch_proxy
                ), patch.object(
                    trainer,
                    "SummaryWriter",
                    return_value=Mock(),
                ), patch.object(
                    trainer, "plt_slices", return_value=Mock()
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    trainer.gpu_worker(0, 0, args)
            finally:
                os.chdir(cwd)
                if dist.is_initialized():
                    dist.destroy_process_group()
            saved = torch.load(
                root / "state_1.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(saved["epoch"], 1)
            self.assertEqual(saved["optimizer"]["param_groups"][0]["lr"], config["lr"])
            self.assertEqual(
                saved["adv_optimizer"]["param_groups"][0]["lr"], config["adv_lr"]
            )
            self.assertEqual((root / "checkpoint.pt").resolve(), root / "state_1.pt")

    def test_inference_only_checkpoint_cannot_resume_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save({"model": {}}, path)
            with self.assertRaisesRegex(ValueError, "full training checkpoint"):
                validate_checkpoint(path, tiny_config())


if __name__ == "__main__":
    unittest.main()
