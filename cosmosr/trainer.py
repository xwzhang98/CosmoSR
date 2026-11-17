"""State56 conditional WGAN-GP training."""

import os
import socket
import time
import sys
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .data import FieldDataset, DistFieldSampler
from . import models
from .models import narrow_cast, resample, lag2eul, wgan_grad_penalty
from .utils import import_attr, load_model_state_dict, plt_slices
from datetime import timedelta

ckpt_link = "checkpoint.pt"


def node_worker(args):
    if "SLURM_STEP_NUM_NODES" in os.environ:
        args.nodes = int(os.environ["SLURM_STEP_NUM_NODES"])
    elif "SLURM_JOB_NUM_NODES" in os.environ:
        args.nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    else:
        args.nodes = 1
    args.gpus_per_node = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus_per_node
    node = int(os.environ.get("SLURM_NODEID", "0"))
    if args.gpus_per_node < 1:
        raise RuntimeError("GPU not found on node {}".format(node))
    spawn(gpu_worker, args=(node, args), nprocs=args.gpus_per_node)


def gpu_worker(local_rank, node, args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = args.gpus_per_node * node + local_rank
    torch.manual_seed(args.seed + rank)
    print(
        f"Rank {rank}: Setting up distributed training with local_rank {local_rank}",
        flush=True,
    )
    dist_init(rank, args, local_rank)
    train_dataset = FieldDataset(
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        style_pattern=args.train_style_pattern,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=args.augment,
        aug_shift=args.aug_shift,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    train_sampler = DistFieldSampler(
        train_dataset,
        shuffle=True,
        div_data=args.div_data,
        div_shuffle_dist=args.div_shuffle_dist,
    )
    _num_workers = args.loader_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True,
        prefetch_factor=3 if _num_workers > 0 else None,
        persistent_workers=True if _num_workers > 0 else False,
        drop_last=True,
    )
    args.in_chan = train_dataset.in_chan
    args.out_chan = train_dataset.tgt_chan
    args.style_size = train_dataset.style_size
    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(
        sum(args.in_chan),
        sum(args.out_chan),
        style_size=args.style_size,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    model.to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    criterion = import_attr(args.criterion, nn, models, callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)
    optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
    optimizer = optimizer(model.parameters(), lr=args.lr, **args.optimizer_args)
    scheduler_args_filtered = {
        k: v for (k, v) in args.scheduler_args.items() if k != "verbose"
    }
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **scheduler_args_filtered
    )
    adv_model = import_attr(args.adv_model, models, callback_at=args.callback_at)
    adv_model = adv_model(
        sum(args.in_chan + args.out_chan),
        1,
        style_size=args.style_size,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    adv_model.to(device)
    adv_model = DistributedDataParallel(
        adv_model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    adv_criterion = import_attr(
        args.adv_criterion, nn, models, callback_at=args.callback_at
    )
    adv_criterion = adv_criterion()
    adv_criterion.to(device)
    adv_optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
    _adv_params = list(adv_model.parameters())
    adv_optimizer = adv_optimizer(
        _adv_params, lr=args.adv_lr, **args.adv_optimizer_args
    )
    scheduler_args_filtered = {
        k: v for (k, v) in args.scheduler_args.items() if k != "verbose"
    }
    adv_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        adv_optimizer, **scheduler_args_filtered
    )
    if (
        args.load_state == ckpt_link
        and (not os.path.isfile(ckpt_link))
        or not args.load_state
    ):
        start_epoch = 0
        if rank == 0:
            min_loss = None
    else:
        state = torch.load(args.load_state, map_location=device, weights_only=False)
        start_epoch = state["epoch"]
        load_model_state_dict(
            model.module, state["model"], strict=args.load_state_strict
        )
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if "adv_model" in state:
            adv_state = state["adv_model"]
            load_model_state_dict(
                adv_model.module, adv_state, strict=args.load_state_strict
            )
        if "adv_optimizer" in state:
            adv_optimizer.load_state_dict(state["adv_optimizer"])
        if "adv_scheduler" in state:
            adv_scheduler.load_state_dict(state["adv_scheduler"])
        torch.set_rng_state(state["rng"].cpu())
        if rank == 0:
            min_loss = state["min_loss"]
            if "adv_model" not in state:
                min_loss = None
            print(
                "state at epoch {} loaded from {}".format(
                    state["epoch"], args.load_state
                ),
                flush=True,
            )
        del state
    torch.backends.cudnn.benchmark = True
    logger = None
    if rank == 0:
        logger = SummaryWriter()
    if rank == 0:
        print("pytorch {}".format(torch.__version__), flush=True)
        pprint(vars(args))
        sys.stdout.flush()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            adv_model,
            adv_criterion,
            adv_optimizer,
            adv_scheduler,
            logger,
            device,
            args,
        )
        epoch_loss = train_loss
        if rank == 0:
            logger.flush()
            if (
                min_loss is None or epoch_loss[0] < min_loss[0]
            ) and epoch >= args.adv_start:
                min_loss = epoch_loss
            state = {
                "epoch": epoch + 1,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "min_loss": min_loss,
            }
            state.update(
                {
                    "adv_model": adv_model.module.state_dict(),
                    "adv_optimizer": adv_optimizer.state_dict(),
                    "adv_scheduler": adv_scheduler.state_dict(),
                }
            )
            state_file = "state_{}.pt".format(epoch + 1)
            torch.save(state, state_file)
            del state
            tmp_link = "{}.pt".format(time.time())
            os.symlink(state_file, tmp_link)
            os.rename(tmp_link, ckpt_link)
    dist.destroy_process_group()


def train(
    epoch,
    loader,
    model,
    criterion,
    optimizer,
    scheduler,
    adv_model,
    adv_criterion,
    adv_optimizer,
    adv_scheduler,
    logger,
    device,
    args,
):
    model.train()
    adv_model.train()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.log_interval <= args.adv_wgan_gp_interval or args.adv_wgan_gp_interval < 1:
        adv_wgan_gp_log_interval = args.log_interval
    else:
        adv_wgan_gp_log_interval = (
            args.log_interval // args.adv_wgan_gp_interval * args.adv_wgan_gp_interval
        )
    # MSE, generator adversarial loss, and discriminator total/fake/real losses.
    epoch_loss = torch.zeros(5, dtype=torch.float64, device=device)
    fake = torch.zeros([1], dtype=torch.float32, device=device)
    real = torch.ones([1], dtype=torch.float32, device=device)
    adv_real = torch.full([1], 1, dtype=torch.float32, device=device)
    for i, data in enumerate(loader):
        batch = epoch * len(loader) + i + 1
        (input, target, style) = (data["input"], data["target"], data["style"])
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        style = style.to(device, non_blocking=True)
        output = model(input, style)
        if hasattr(model.module, "scale_factor") and model.module.scale_factor != 1:
            input = resample(input, model.module.scale_factor, narrow=False)
        (input, output, target) = narrow_cast(input, output, target)
        # After the initial warmup, MSE is logged but not optimized.
        loss = criterion(output, target)
        epoch_loss[0] += loss.detach()
        if epoch >= args.adv_start:
            lag_out = output[:, :3]
            lag_tgt = target[:, :3]
            eul_out = lag2eul(
                lag_out, a=np.float64(style[:, 0]), Om=np.float64(style[:, 1])
            )[0]
            eul_tgt = lag2eul(
                lag_tgt, a=np.float64(style[:, 0]), Om=np.float64(style[:, 1])
            )[0]
            output = torch.cat([eul_out, output], dim=1)
            target = torch.cat([eul_tgt, target], dim=1)
            output = torch.cat([input, output], dim=1)
            target = torch.cat([input, target], dim=1)
            set_requires_grad(adv_model, True)
            score_out = adv_model(output.detach(), style=style)
            adv_loss_fake = adv_criterion(score_out, fake.expand_as(score_out))
            epoch_loss[3] += adv_loss_fake.item()
            adv_optimizer.zero_grad()
            adv_loss_fake.backward()
            score_tgt = adv_model(target, style=style)
            adv_loss_real = adv_criterion(score_tgt, adv_real.expand_as(score_tgt))
            epoch_loss[4] += adv_loss_real.item()
            adv_loss_real.backward()
            adv_loss = adv_loss_fake + adv_loss_real
            epoch_loss[2] += adv_loss.item()
            if args.adv_wgan_gp_interval > 0 and batch % args.adv_wgan_gp_interval == 0:
                adv_loss_reg = wgan_grad_penalty(adv_model, output, target, style=style)
                adv_loss_reg_ = adv_loss_reg * args.adv_wgan_gp_interval
                adv_loss_reg_.backward()
                if batch % adv_wgan_gp_log_interval == 0 and rank == 0:
                    logger.add_scalar(
                        "loss/batch/train/adv/reg",
                        adv_loss_reg.item(),
                        global_step=batch,
                    )
            adv_optimizer.step()
            set_requires_grad(adv_model, False)
            score_out = adv_model(output, style=style)
            loss_adv = adv_criterion(score_out, real.expand_as(score_out))
            epoch_loss[1] += loss_adv.item()
            optimizer.zero_grad()
            loss_adv.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if batch % args.log_interval == 0:
            loss_to_log = loss.detach()
            dist.reduce(loss_to_log, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                loss_to_log /= world_size
                if logger is not None:
                    logger.add_scalar(
                        "loss/batch/train", loss_to_log.item(), global_step=batch
                    )
                    if epoch >= args.adv_start:
                        logger.add_scalar(
                            "loss/batch/train/adv/G", loss_adv.item(), global_step=batch
                        )
                        logger.add_scalars(
                            "loss/batch/train/adv/D",
                            {
                                "total": adv_loss.item(),
                                "fake": adv_loss_fake.item(),
                                "real": adv_loss_real.item(),
                            },
                            global_step=batch,
                        )
        if args.save_interval > 0 and batch % args.save_interval == 0 and (rank == 0):
            mid_state = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "min_loss": None,
            }
            mid_state.update(
                {
                    "adv_model": adv_model.module.state_dict(),
                    "adv_optimizer": adv_optimizer.state_dict(),
                    "adv_scheduler": adv_scheduler.state_dict(),
                }
            )
            mid_file = "state_{}_b{}.pt".format(epoch + 1, batch)
            torch.save(mid_state, mid_file)
            del mid_state
            tmp_link = "{}.pt".format(time.time())
            os.symlink(mid_file, tmp_link)
            os.rename(tmp_link, ckpt_link)
            print("mid-epoch checkpoint saved: {}".format(mid_file), flush=True)
    dist.barrier()
    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar("loss/epoch/train", epoch_loss[0], global_step=epoch + 1)
        if epoch >= args.adv_start:
            logger.add_scalar(
                "loss/epoch/train/adv/G", epoch_loss[1], global_step=epoch + 1
            )
            logger.add_scalars(
                "loss/epoch/train/adv/D",
                {"total": epoch_loss[2], "fake": epoch_loss[3], "real": epoch_loss[4]},
                global_step=epoch + 1,
            )
        if epoch >= args.adv_start:
            skip_chan = sum(args.in_chan)
            output = output[:, skip_chan:]
            target = target[:, skip_chan:]
        fig = plt_slices(
            input[-1, -6:],
            output[-1, -6:],
            target[-1, -6:],
            output[-1, -6:] - target[-1, -6:],
            title=["in", "out", "tgt", "out - tgt"],
            **args.misc_kwargs,
        )
        logger.add_figure("fig/train", fig, global_step=epoch + 1)
        fig.clf()
    return epoch_loss


def dist_init(rank, args, local_rank):
    dist_file = "dist_addr"
    print(
        f"Rank {rank}: Setting up distributed training with local_rank {local_rank}",
        flush=True,
    )
    if rank == 0:
        if os.path.exists(dist_file):
            os.remove(dist_file)
        addr = socket.gethostname()
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, 0))
            (_, port) = s.getsockname()
        args.dist_addr = f"tcp://{addr}:{port}"
        print(f"Rank {rank}: Master node using address {args.dist_addr}", flush=True)
        with open(dist_file, "w") as f:
            f.write(args.dist_addr)
            f.flush()
            os.fsync(f.fileno())
    else:
        for _ in range(120):
            if os.path.exists(dist_file) and os.path.getsize(dist_file) > 0:
                break
            time.sleep(0.5)
        with open(dist_file, "r") as f:
            args.dist_addr = f.read().strip()
        if not args.dist_addr:
            raise RuntimeError(f"Rank {rank}: dist_addr file empty after wait")
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_addr,
        world_size=args.world_size,
        rank=rank,
        device_id=torch.device("cuda", local_rank),
        timeout=timedelta(hours=2),
    )
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        os.remove(dist_file)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad
