"""Internal arguments for the state56 training loop."""

import argparse
import json


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="CosmoSR state56 training")
    commands = parser.add_subparsers(dest="mode", required=True)
    train = commands.add_parser("train")
    train.add_argument("--train-in-patterns", required=True, type=str_list)
    train.add_argument("--train-tgt-patterns", required=True, type=str_list)
    train.add_argument("--train-style-pattern", required=True)
    train.add_argument("--in-norms", required=True, type=str_list)
    train.add_argument("--tgt-norms", required=True, type=str_list)
    train.add_argument("--model", default="styled_srsgan.G")
    train.add_argument("--adv-model", default="styled_srsgan.D")
    train.add_argument("--crop", required=True, type=int)
    train.add_argument("--crop-step", required=True, type=int)
    train.add_argument("--pad", dest="in_pad", required=True, type=int)
    train.add_argument("--tgt-pad", required=True, type=int)
    train.add_argument("--scale-factor", default=2, type=int, choices=[2])
    train.add_argument("--augment", action="store_true")
    train.add_argument("--aug-shift", required=True, type=int)
    train.add_argument("--batch-size", required=True, type=int, choices=[1])
    train.add_argument("--loader-workers", required=True, type=int)
    train.add_argument("--lr", required=True, type=float)
    train.add_argument("--adv-lr", required=True, type=float)
    train.add_argument("--optimizer-args", required=True, type=json.loads)
    train.add_argument("--adv-start", required=True, type=int)
    train.add_argument("--adv-wgan-gp-interval", required=True, type=int)
    train.add_argument("--div-data", action="store_true")
    train.add_argument("--div-shuffle-dist", required=True, type=float)
    train.add_argument("--epochs", required=True, type=int)
    train.add_argument("--seed", default=42, type=int)
    train.add_argument("--log-interval", default=100, type=int)
    train.add_argument("--save-interval", default=0, type=int)
    train.add_argument("--load-state", default="")
    train.set_defaults(
        criterion="MSELoss",
        adv_criterion="WDistLoss",
        optimizer="Adam",
        scheduler_args={},
        load_state_strict=True,
        dist_backend="nccl",
        crop_start=None,
        crop_stop=None,
        callback_at=None,
        misc_kwargs={},
    )
    args = parser.parse_args(argv)
    args.adv_optimizer_args = args.optimizer_args
    return args


def str_list(value):
    return value.split(",")
