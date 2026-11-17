"""Internal entrypoint for the retained training implementation."""


def main():
    from .args import get_args
    from .trainer import node_worker

    args = get_args()
    if args.mode != "train":
        raise ValueError("This release contains the training entrypoint only")
    node_worker(args)


if __name__ == "__main__":
    main()
