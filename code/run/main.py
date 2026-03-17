import argparse
import os

from dsh.utils.pipelining import (
    train_model,
    calculate_metrics,
    interleaved_pipeline,
    sequential_pipeline,
    infer_model,
    parallelized_pipeline,
)
from dsh.utils.selector import get_config, add_arguments_to_parser, parse_args, init_logger, ConfigMode, get_model
from dsh.utils.setup import interactive_config_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "train",
            "infer",
            "metrics",
            "all",
            "sequentialpipe",
            "parallelpipe",
            "create-config",
        ],
        help="select the operation to perform",
    )
    add_arguments_to_parser(parser)
    args = parse_args(parser, os.path.dirname(__file__))
    init_logger(args)

    # load config, model and dataset
    if args.mode == "train":
        cfg = get_config(args, ConfigMode.TRAIN)
        model = get_model(cfg.model, cfg.env)
        train_model(cfg, model)
    elif args.mode == "infer":
        cfg = get_config(args, ConfigMode.INFERENCE)
        infer_model(cfg)
    elif args.mode == "metrics":
        cfg = get_config(args, ConfigMode.METRICS)
        calculate_metrics(cfg)
    elif args.mode == "all":
        # train, infer, metrics interleaved on an epoch basis in the same process
        interleaved_pipeline(args)
    elif args.mode == "sequentialpipe":
        # train, infer, metrics sequentially
        sequential_pipeline(args)
    elif args.mode == "parallelpipe":
        # train, infer, metrics interleaved on an epoch basis using subprocesses
        parallelized_pipeline(args)
    elif args.mode == "create-config":
        # interactive configuration setup
        interactive_config_setup(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    print("DONE")


if __name__ == "__main__":
    main()
