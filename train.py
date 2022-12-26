import json
import os
import pandas as pd
import numpy as np
import sys

from utils import util
from data_loader import dataloader

from tqdm import tqdm
from logging import getLogger
from recbole.utils import init_logger, get_trainer, init_seed, set_color
from recbole.quick_start import run_recbole, run_recboles
from recbole.config import Config
from recbole.data.transform import construct_transform
from recbole.data import create_dataset, data_preparation
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
)
import argparse


def main():
    parser = argparse.ArgumentParser()
    BASE = './input/base/'
    parser.add_argument(
        "--data_dir", default="/opt/ml/input/data/train/", type=str)
    parser.add_argument("--output_dir", default=BASE+"output/", type=str)
    parser.add_argument("--dataset", default="recbole", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--atomic_dir", default=BASE+"data/recbole/", type=str)
    parser.add_argument("--model", default='EASE', type=str)
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument("--config_files", default="./input/base/data/recbole.yaml",
                        type=str, help="config files")
    args = parser.parse_args()

    util.set_seed(args.seed)

    # load raw data
    df = pd.read_csv(args.data_dir + 'train_ratings.csv')
    df = df.drop(columns='time')

    # convert df to atomic files & save to path
    util.make_atomic(args, df)

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    if args.nproc == 1 and args.world_size <= 0:
        model = args.model
        dataset = args.dataset
        config_file_list = config_file_list
        config_dict = None
        saved = True
        # configurations initialization
        config = Config(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
        )
        init_seed(config["seed"], config["reproducibility"])
        # logger initialization
        init_logger(config)
        logger = getLogger()
        logger.info(sys.argv)
        logger.info(config)

        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # model loading and initialization
        init_seed(config["seed"] + config["local_rank"],
                  config["reproducibility"])
        model = get_model(config["model"])(
            config, train_data._dataset).to(config["device"])
        logger.info(model)

        transform = construct_transform(config)
        flops = get_flops(model, dataset, config["device"], logger, transform)
        logger.info(set_color("FLOPs", "blue") + f": {flops}")

        # trainer loading and initialization
        trainer = get_trainer(config["MODEL_TYPE"],
                              config["model"])(config, model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config["show_progress"]
        )

        # model evaluation
        test_result = trainer.evaluate(
            test_data, load_best_model=saved, show_progress=config["show_progress"]
        )

        logger.info(set_color("best valid ", "yellow") +
                    f": {best_valid_result}")
        logger.info(set_color("test result", "yellow") + f": {test_result}")

        return {
            "best_valid_score": best_valid_score,
            "valid_score_bigger": config["valid_metric_bigger"],
            "best_valid_result": best_valid_result,
            "test_result": test_result,
        }
        # run_recbole(
        #     model=args.model, dataset=args.dataset, config_file_list=config_file_list
        # )
    else:
        if args.world_size == -1:
            args.world_size = args.nproc
        import torch.multiprocessing as mp

        mp.spawn(
            run_recboles,
            args=(
                args.model,
                args.dataset,
                config_file_list,
                args.ip,
                args.port,
                args.world_size,
                args.nproc,
                args.group_offset,
            ),
            nprocs=args.nproc,
        )


if __name__ == "__main__":
    main()
