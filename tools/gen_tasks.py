import argparse
import os
import json

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser("Generate DLRM tasks")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--T-range', type=str, default="80,81")
    parser.add_argument('--data-dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--out-dir', type=str, default="data/dlrm_tasks")
    parser.add_argument('--num-train', type=int, default=50)
    parser.add_argument('--num-test', type=int, default=50)

    args = parser.parse_args()
    np.random.seed(args.seed)
    if "," in args.T_range:
        args.T_range = list(map(int, args.T_range.split(",")))
    else:
        args.T_range = [int(args.T_range), int(args.T_range)+1]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.data_dir, "table_configs.json"), "r") as f:
        table_configs = json.load(f)["tables"]
    T = len(table_configs)

    pool_indices = [i for i in range(T)]
    np.random.shuffle(pool_indices)
    train_pool_indices = pool_indices[:len(pool_indices)//2]
    test_pool_indices = pool_indices[len(pool_indices)//2:]

    gen_tasks(
        table_configs,
        "train",
        train_pool_indices,
        args.T_range,
        args.num_train,
        args.out_dir,
    )

    gen_tasks(
        table_configs,
        "test",
        test_pool_indices,
        args.T_range,
        args.num_test,
        args.out_dir,
    )

def gen_tasks(
    table_configs,
    split_name,
    pool_indices,
    T_range,
    num_tasks,
    out_dir,
):
    out_path = os.path.join(out_dir, split_name+".txt")
    # Generate tasks
    with open(out_path, "w") as f:
        for task_id in range(num_tasks):
            task_T = np.random.randint(T_range[0], T_range[1])
            table_ids = np.random.choice(pool_indices, size=task_T, replace=False)
            table_ids = ",".join(list(map(str, table_ids)))

            f.write(table_ids+"\n")

if __name__ == '__main__':
    main()
