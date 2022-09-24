import os
import argparse
import json
import torch
import numpy as np
import copy
import time
import traceback

from dreamshard.env import Env
from dreamshard.models import Model
from dreamshard import sharders
from dreamshard.utils import (
    table_size,
    allocation2plan,
    load_table_configs_features_sizes,
    get_table_ids_list,
)
from dreamshard.multi_gpu_bench_interface import Evaluator

def main():
    parser = argparse.ArgumentParser("Benchmark sharding")
    parser.add_argument('--data-dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--task-path', type=str, default="data/dlrm_tasks_50/test.txt")
    parser.add_argument('--alg', type=str, default="random")
    parser.add_argument('--max-memory', type=int, default=5, help="Max memory for each shard in GB")
    parser.add_argument('--gpu-devices', type=str, default="0,1,2,3")

    args = parser.parse_args()
    args.ndevices = len(args.gpu_devices.split(","))

    model = Model(table_feature_dim=21)
    if args.alg[-3:] == ".pt":
        model.load_state_dict(torch.load(args.alg))
        args.alg = "dreamshard"
    
    table_ids_list = get_table_ids_list(args.task_path)
    try:
        evaluator = Evaluator(
            args.data_dir,
            args.task_path,
            args.gpu_devices,
        )
        latencies = [] 
        for task_id, table_ids in enumerate(table_ids_list):
            print("Task", str(task_id+1)+"/"+str(len(table_ids_list)))

            table_configs, table_features, table_sizes = load_table_configs_features_sizes(args.data_dir, table_ids)
            
            env = Env(
                table_features,
                table_sizes,
                model,
                args.ndevices,
                args.max_memory,
            )
            env.table_configs = table_configs

            sharding = sharders.shard(env, args.alg)
            print("Sharding:", sharding)
            plan = allocation2plan(sharding, env.ndevices)
            # Dim sums
            dims = [config["dim"] for config in env.table_configs]
            dim_sums = [sum([dims[i] for i in shard]) for shard in plan]
            print("Dims:", dim_sums)

            # Check size
            sizes = [table_size(config["row"], config["dim"], fp16=True) for config in env.table_configs]
            size_sums = [sum([sizes[i] for i in shard]) for shard in plan]
            print("Sizes:", size_sums)
            max_size_sum = max(size_sums)
            if max_size_sum > env.max_memory: 
                print("Out of memory")
                continue

            max_latency, latency = evaluator.evaluate(task_id, sharding)
            latencies.append(max_latency)
            print("Latency:", max_latency)
        print("Average:", np.mean(latencies))
    except:
        traceback.print_exc()
    finally:
        evaluator.terminate()


if __name__ == '__main__':
    main()

