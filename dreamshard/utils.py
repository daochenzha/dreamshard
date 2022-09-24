import os
import json
import copy

import numpy as np
import torch

def table_size(hash_size, dim, fp16=False):
    gb_size = hash_size * dim / (1024 * 1024 * 1024)
    if fp16:
        gb_size = 2 * gb_size
    else:
        gb_size = 4 * gb_size
    return gb_size

def plan2allocation_tables(plan, num_tables):
    table_device_indices = [-1] * num_tables
    for bin_id, partition in enumerate(plan):
        for index in partition:
            table_device_indices[index] = bin_id
    return table_device_indices

def plan2allocation(plan):
    num_tables = sum([len(shard) for shard in plan])
    table_device_indices = [-1] * num_tables
    for bin_id, partition in enumerate(plan):
        for index in partition:
            table_device_indices[index] = bin_id
    return table_device_indices

def allocation2plan(allocation, ndevices):
    plan = [[] for _ in range(ndevices)]
    for i, d in enumerate(allocation):
        if d != -1:
            plan[d].append(i)
    return plan

def get_table_ids_list(task_path):
    with open(task_path, "r") as f:
        lines = f.readlines()
    table_ids_list = [list(map(int, line.strip().split(","))) for line in lines]

    return table_ids_list
    
def load_table_configs_features_sizes(data_dir, table_ids):
    # Load table configs
    table_config_path = os.path.join(data_dir, "table_configs.json")
    with open(table_config_path) as f:
        table_configs = json.load(f)["tables"]
    table_configs = [copy.deepcopy(table_configs[table_id]) for table_id in table_ids]
    for table_index in range(len(table_configs)):
        table_configs[table_index]["index"] = table_index
    for i in range(len(table_configs)):
        table_configs[i]["size"] = table_configs[i]["dim"] * table_configs[i]["row"]
    table_gb_sizes = [table_size(
        table_config["row"],
        table_config["dim"],
        fp16=True,
    ) for table_config in table_configs]

    # Normalize the features
    norm_table_configs = copy.deepcopy(table_configs)
    features = ["row", "size", "pooling_factor"]
    for feature in features:
        vals = [table_config[feature] for table_config in norm_table_configs]
        mean = np.mean(vals)
        std = np.std(vals)
        for i in range(len(norm_table_configs)):
            norm_table_configs[i][feature] = (norm_table_configs[i][feature] - mean) / std

    # Pre-compute all the table features
    feature_names = [
        "dim",
        "row",
        "pooling_factor",
        "size",
        "bin_0",
        "bin_1",
        "bin_2",
        "bin_3",
        "bin_4",
        "bin_5",
        "bin_6",
        "bin_7",
        "bin_8",
        "bin_9",
        "bin_10",
        "bin_11",
        "bin_12",
        "bin_13",
        "bin_14",
        "bin_15",
        "bin_16",
    ]
    table_features = [torch.tensor([table_config[key] for key in feature_names], dtype=torch.float32) for table_config in norm_table_configs]

    return table_configs, table_features, table_gb_sizes
    

