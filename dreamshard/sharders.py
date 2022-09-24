from dataclasses import dataclass
import numpy as np

from dreamshard.utils import allocation2plan, plan2allocation

_sharders = {}

@dataclass
class TableInfo:
    index: int
    cost: float
    size: float

    def __lt__(self, o: "TableInfo") -> bool:
        return (self.cost, self.size, self.index) < (o.cost, o.size, o.index)

def table_size(hash_size, embedding_dim, fp16: bool = False) -> float:
    gb_size = hash_size * embedding_dim / (1024 * 1024 * 1024)
    if fp16:
        gb_size = 2 * gb_size
    else:
        gb_size = 4 * gb_size
    return gb_size

def register_sharder(sharder_name):
    def decorate(func):
        _sharders[sharder_name] = func
        return func
    return decorate

# get device indices for tables
# e.g 8 tables, No. [1,3,5,6] on device 0, No. [2,4,7,8] on device 1, then
# return [0, 1, 0, 1, 0, 0, 1, 1]
def shard(env, alg="naive"):
    if alg not in _sharders:
        import sys
        sys.exit("ERROR: sharder not found")
    return _sharders[alg](env)

@register_sharder("naive")
def naive_shard(env):
    return [(x % env.ndevices) for x in range(env.num_tables)]

@register_sharder("dim_greedy")
def dim_greedy_shard(env, return_plan=False):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(env.table_configs):

        index = config["index"]
        dim = config["dim"]
        row = config["row"]
        size = table_size(row, dim, fp16=True)
        idx_weight_pairs.append(TableInfo(index=index, cost=dim, size=size))

    # Greedy algorithm
    num_bins = env.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [env.max_memory] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions if return_plan else plan2allocation(partitions)

@register_sharder("size_greedy")
def size_greedy_shard(env, return_plan=False):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(env.table_configs):

        index = config["index"]
        dim = config["dim"]
        row = config["row"]
        size = table_size(row, dim, fp16=True)
        idx_weight_pairs.append(TableInfo(index=index, cost=size, size=size))

    # Greedy algorithm
    num_bins = env.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [env.max_memory] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions if return_plan else plan2allocation(partitions)

@register_sharder("lookup_greedy")
def lookup_greedy_shard(env, return_plan=False):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(env.table_configs):

        index = config["index"]
        dim = config["dim"]
        row = config["row"]
        pooling_factor = config["pooling_factor"]
        size = table_size(row, dim, fp16=True)
        idx_weight_pairs.append(TableInfo(index=index, cost=dim*pooling_factor, size=size))

    # Greedy algorithm
    num_bins = env.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [env.max_memory] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions if return_plan else plan2allocation(partitions)

@register_sharder("size_lookup_greedy")
def size_lookup_greedy_shard(env, return_plan=False):
    # Get the embedding dims
    idx_weight_pairs = []
    for i, config in enumerate(env.table_configs):

        index = config["index"]
        dim = config["dim"]
        row = config["row"]
        pooling_factor = config["pooling_factor"]
        size = table_size(row, dim, fp16=True)
        idx_weight_pairs.append(TableInfo(index=index, cost=dim*pooling_factor*np.log10(row), size=size))

    # Greedy algorithm
    num_bins = env.ndevices
    sorted_idx_weight_pairs = sorted(idx_weight_pairs)
    partitions = [[] for p in range(num_bins)]
    partition_sums = [0.0] * num_bins
    partition_size_sums = [0.0] * num_bins

    mem_cap = [env.max_memory] * num_bins

    while sorted_idx_weight_pairs:
        table_info = sorted_idx_weight_pairs.pop()
        min_sum = np.inf
        min_size_taken = np.inf
        min_r = -1
        for r in range(num_bins):
            if partition_size_sums[r] + table_info.size <= mem_cap[r]:
                if partition_sums[r] < min_sum or (
                    partition_sums[r] == min_sum
                    and partition_size_sums[r] < min_size_taken
                ):
                    min_sum = partition_sums[r]
                    min_r = r
                    min_size_taken = partition_size_sums[r]

        partitions[min_r].append(table_info)
        partition_sums[min_r] += table_info.cost
        partition_size_sums[min_r] += table_info.size

    partitions = [[table_info.index for table_info in partition] for partition in partitions]

    return partitions if return_plan else plan2allocation(partitions)

@register_sharder("random")
def random_shard(env):
    table_device_indices = []
    for _ in range(env.num_tables):
        table_device_indices.append(np.random.randint(env.ndevices))
    return table_device_indices

@register_sharder("dreamshard")
def dreamshard(env, return_plan=False):
    import torch
    from torch.nn import functional as F
    done = False
    obs, info = env.reset()
    while not done:
        obs = [obs[device] for device in range(env.ndevices) if device in info["legal_actions"]]
        with torch.no_grad():
            policy_logits = env.model.forward([obs])
            action_id = torch.argmax(policy_logits, dim=1)
            action_id = action_id.item()
            action = info["legal_actions"][action_id]
        obs, reward, done, info = env.step(action)
    allocation = info["sharding"]

    if return_plan == True:
        return plan2allocation(allocation, env.ndevices)
    else:
        return allocation
