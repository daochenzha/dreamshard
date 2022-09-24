import os
import json
import argparse
import time
import numpy as np
from typing import Dict, Any, Callable
from collections import OrderedDict
import gc
import copy

import torch
from fbgemm_gpu import split_table_batched_embeddings_ops
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType

from dreamshard import extend_distributed as ext_dist
from dreamshard.utils import allocation2plan, get_table_ids_list

def main():
    parser = argparse.ArgumentParser("Multi GPU benchmark for embedding tables")
    parser.add_argument('--data-dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--task-path', type=str, default="data/dlrm_tasks_50/train.txt")
    parser.add_argument('--ndevices', type=int, default=4)

    args = parser.parse_args()

    # Init distributed training
    ext_dist.init_distributed(use_gpu=True)
    device = "cuda:{}".format(ext_dist.my_local_rank)
    torch.set_num_threads(1)
    torch.cuda.set_device(device)

    # Read data
    table_config_path = os.path.join(args.data_dir, "table_configs.json")
    data_path = os.path.join(args.data_dir, "data.pt")
    with open(table_config_path) as f:
        table_configs = json.load(f)["tables"]
    data = torch.load(data_path)

    # Create an env for each task
    table_ids_list = get_table_ids_list(args.task_path)
    envs = []
    for task_id, table_ids in enumerate(table_ids_list):
        task_offsets = [data["lS_offsets"][table_id] for table_id in table_ids]
        task_indices = [data["lS_indices"][table_id] for table_id in table_ids]
        task_data = {
            "lS_offsets": task_offsets,
            "lS_indices": task_indices,
        }
        task_table_configs = [copy.deepcopy(table_configs[table_id]) for table_id in table_ids]
        for table_index in range(len(task_table_configs)):
            task_table_configs[table_index]["index"] = table_index

        # Create environment
        env = Env(
            table_configs=task_table_configs,
            data=task_data,
            ndevices=args.ndevices,
            device=device,
        )

        envs.append(env)

    # Synchronize workers
    a2a_req = ext_dist.alltoall([torch.zeros(args.ndevices, 4, 1).to(device)], [4 for _ in range(args.ndevices)], True)
    tmp = a2a_req.wait()
    print("1") # Show it is ready

    while True:
        if ext_dist.my_local_rank == 0:
            inputs = input()
            if inputs == "-1":
                task_id = -1
            else:
                task_id, sharding = inputs.split()
                task_id = int(task_id)
                sharding = list(map(int, sharding.split(",")))
            task_id_tensor = torch.tensor([task_id], device=device, dtype=torch.float32).repeat(args.ndevices, 1).unsqueeze(-1)
        else:
            task_id_tensor = torch.zeros((args.ndevices, 1, 1), device=device)
        a2a_req = ext_dist.alltoall([task_id_tensor], [1 for _ in range(args.ndevices)], True)
        task_id = a2a_req.wait()[0].long().flatten().tolist()[0]
        if task_id == -1:
            break

        if ext_dist.my_local_rank == 0:
            sharding_tensor = torch.tensor(sharding, device=device, dtype=torch.float32).repeat(args.ndevices, 1).unsqueeze(-1)
        else:
            sharding_tensor = torch.zeros((args.ndevices, envs[task_id].num_tables, 1), device=device)
        a2a_req = ext_dist.alltoall([sharding_tensor], [sharding_tensor.shape[1] for _ in range(args.ndevices)], True)
        sharding = a2a_req.wait()[0].long().flatten().tolist()

        # Get number of tables in each rank and local indices
        plan = allocation2plan(sharding, args.ndevices)
        num_elements_per_rank = [sum([envs[task_id].table_configs[index]["dim"] for index in indices]) for indices in plan]
        local_indices = plan[ext_dist.my_local_rank]
        #ext_dist.print_all("Indices:", local_indices, ext_dist.my_local_rank)

        # Benchmark
        latency = envs[task_id].step(local_indices, num_elements_per_rank)
        # Output results
        #latentcy = (",".join(list(map(str, latency))))
        #ext_dist.print_all("Latency:", latency, ext_dist.my_local_rank)
        result_path = os.path.join("tmp", str(ext_dist.my_local_rank))
        with open(result_path, "w") as f:
            f.write(",".join(list(map(str, latency))))
            f.flush()

        # Synchronize workers
        a2a_req = ext_dist.alltoall([torch.zeros(args.ndevices, 4, 1).to(device)], [4 for _ in range(args.ndevices)], True)
        tmp = a2a_req.wait()

        time.sleep(0.5) # Wait until file writing finished
        print("2") # Succuss signal

class Env:
    def __init__(
        self,
        table_configs,
        data,
        ndevices,
        warmup_iter=5,
        num_iter=10,
        device="cuda:0",
        verbose=False,
    ):

        self.table_configs = table_configs
        self.num_tables = len(self.table_configs)

        # Load indices and offsets
        self.offsets = data["lS_offsets"]
        self.indices = data["lS_indices"]

        self.ndevices = ndevices
        self.warmup_iter = warmup_iter
        self.num_iter = num_iter
        self.device = device
        self.verbose = verbose
        self.batch_size = self.offsets[0].shape[0] - 1

    def step(self, table_indices, num_elements_per_rank):
        gc.collect()
        torch.cuda.empty_cache()

        if len(table_indices) > 0:
            # Build the op
            shard_table_configs = [self.table_configs[i] for i in table_indices]

            op = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
                [
                    (
                        table_config["row"],
                        table_config["dim"],
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                    )
                    for table_config in shard_table_configs
                ],
                optimizer=OptimType.EXACT_SGD,
                cache_algorithm=split_table_batched_embeddings_ops.CacheAlgorithm.LFU,
                cache_reserved_memory=8.0,
                eps=0.01,
                device=self.device,
                weights_precision=SparseType.FP16,
            )

            # Get data
            shard_offsets = [self.offsets[i] for i in table_indices]
            shard_indices = [self.indices[i] for i in table_indices]
            args, kwargs, shard_grads_tensor = get_data(
                self.ndevices,
                self.batch_size,
                shard_offsets,
                shard_indices,
                sum([self.table_configs[index]["dim"] for index in table_indices]),
                self.device,
            )
        else:
            op, args, kwargs = None, [], {}
            shard_grads_tensor = torch.randn(
                (
                    self.batch_size,
                    0,
                    1,
                ),
            )

        grads_tensor = torch.randn(
            (
                self.batch_size // self.ndevices,
                #sum([table_config["dim"] for table_config in self.table_configs])
                sum(num_elements_per_rank),
            ),
        )

        # Warmup
        warmup_time_records = warmup(
            op,
            args,
            kwargs,
            grads_tensor,
            shard_grads_tensor,
            self.device,
            self.ndevices,
            num_elements_per_rank,
            num_iter=self.warmup_iter,
        )
        if self.verbose:
            print("Warmup:", warmup_time_records, table_indices)

        # Benchmark
        time_records = measure_latency(
            op,
            args,
            kwargs,
            grads_tensor,
            shard_grads_tensor,
            self.device,
            self.ndevices,
            num_elements_per_rank,
            num_iter=self.num_iter,
        )

        if self.verbose:
            print("Benchmark:", time_records, table_indices)

        # Remove the largest two and smallest two values
        time_records = np.array(time_records)
        time_records = np.sort(time_records, axis=0)
        return np.median(time_records, axis=0)

# Timer in seconds
class Timer:
    def __init__(self, device: str):
        self.device: str = device
        self.start_time: float = 0
        self.end_time: float = 0
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if self.device == "cpu":
            self.start_time = time.perf_counter()
        else:
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.start_time = 0
        return self

    def __exit__(self, type, value, traceback):
        if self.device == "cpu":
            self.end_time = time.perf_counter()
        else:
            self.end_event.record()
            torch.cuda.synchronize()
            self.end_time = self.start_event.elapsed_time(self.end_event) * 1.0e-3

    # returns time in seconds
    def elapsed_time(self):
        return self.end_time - self.start_time

def benchmark_op(
    op: Callable,
    args: Any,
    kwargs: Any,
    grads_tensor: Any,
    shard_grads_tensor: Any,
    device: str,
    ndevices: int,
    num_elements_per_rank: list,
    num_iter: int
):
    batch_size = shard_grads_tensor.shape[0]
    time_records = []
    for _ in range(num_iter):
        tmp_grads_tensor = grads_tensor.to(device)
        torch.cuda.synchronize()

        # Synchronize workers
        a2a_req = ext_dist.alltoall([torch.zeros(ndevices, 4, 1).to(device)], [4 for _ in range(ndevices)], True)
        tmp = a2a_req.wait()
        
 
        iter_time = []
        with Timer(device) as timer:
            if op is None:
                y = torch.randn((batch_size, 0, 1), device=device, requires_grad=True)
            else:
                y = op(*args, **kwargs)
                y = y.view(batch_size, -1, 1)
        iter_time.append(timer.elapsed_time() * 1000)
        with Timer(device) as timer:
            a2a_req = ext_dist.alltoall([y], num_elements_per_rank, True)
            y = a2a_req.wait()
        iter_time.append(timer.elapsed_time() * 1000)

        with Timer(device) as timer:
            y = torch.cat(y, dim=1)
            y.backward(tmp_grads_tensor)
        iter_time.append(timer.elapsed_time() * 1000)
        del y
        del tmp_grads_tensor

        # Benchmark kernel alone
        tmp_shard_grads_tensor = shard_grads_tensor.to(device)
        torch.cuda.synchronize()
        if op is None:
            y = torch.randn((batch_size, 0, 1), device=device, requires_grad=True)
        else:
            y = op(*args, **kwargs)
            y = y.view(batch_size, -1, 1)

        with Timer(device) as timer:
            y = y.backward(tmp_shard_grads_tensor)
        iter_time[-1] -= timer.elapsed_time() * 1000
        iter_time.append(timer.elapsed_time() * 1000)        
        time_records.append(iter_time)
        del y
        del tmp
        del tmp_shard_grads_tensor

    return time_records

def warmup(
    op: Callable,
    args: Any,
    kwargs: Any,
    grads_tensor: Any,
    shard_grads_tensor: Any,
    device: str,
    ndevices: int,
    num_elements_per_rank: list,
    num_iter: int,
):
    # warm up
    time_records = benchmark_op(op, args, kwargs, grads_tensor, shard_grads_tensor, device, ndevices, num_elements_per_rank, num_iter)
    return time_records

def measure_latency(
    op: Callable,
    args: Any,
    kwargs: Any,
    grads_tensor: Any,
    shard_grads_tensor: Any,
    device: str,
    ndevices: int,
    num_elements_per_rank: list,
    num_iter: int,
):
    torch.cuda.nvtx.range_push("op_bench")
    time_records = benchmark_op(op, args, kwargs, grads_tensor, shard_grads_tensor, device, ndevices, num_elements_per_rank, num_iter)

    return time_records

def get_data(ndevices, batch_size, offsets, indices, shard_dim, device):
    args_indices = torch.cat([x.view(-1) for x in indices], dim=0).int()
    E_offsets = [0] + np.cumsum([x.view(-1).shape[0] for x in indices]).tolist()
    args_offsets = torch.cat([torch.tensor([0])] + [x[1:] + y for x, y in zip(offsets, E_offsets[:-1])], dim=0).int()

    shard_grads_tensor = torch.randn(
        (
            batch_size,
            shard_dim,
            1,
        ),
    )

    return (
        [
            args_indices.to(device),
            args_offsets.to(device),
        ],
        {},
        shard_grads_tensor,
    )

if __name__ == "__main__":
    main()
