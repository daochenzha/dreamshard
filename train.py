import os
import argparse

import numpy as np
import torch

from dreamshard.training import train

def main():
    parser = argparse.ArgumentParser("RLShard")
    parser.add_argument('--data-dir', type=str, default="data/dlrm_datasets")
    parser.add_argument('--task-path', type=str, default="data/dlrm_tasks_50/train.txt")
    parser.add_argument('--gpu-devices', type=str, default="3,4,5,6")
    parser.add_argument('--num-iterations', type=int, default=10)
    parser.add_argument('--bench-steps', type=int, default=10)
    parser.add_argument('--rl-num-batches', type=int, default=10)
    parser.add_argument('--rl-batch-size', type=int, default=10)
    parser.add_argument('--bench-training-steps', type=int, default=300)
    parser.add_argument('--entropy-weight', type=int, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-memory', type=int, default=5, help="Max memory for each shard in GB")
    parser.add_argument('--out-dir', type=str, default="models/dreamshard")
    args = parser.parse_args()
    args.ndevices = len(args.gpu_devices.split(","))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    train(args)

if __name__ == '__main__':
    main()
