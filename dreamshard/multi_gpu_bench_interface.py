import os
import subprocess
import numpy as np
import traceback

import dreamshard

root_path = dreamshard.__path__[0]
bench_path = os.path.join(root_path, "multi_gpu_bench.py")

class Evaluator:
    def __init__(
        self,
        data_dir,
        task_path,
        gpu_devices,
    ):
        self.ndevices = len(gpu_devices.split(","))

        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        
        # Construct command
        command = "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={} python -m torch.distributed.run --nproc_per_node={} {} --data-dir {} --task-path {} --ndevices {}".format(
            gpu_devices,
            self.ndevices,
            bench_path,
            data_dir,
            task_path,
            self.ndevices,
        )

        # Run command
        self.process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            encoding='utf8',
        )

        # Wait until it is ready
        while True:
            line = self.process.stdout.readline().strip()
            if line == "1":
                break
        print("Evaluator initiated!")


    def evaluate(self, task_id, sharding):
        sharding = ",".join(map(str, sharding))
        self.process.stdin.write(str(task_id) + " " + sharding+"\n")
        self.process.stdin.flush()

        while True:
            line = self.process.stdout.readline().strip()
            if line == "2":
                break

        # Read results
        latencies = []
        for device in range(self.ndevices):
            result_path = os.path.join("tmp", str(device))
            with open(result_path, "r") as f:
                latency = list(map(float, f.readlines()[0].split(",")))
            latencies.append(latency)
        latencies = np.array(latencies)
        max_latency = np.max(np.sum(latencies, axis=1))

        return max_latency, latencies

    def terminate(self):
        #self.process.terminate()
        self.process.stdin.write("-1\n")
        self.process.stdin.flush()
        self.process.stdin.close()
        self.process.wait()
        print("Evaluator sub-process terminated!")
