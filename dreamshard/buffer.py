import numpy as np
import torch

class Buffer:
    def __init__(
        self,
        table_features_list,
        batch_size,
    ):
        self.table_features_list = table_features_list
        self.batch_size = batch_size

        self.overall_X = []
        self.overall_y = torch.tensor([])

        self.kernel_X = []
        self.forward_y = torch.tensor([])
        self.communication_y = torch.tensor([])
        self.backward_y = torch.tensor([])

        self.overall_size = 0
        self.kernel_size = 0

    def add_overall(self, X, y, task_id):
        processed_X = []
        for shard in X:
            if len(shard) > 0:
                processed_X.append(torch.stack([self.table_features_list[task_id][index] for index in shard]))
            else:
                processed_X.append(torch.zeros(1, self.table_features_list[task_id][0].shape[0], dtype=torch.float32))

        self.overall_X.append(processed_X)
        self.overall_y = torch.cat((self.overall_y, torch.tensor([y])))
        self.overall_size += 1

    def add_kernel(self, X, forward_y, communication_y, backward_y, task_id):
        if len(X) > 0:
            self.kernel_X.append(torch.stack([self.table_features_list[task_id][index] for index in X]))
        else:
            self.kernel_X.append(torch.zeros(1, self.table_features_list[task_id][0].shape[0], dtype=torch.float32))

        self.forward_y = torch.cat((self.forward_y, torch.tensor([forward_y])))
        self.communication_y = torch.cat((self.communication_y, torch.tensor([communication_y])))
        self.backward_y = torch.cat((self.backward_y, torch.tensor([backward_y])))
        self.kernel_size += 1

    def sample_overall(self):
        indices = np.random.randint(
            self.overall_size,
            size=self.batch_size,
        )
        X = [self.overall_X[index] for index in indices]
        y = self.overall_y[indices]
        return X, y

    def sample_kernel(self):
        indices = np.random.randint(
            self.kernel_size,
            size=self.batch_size,
        )
        X = [self.kernel_X[index] for index in indices]
        forward_y = self.forward_y[indices]
        communication_y = self.communication_y[indices]
        backward_y = self.backward_y[indices]
        return X, forward_y, communication_y, backward_y
