import numpy as np
import torch
import gym

from dreamshard.utils import plan2allocation

class Env(gym.Env):
    def __init__(
        self,
        table_features,
        table_sizes,
        model,
        ndevices,
        max_memory,
    ):
        self.table_features = table_features
        self.table_sizes = table_sizes
        self.model = model
        self.ndevices = ndevices
        self.max_memory = max_memory
        self.num_tables = len(self.table_features)
        self.num_features = self.table_features[0].shape[0]
        self.env2real = [i for i in range(self.num_tables)] # Map environmental index to real index


        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,))
        self.action_space = gym.spaces.Discrete(self.ndevices)

    def reset(self):
        # Sort based on single table forward and backward costs
        def predict_single_table_cost(inputs):
            with torch.no_grad():
                forward_cost, backward_cost, communication_cost = self.model.kernel_forward([inputs[0].unsqueeze(0)])
            return forward_cost.item() + backward_cost.item() + communication_cost.item()

        (
            self.table_features,
            self.table_sizes,
            self.env2real,
        ) = tuple(map(list, zip(*sorted(
            list(zip(
                self.table_features,
                self.table_sizes,
                self.env2real,
            )),
            key=predict_single_table_cost,
            reverse=True,
        ))))

        self.plan = [[] for _ in range(self.ndevices)]
        self.sizes = [0 for _ in range(self.ndevices)]
        self.cur_step = 0
        self._get_obs()

        info = {"legal_actions": self._get_legal_actions()}

        return self.obs, info

    def step(self, action):
        self.plan[action].append(self.cur_step)
        self.sizes[action] += self.table_sizes[self.cur_step]
        self.cur_step += 1
        done = True if self.cur_step >= self.num_tables else False
        self._get_obs()

        info = {"legal_actions": self._get_legal_actions()}
        if done:
            info["sharding"] = plan2allocation([[self.env2real[index] for index in shard] for shard in self.plan])

        return self.obs, self._get_reward(), done, info

    def _get_obs(self):
        self.obs = []
        for shard in self.plan:
            if len(shard) > 0:
                self.obs.append(torch.stack([self.table_features[index] for index in shard]))
            else:
                self.obs.append(torch.zeros(1, self.table_features[0].shape[0], dtype=torch.float32))

    def _get_reward(self):
        if self.cur_step >= self.num_tables:
            overall_cost = self.model.overall_forward([self.obs])[0].item()
            return -overall_cost
        else:
            return 0

    def _get_legal_actions(self):
        legal_actions = []
        if self.cur_step < self.num_tables:
            for device in range(self.ndevices):
                if self.sizes[device] + self.table_sizes[self.cur_step] <= self.max_memory:
                    legal_actions.append(device)
        return legal_actions

