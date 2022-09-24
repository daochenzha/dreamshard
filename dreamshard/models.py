import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, table_feature_dim):
        super().__init__()
        table_latent_dim = 32
        # Table feature extraction
        self.table_fc_0 = nn.Linear(table_feature_dim, 128)
        self.table_fc_1 = nn.Linear(128, table_latent_dim)

        # Table feature extraction RL
        self.rl_table_fc_0 = nn.Linear(table_feature_dim, 128)
        self.rl_table_fc_1 = nn.Linear(128, table_latent_dim)

        # Forward head
        self.forward_fc_0 = nn.Linear(table_latent_dim, 64)
        self.forward_fc_1 = nn.Linear(64, 1)

        # Communication head
        self.communication_fc_0 = nn.Linear(table_latent_dim, 64)
        self.communication_fc_1 = nn.Linear(64, 1)

        # Backward head
        self.backward_fc_0 = nn.Linear(table_latent_dim, 64)
        self.backward_fc_1 = nn.Linear(64, 1)

        # Overall head
        self.overall_fc_0 = nn.Linear(table_latent_dim, 64)
        self.overall_fc_1 = nn.Linear(64, 1)

        # Cost features extraction
        self.cost_fc_0 = nn.Linear(3, 64)
        self.cost_fc_1 = nn.Linear(64, 32)

        # 32 for table raw features
        # 32 for cost features
        self.policy_net = nn.Linear(32*2, 1)

        # Value net
        # 32 for table raw features
        # 32 for cost features
        # We consider 4 devices
        self.value_net = nn.Linear(32*2*4, 1)

    def overall_forward(self, X):
        """ Overall cost
        """
        # X: a nested list: B x number of devices x number of tables in a shard x table_feature_dim
        X = self._multi_latent(X) # B x table_latent_dim
        overall_cost = F.relu(self.overall_fc_0(X))
        overall_cost = self.overall_fc_1(overall_cost)
        overall_cost = overall_cost.flatten()

        return overall_cost

    def kernel_forward(self, X):
        """ Forward, backward, communication
        """
        # X is a list of tensors, B x number of tables in a shard x table_feature_dim
        # Forward
        X_len = torch.tensor([x.shape[0] for x in X])
        B = X_len.shape[0]

        X = torch.cat(X, dim=0)
        X = self.table_forward(X)

        ind = torch.repeat_interleave(torch.arange(len(X_len)), X_len)
        tmp = torch.zeros((X_len.shape[0], X.shape[1]))
        tmp.index_add_(0, ind, X)
        X = tmp

        forward_cost = F.relu(self.forward_fc_0(X))
        forward_cost = self.forward_fc_1(forward_cost)
        forward_cost = forward_cost.flatten()

        # Communication
        communication_cost = F.relu(self.communication_fc_0(X))
        communication_cost = self.communication_fc_1(communication_cost)
        communication_cost = communication_cost.flatten()

        # Backward
        backward_cost = F.relu(self.backward_fc_0(X))
        backward_cost = self.backward_fc_1(backward_cost)
        backward_cost = backward_cost.flatten()

        return forward_cost, communication_cost, backward_cost

    def table_forward(self, X):
        # X: B x table_feature_dim
        X = F.relu(self.table_fc_0(X))
        X = F.relu(self.table_fc_1(X))
        return X

    def table_forward_rl(self, X):
        # X: B x table_feature_dim
        X = F.relu(self.rl_table_fc_0(X))
        X = F.relu(self.rl_table_fc_1(X))
        return X

    def forward(self, obs):
        latent = self._get_latent(obs)

        # Policy head
        policy_logits = self.policy_net(latent).squeeze(-1)

        return policy_logits

    def _get_latent(self, table_obs):

        X_len = torch.tensor([[x.shape[0] for x in index_X] for index_X in table_obs])
        B, D = X_len.shape
        X_len = X_len.flatten()
        table_obs = [j for sub in table_obs for j in sub]

        # Get the cost features latent
        with torch.no_grad():
            forward_cost, backward_cost, communication_cost = self.kernel_forward(table_obs)
        cost_obs = torch.cat(
            (
                forward_cost.view(B, D, -1).detach(),
                backward_cost.view(B, D, -1).detach(),
                communication_cost.view(B, D, -1).detach(),
            ),
            dim=-1,
        )
        cost_obs = F.relu(self.cost_fc_0(cost_obs))
        cost_obs = F.relu(self.cost_fc_1(cost_obs))

        # Get the table latent
        table_obs = torch.cat(table_obs, dim=0)
        table_obs = self.table_forward_rl(table_obs)
        ind = torch.repeat_interleave(torch.arange(B*D), X_len)
        tmp = torch.zeros((B*D, table_obs.shape[1]))
        tmp.index_add_(0, ind, table_obs)
        table_obs = tmp.view(B, D, -1)

        latent = torch.cat((table_obs, cost_obs), dim=-1)
        return latent

    def _multi_latent(self, X):
        """ Get the latent for multple shards
        """
        # X: a nested list: B x number of devices x number of tables in a shard x table_feature_dim
        X_len = torch.tensor([[x.shape[0] for x in index_X] for index_X in X])
        B, D = X_len.shape
        X_len = X_len.flatten()

        X = [j for sub in X for j in sub]
        X = torch.cat(X, dim=0)
        X = self.table_forward(X)
 
        ind = torch.repeat_interleave(torch.arange(B*D), X_len)
        tmp = torch.zeros((B*D, X.shape[1]))
        tmp.index_add_(0, ind, X)
        X = tmp.view(B, D, -1)
        X = torch.max(X, dim=1)[0]

        return X    
