import torch
import torch.nn as nn
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from collections import OrderedDict


class GaitTrackingPolicy(nn.Module):
    def __init__(self,
                 actions_num,
                 obs_shape,
                 unit,
                 normalize_input=True):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape

        self.a2c_network = ActorNetwork(actions_num, obs_shape, unit)

        self.normalize_input = normalize_input
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape)

    def forward(self, obs):
        norm_obs = self.norm_obs(obs)
        mu = self.a2c_network(norm_obs)
        # print(mu[0])
        return mu

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def restore_from_file(self, fn):
        model_params = torch_ext.load_checkpoint(fn)['model']
        actor_net_strings = ['actor_mlp', 'mu']
        actor_net_params = OrderedDict()
        for key, value in model_params.items():
            for search_string in actor_net_strings:
                if search_string in key:
                    actor_net_params[key] = value
                    break
        self.load_state_dict(actor_net_params, strict=False)

        if self.normalize_input:
            input_normalize_strings = ['running_mean_std']
            input_normalize_params = OrderedDict()
            for key, value in model_params.items():
                # Check if any of the search strings are in the key
                if any(item in key for item in input_normalize_strings):
                    # Iterate over all occurrences of the search strings in the key
                    for search_string in input_normalize_strings:
                        for i in range(len(key) - len(search_string) + 1):
                            # Extract the portion of the key value that contains the search string
                            substring = key[i:i + len(search_string)]
                            if substring == search_string and i + len(search_string) < len(key) - 1:
                                input_normalize_params[key[i + len(search_string) + 1:]] = value
                                break
            if input_normalize_params:
                self.running_mean_std.load_state_dict(input_normalize_params)
        print("gait tracking policy actor net initialized.")


class ActorNetwork(nn.Module):
    def __init__(self, actions_num, input_size, units):
        nn.Module.__init__(self)

        self.actions_num = actions_num
        self.input_size = input_size
        self.units = units

        self.actor_mlp = nn.Sequential()
        print('build mlp:', input_size)
        in_size = self.input_size
        layers = []
        for unit in self.units:
            layers.append(nn.Linear(in_size, unit))
            layers.append(nn.ELU())
            in_size = unit
        self.actor_mlp = nn.Sequential(*layers)

        self.mu = nn.Linear(self.units[-1], self.actions_num)
        self.mu_act = nn.Identity()

    def forward(self, obs):
        a_out = obs
        a_out = a_out.contiguous().view(a_out.size(0), -1)
        a_out = self.actor_mlp(a_out)
        mu = self.mu_act(self.mu(a_out))

        return mu
