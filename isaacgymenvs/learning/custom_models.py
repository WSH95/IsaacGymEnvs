import torch
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class CustomModelContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    class Network(ModelA2CContinuousLogStd.Network):
        # self.a2c_network -> A2CBuilder.Network(cfg.train.params['network'], PpoPlayerContinuous.config)
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            # Apply clamp to logstd
            logstd = torch.clamp(logstd, min=-10, max=2)
            sigma = torch.exp(logstd)
            # sigma = torch.clamp(sigma, 0.2, None)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp': torch.squeeze(prev_neglogp),
                    'values': value,
                    'entropy': entropy,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs': torch.squeeze(neglogp),
                    'values': self.unnorm_value(value),
                    'actions': selected_action,
                    'rnn_states': states,
                    'mus': mu,
                    'sigmas': sigma
                }
                return result
