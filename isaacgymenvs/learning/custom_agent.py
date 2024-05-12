import torch
from rl_games.algos_torch import a2c_continuous, torch_ext
from rl_games.common.a2c_common import A2CBase, swap_and_flatten01
import time
from collections import OrderedDict


class CustomAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        # A2CBase.__init__(self, base_name, params)

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])

        self.tensor_list += ['next_obses']
        self.tensor_list += ['ref_actions']  # wsh annotation: add ref_actions
        return

    def init_actor_net(self, fn):
        model_params = torch_ext.load_checkpoint(fn)['model']
        actor_net_strings = ['actor_mlp', 'mu']
        actor_net_params = OrderedDict()
        for key, value in model_params.items():
            for search_string in actor_net_strings:
                if search_string in key:
                    actor_net_params[key] = value
                    break
        self.model.load_state_dict(actor_net_params, strict=False)
        print("actor net initialized.")

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            # print(f"sigmas: {res_dict['sigmas']}")
            # self.experience_buffer.update_data('obses', n, self.obs['obs'])
            # self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            self.experience_buffer.update_data('ref_actions', n, infos['ref_actions'])

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float().to(self.device)
            terminated = terminated.unsqueeze(-1)
            next_vals = self.get_values(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        # last_values = self.get_values(self.obs)

        # fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        ref_actions_batch = input_dict['ref_actions']  # wsh annotation: add ref_actions

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo,
                                          curr_e_clip)

            # if self.has_value_loss:
            #     c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch,
            #                                        self.clip_value)
            # else:
            #     c_loss = torch.zeros(1, device=self.ppo_device)

            # wsh annotation: imitation loss as c_loss
            c_loss = torch.mean((mu - ref_actions_batch) ** 2)
            loss = c_loss

            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, entropy, \
                             kl_dist, self.last_lr, lr_mul, \
                             mu.detach(), sigma.detach(), b_loss)

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def _env_reset_done(self):
        obs, done_env_ids = self.vec_env.reset_done()
        return self.obs_to_tensors(obs), done_env_ids
