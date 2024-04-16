import torch
from rl_games.algos_torch import players, torch_ext
import time
from collections import OrderedDict


class CustomPlayer(players.PpoPlayerContinuous):
    # self.network -> self.config['network'] -> 'custom_model_continuous.CustomModelContinuous(self.network_builder)';
    # self.network_builder -> network_builder.A2CBuilder();
    # self.model -> CustomModelContinuous.Network(A2CBuilder.Network(cfg.train.params['network'], PpoPlayerContinuous.config),
    #                                             obs_shape, normalize_value, normalize_input, value_size);
    def __init__(self, params):
        super().__init__(params)

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
                self.model.running_mean_std.load_state_dict(input_normalize_params)
        print("actor net initialized.")

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                obses, done_env_ids = self._env_reset_done()
                # time_forward = time.time()
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)
                # print(f"time_forward: {(time.time() - time_forward)*1000} ms")
                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                        all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards / done_count
                        cur_steps_done = cur_steps / done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1} w: {game_res:.1}')
                        else:
                            print(f'reward: {cur_rewards_done:.3f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def _env_reset_done(self):
        obs, done_env_ids = self.env.reset_done()
        return self.obs_to_torch(obs), done_env_ids
