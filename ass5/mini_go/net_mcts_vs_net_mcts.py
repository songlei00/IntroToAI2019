from absl import logging, flags, app
from environment.GoEnv import Go
import time, os
import numpy as np
import tensorflow as tf
import random
import agent.agent as agent

NUM_EVAL = 100
NUM_TRAIN = 100
NUM_SAVE_EVERY = 20
max_len = 100

def main(unused_argv):
    begin = time.time()
    env = Go()
    ret = [0]

    best_policy = ['saved_model/self_play/self_play_policy_fn_0_100',
                    'saved_model/self_play/self_play_policy_fn_1_100']
    best_value = 'saved_model/self_play/self_play_value_fn_100'

    random_policy = ['saved_model/self_play/self_play_policy_fn_0_%d'%(random.randint(1, 5)*20),
                    'saved_model/self_play/self_play_policy_fn_1_%d'%(random.randint(1, 5)*20)]
    random_value = 'saved_model/self_play/self_play_value_fn_%d'%(random.randint(1, 5)*20)

    agents = [agent.Net_MCTS_Agent(best_value, best_policy, n_playout=50), 
        agent.Net_MCTS_Agent(random_value, random_policy, n_playout=50)]
    
    for ep in range(NUM_TRAIN):
        if (ep + 1) % NUM_SAVE_EVERY == 0:
            if not os.path.exists("saved_model/self_play"):
                os.mkdir('saved_model/self_play')
            agents[0].mcts._policy_fn[0].save(checkpoint_root='saved_model/self_play', 
                            checkpoint_name='self_play_policy_fn_0_{}'.format(ep+1))
            agents[0].mcts._policy_fn[1].save(checkpoint_root='saved_model/self_play', 
                            checkpoint_name='self_play_policy_fn_1_{}'.format(ep+1))
            agents[0].mcts._value_fn.save(checkpoint_root='saved_model/self_play', 
                            checkpoint_name='self_play_value_fn_{}'.format(ep+1))

        time_step = env.reset()  # a new env
        print('start ep: %d'%ep)
        while not time_step.last(): # play until the game is over
            cur_player = time_step.observations["current_player"]
            state = time_step.observations["info_state"][cur_player]
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step, env)
            action_list = agent_output.action
            time_step = env.step(action_list)
        print('end')

        agents[0].step(time_step, env)
        agents[1].step(time_step, env)
        if len(ret) < max_len:
            ret.append(time_step.rewards[0])
        else:
            ret[ep % max_len] = time_step.rewards[0]

    # evaluated the trained mcts agent
    ret = []
    for ep in range(NUM_EVAL):
        print('eval ep: %d'%ep)
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step, env)
            action_list = agent_output.action
            time_step = env.step(action_list)
        # Episode is over, step all agents with final info state.
        agents[0].step(time_step, env)
        agents[1].step(time_step, env)
        ret.append(time_step.rewards[0])
    print(np.mean(ret))
    print(ret)

    print('Time elapsed:', time.time()-begin)


if __name__ == '__main__':
    app.run(main)
