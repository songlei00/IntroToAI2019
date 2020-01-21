from absl import logging, flags, app
from environment.GoEnv import Go
import time, os
import numpy as np
import tensorflow as tf
import random

import agent.agent as agent

NUM_EVAL = 10
NUM_TRAIN = 20
NUM_SAVE_EVERY = 10
max_len = 100

def main(unused_argv):
    begin = time.time()
    env = Go()
    ret = [0]

    # policy_function = 'saved_model/%d'%(random.randint(1, 5)*2000)
    # value_function = 'saved_model/%d'%(random.randint(1, 5)*2000)

    policy_function = 'saved_model/2000'
    value_function = 'saved_model/4000'

    policy_function = ['saved_model/self_play_policy_fn_30', 'saved_model/self_play_policy_fn_30']
    value_function = 'saved_model/self_play_value_fn_30'

    agents = [agent.Net_MCTS_Agent(value_function, policy_function, n_playout=50), 
        agent.RandomAgent(1)]
    
    for ep in range(NUM_TRAIN):
        if (ep + 1) % NUM_SAVE_EVERY == 0:
            if not os.path.exists("saved_model"):
                os.mkdir('saved_model')
            agents[0].mcts._policy_fn[0].save(checkpoint_root='saved_model', checkpoint_name='_policy_fn_0_{}'.format(ep+1))
            agents[0].mcts._policy_fn[1].save(checkpoint_root='saved_model', checkpoint_name='_policy_fn_1_{}'.format(ep+1))
            agents[0].mcts._value_fn.save(checkpoint_root='saved_model', checkpoint_name='_value_fn_{}'.format(ep+1))

        time_step = env.reset()  # a new env
        print('start ep: %d'%ep)
        while not time_step.last(): # play until the game is over
            cur_player = time_step.observations["current_player"]
            state = time_step.observations["info_state"][cur_player]
            player_id = time_step.observations["current_player"]
            if player_id == 0:
                agent_output = agents[player_id].step(time_step, env)
            else:
                agent_output = agents[player_id].step(time_step)
            action_list = agent_output.action
            time_step = env.step(action_list)
        print('end')

        agents[0].step(time_step, env)
        agents[1].step(time_step)
        if len(ret) < max_len:
            ret.append(time_step.rewards[0])
        else:
            ret[ep % max_len] = time_step.rewards[0]

    # evaluated the trained mcts agent
    ret = []
    for ep in range(NUM_EVAL):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if player_id == 0:
                agent_output = agents[player_id].step(time_step, env)
            else:
                agent_output = agents[player_id].step(time_step)
            action_list = agent_output.action
            time_step = env.step(action_list)
        # Episode is over, step all agents with final info state.
        agents[0].step(time_step, env)
        agents[1].step(time_step)
        ret.append(time_step.rewards[0])
    print(np.mean(ret))
    print(ret)

    print('Time elapsed:', time.time()-begin)


if __name__ == '__main__':
    app.run(main)
