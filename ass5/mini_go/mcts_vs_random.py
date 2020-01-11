from absl import logging, flags, app
from environment.GoEnv import Go
import time, os
import numpy as np

import tensorflow as tf

from algorimths.mcts import MCTSPlayer
from algorimths.value_dqn import rollout_policy_fn

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", 2,
                     "Number of training episodes for each base policy.")
flags.DEFINE_integer("num_eval", 10,
                     "Number of evaluation episodes")

def main(unused_argv):
    begin = time.time()
    env = Go()
    info_state_size = env.state_size
    num_actions = env.action_size

    import agent.agent as agent
    ret = [0]
    max_len = 2000


    with tf.Session() as sess:
        # agents = [DQN(sess, _idx, info_state_size,
        #                   num_actions, hidden_layers_sizes, **kwargs) for _idx in range(2)]  # for self play
        agents = [MCTSPlayer(rollout_policy_fn), agent.RandomAgent(1)]
        # sess.run(tf.global_variables_initializer())

        # train the mcts agent
        for ep in range(FLAGS.num_train_episodes):
            time_step = env.reset()  # a new env
            print('start')
            while not time_step.last(): # play until the game is over
                
                
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
            time_step.observations["info_state"][cur_player]
            if len(ret) < max_len:
                ret.append(time_step.rewards[0])
            else:
                ret[ep % max_len] = time_step.rewards[0]

        # evaluated the trained mcts agent
        # ret = []
        # for ep in range(FLAGS.num_eval):
        #     time_step = env.reset()
        #     while not time_step.last():
        #         player_id = time_step.observations["current_player"]
        #         agent_output = agents[player_id].step(time_step)
        #         action_list = agent_output.action
        #         time_step = env.step(action_list)
        #     # Episode is over, step all agents with final info state.
        #     # for agent in agents:
        #     agents[0].step(time_step)
        #     agents[1].step(time_step)
        #     ret.append(time_step.rewards[0])
        # print(np.mean(ret))
    print(ret)

    print('Time elapsed:', time.time()-begin)


if __name__ == '__main__':
    app.run(main)
