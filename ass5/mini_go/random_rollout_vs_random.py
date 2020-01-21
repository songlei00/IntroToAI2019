from absl import logging, app
from environment.GoEnv import Go
import time, os
import numpy as np
import tensorflow as tf
import random
import agent.agent as agent

NUM_EPISODES = 10

def main(unused_argv):
    begin = time.time()
    env = Go()
    agents = [agent.Random_Rollout_MCTS_Agent(n_playout=100), agent.RandomAgent(1)]
    ret = []

    for ep in range(NUM_EPISODES):
        time_step = env.reset()
        print('start ep: %d'%ep)
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
        print('end')
    print(np.mean(ret))
    print(ret)

    print('Time elapsed:', time.time()-begin)


if __name__ == '__main__':
    app.run(main)
