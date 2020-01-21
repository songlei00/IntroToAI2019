import random, collections
import numpy as np
StepOutput = collections.namedtuple("step_output", ["action", "probs"])

class Agent(object):
    def __init__(self):
        pass

    def step(self, timestep):
        raise NotImplementedError

# random agent
class RandomAgent(Agent):
    def __init__(self, _id):
        super().__init__()
        self.player_id = _id

    def step(self, timestep):
        cur_player = timestep.observations["current_player"]
        return StepOutput(action=random.choice(timestep.observations["legal_actions"][cur_player]), probs=1.0)

# random rollout mcts agent
from algorithms.random_rollout_mcts import MCTS as Random_Rollout_MCTS

def rollout_policy_fn(time_step):
    cur_player = time_step.observations["current_player"]
    sensible_moves = time_step.observations["legal_actions"][cur_player]
    action_probs = np.random.rand(len(sensible_moves))

    return sensible_moves, action_probs

class Random_Rollout_MCTS_Agent(Agent):
    def __init__(self, max_simulations=200):
        self.mcts = Random_Rollout_MCTS(rollout_policy_fn, max_simulations)

    def step(self, time_step, env):
        cur_player = time_step.observations["current_player"]
        sensible_moves = time_step.observations["legal_actions"][cur_player]
        move_probs = np.zeros(5*5)

        if len(sensible_moves) > 0:
            ret = self.mcts.get_move_probs(time_step, env)
            if ret == None:
                return StepOutput(action=25, probs=[1.0]) # pass
            
            acts, probs = ret
            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
            # print(move)
            
            return StepOutput(action=move, probs=move_probs)
        else:
            return StepOutput(action=25, probs=[1.0]) # pass

from algorithms.net_mcts import MCTS as NETMCTS

class Net_MCTS_Agent(Agent):
    # need to modify
    def __init__(self, alg, policy_model=None, value_model=None, rollout_fn=None, n_playout=100):
        if rollout_fn == None:
            rollout_fn = rollout_policy_fn
        self.mcts = NETMCTS(alg, policy_model, value_model, rollout_fn, n_playout)
        
    def step(self, time_step, env):
        cur_player = time_step.observations["current_player"]
        sensible_moves = time_step.observations["legal_actions"][cur_player]
        move_probs = np.zeros(5*5)
        if len(sensible_moves) > 0:
            ret = self.mcts.get_move_probs(time_step, env)
            if ret == None:
                # need to modify
                # print("finish")
                return StepOutput(action=25, probs=[1.0]) # pass 
            
            acts, probs = ret

            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
            # print(move)
            
            return StepOutput(action=move, probs=move_probs)
        else:
            return StepOutput(action=25, probs=[1.0]) # pass
