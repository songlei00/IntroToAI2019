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
    def __init__(self, rollout_module=None, n_playout=50):
        # self._rollout_fn = rollout_module
        self._rollout_fn = rollout_policy_fn

        self.mcts = Random_Rollout_MCTS(self._rollout_fn, n_playout)

    def step(self, time_step, env):
        cur_player = time_step.observations["current_player"]
        sensible_moves = time_step.observations["legal_actions"][cur_player]
        move_probs = np.zeros(5*5)
        if len(sensible_moves) > 0:
            # print('aaa')
            ret = self.mcts.get_move_probs(time_step, env)
            if ret == None:
                # print("finish")
                return StepOutput(action=25, probs=[1.0])
            
            acts, probs = ret
            # print(acts)

            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
            # print(move)
            
            return StepOutput(action=move, probs=move_probs)
        else:
            return StepOutput(action=move, probs=move_probs)


# mcts using neural network
from algorithms.net_mcts import MCTS as NetMCTS

class Net_MCTS_Agent(Agent):
    def __init__(self, value_module=None, policy_module=None, n_playout=50):
        self._value_fn = value_module
        self._policy_fn = policy_module
        self._rollout_fn = rollout_policy_fn

        self.mcts = NetMCTS(self._value_fn, self._policy_fn, self._rollout_fn, n_playout)

    def step(self, time_step, env):
        cur_player = time_step.observations["current_player"]
        sensible_moves = time_step.observations["legal_actions"][cur_player]
        move_probs = np.zeros(5*5)
        if len(sensible_moves) > 0:
            # print('aaa')
            ret = self.mcts.get_move_probs(time_step, env)
            if ret == None:
                # print("finish")
                return StepOutput(action=25, probs=[1.0])
            
            acts, probs = ret
            # print(acts)

            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
            # print(move)
            
            return StepOutput(action=move, probs=move_probs)
        else:
            return StepOutput(action=move, probs=move_probs)
