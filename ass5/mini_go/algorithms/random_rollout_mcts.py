import numpy as np 
import collections
import copy
from environment.GoEnv import Go
import tensorflow as tf

StepOutput = collections.namedtuple("step_output", ["action", "probs"])

# mcts is referenced from
# https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py
class TreeNode:
    def __init__(self, parent, prior):
        self.parent = parent
        self.p = prior
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0 # ucb
    
    # select the best node
    def select(self):
        return max(self.children.items(), key=lambda node: node[1].get_value())

    def expand(self, action_priors, time_step):
        actions = action_priors[0]
        priors = action_priors[1]
        cur_player = time_step.observations["current_player"]
        sensible_moves = time_step.observations["legal_actions"][cur_player]

        for action, prob in zip(actions, priors):
            if action not in self.children and action in sensible_moves and action < 25: # action is legal
                self.children[action] = TreeNode(self, prob)
    
    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += 1.0*(leaf_value - self.Q)/self.n_visits

    def update_back(self, leaf_value):
        if self.parent:
            # the parent is another player, so add the subtract
            self.parent.update_back(-leaf_value)
        self.update(leaf_value)
    
    def get_value(self):
        self.u = (3*self.p*np.sqrt(self.parent.n_visits)/(1+self.n_visits))

        return self.Q + self.u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.panent is None


class MCTS:
    def __init__(self, fn, max_simulations=200):
        self.root = TreeNode(None, 1.0)
        self.policy = fn
        self.max_simulations = max_simulations

    def simulation(self, time_step, env):
        node = self.root
        env_cpy = copy.deepcopy(env)

        # reach the leaf
        while node.is_leaf() == 0:
            action, node = node.select()
            time_step = env_cpy.step(action)
        # random rollout until the game is over
        action_probs = self.policy(time_step)
        # print(action_probs)
        time_step = env_cpy.step(action_probs[0][np.argmax(action_probs[1])])

        end = time_step.last()
        leaf_value = 0.0
        
        if not end:
            # print(action_probs)
            node.expand(action_probs, time_step)
        else:
            cur_player = time_step.observations["current_player"]
            leaf_value = -time_step.rewards[cur_player]

        node.update_back(-leaf_value)

    def get_move_probs(self, time_step, env, tmp=1e-3):
        # rollout 
        for i in range(self.max_simulations):
            self.simulation(time_step, env)
        
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        if act_visits == []:
            return

        acts, visits = zip(*act_visits)
        act_probs = self.softmax(1.0/tmp*np.log(np.array(visits)+1e-10))
        # print("get_move_probs", acts)

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs
    
