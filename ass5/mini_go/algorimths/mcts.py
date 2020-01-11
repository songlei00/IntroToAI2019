import numpy as np 
import collections
import copy
from environment.GoEnv import Go

# need to modify, policy to probs
StepOutput = collections.namedtuple("step_output", ["action", "policy"])

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class SearchNode:
    # __slots__ = [
    #     "action",
    #     "explore_count",
    #     "total_reward",
    #     "children",
    # ]

    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0 # ucb
        self.p = prior
        
    
    # select the best node until reach a leaf
    def select(self, c_puct):
        return max(self.children.items(), key=lambda node: node[1].get_value(c_puct))

    def expand(self, action_priors, time_step):
        # need to modify
        # print(action_priors[0])
        actions = action_priors[0]
        priors = action_priors[1]
        cur_player = time_step.observations["current_player"]
        sensible_moves = time_step.observations["legal_actions"][cur_player]
        for action, prob in zip(actions, priors):
            # need to modify
            if action not in self.children and action in sensible_moves and action < 25:
                self.children[action] = SearchNode(self, prob)

    
    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += 1.0*(leaf_value - self.Q)/self.n_visits

    def update_recuesive(self, leaf_value):
        if self.parent:
            # the parent is another player, so add the subtract
            self.parent.update_recuesive(-leaf_value)
        self.update(leaf_value)
    
    def get_value(self, c_puct):
        self.u = (c_puct*self.p*np.sqrt(self.parent.n_visits)/(1+self.n_visits))

        return self.Q + self.u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.panent is None


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self.root = SearchNode(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def playout(self, time_step, env):
        node = self.root
        env_cpy = copy.deepcopy(env)
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            time_step = env_cpy.step(action)

        action_probs = self.policy(time_step)
        end = time_step.last()
        winner = time_step.rewards[0]
        leaf_value = 0.0
        
        if not end:
            # print(action_probs)
            node.expand(action_probs, time_step)
        else:
            # need to modify
            # print("winner:", winner)
            if winner == 0:
                leaf_value = 0.0
            else:
                cur_player = time_step.observations["current_player"]
                leaf_value = (
                    1.0 if winner == cur_player else -1.0
                )
            
        node.update_recuesive(-leaf_value)

    def get_move_probs(self, time_step, env, tmp=1e-3):
        for _ in range(self.n_playout):
            # no need to copy
            env_cpy = copy.deepcopy(env)
            
            self.playout(time_step, env)
        
        act_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        # print(act_visits)
        if act_visits == []:
            print("finish")
            return

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/tmp*np.log(np.array(visits)+1e-10))
        # print("get_move_probs", acts)

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = SearchNode(None, 1.0)
    
    def __str__(self):
        return "MCTS"

    '''------------------------------'''

    def step(self, time_step, env):
        
        # if (not time_step.last()) and (self.player_id == time_step.current_player()):
        #     root = self.mcts_search(time_step)
        #     best = root.best_child()
        #     action = best.action
        # else:
        #     action = []

        # print("aaa")
        # print(time_step)
        self.env_cpy = copy.deepcopy(env)
        best_action = self.mcts_search(time_step)
        # best_action = root.choose_best()

        # cur_player = time_step.observations["current_player"]
        # legal_actions = time_step.observations["legal_actions"][cur_player]
        # print(time_step.observations["info_state"][cur_player])
        # # print(legal_actions)
        # action = legal_actions[0]

        return StepOutput(action=best_action, policy=None)

class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self.is_selfplay = is_selfplay
    
    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def step(self, time_step, env, tmp=1e-3):
        cur_player = time_step.observations["current_player"]
        sensible_moves = time_step.observations["legal_actions"][cur_player]
        move_probs = np.zeros(5*5)
        if len(sensible_moves) > 0:
            # print('aaa')
            ret = self.mcts.get_move_probs(time_step, env)
            if ret == None:
                print("finish")
                return StepOutput(action=25, policy=[1.0])
            
            acts, probs = ret
            # print('aaa')
            # print(acts)

            move_probs[list(acts)] = probs
            if self.is_selfplay:
                move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
            print(move)
            
            # need to modify
            return StepOutput(action=move, policy=move_probs)
        else:
            return StepOutput(action=move, policy=move_probs)
        

