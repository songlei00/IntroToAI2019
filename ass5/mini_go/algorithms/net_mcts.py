import numpy as np 
import copy
import collections
from algorithms.dqn_cnn import DQN
import tensorflow as tf

StepOutput = collections.namedtuple("step_output", ["action", "probs"])

# mcts is referenced from
# https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py
class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors, time_step):
        # print(action_priors)
        actions = action_priors[0]
        priors = action_priors[1]
        cur_player = time_step.observations["current_player"]
        sensible_moves = time_step.observations["legal_actions"][cur_player]

        for action, prob in zip(actions, priors):
            if action not in self._children and action in sensible_moves and action < 25: # action is legal
                self._children[action] = TreeNode(self, prob)
    
    def select(self):
        return max(self._children.items(), key=lambda node: node[1].get_value())
    
    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q)/self._n_visits

    def update_recuesive(self, leaf_value):
        if self._parent:
            # the parent is another player, so add the subtract
            self._parent.update_recuesive(-leaf_value)
        self.update(leaf_value)

    def get_value(self):
        self._u = (self._P*np.sqrt(self._parent._n_visits)/(1+self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS:
    def __init__(self, player_id, value_fn, policy_fn, rollout_fn, n_playout=50, limit=50, lambda_val=1):
        self._root = TreeNode(None, 1.0)
        self._player_id = player_id
        self._value_fn = None
        self._policy_fn = []
        self._rollout_fn = rollout_fn
        self._n_playout = n_playout
        self.limit = limit
        self.lambda_val = lambda_val

        kwargs = {
            "replay_buffer_capacity": int(5e4),
            "epsilon_decay_duration": int(600),
            "epsilon_start": 0.8,
            "epsilon_end": 0.001,
            "learning_rate": 1e-3,
            "learn_every": 2000,
            "batch_size": 128,
            "max_global_gradient_norm": 10,
        }

        # output_channels, kernel_shapes, strides, paddings
        graph_1 = tf.Graph()
        self.sess_1 = tf.Session(graph=graph_1)
        with graph_1.as_default():
            self._policy_fn.append(DQN(self.sess_1, 0, 25, 26, [128, 128], **kwargs))
            saver = tf.train.import_meta_graph(policy_fn[0] + '.meta', clear_devices=True)
            saver.restore(self.sess_1, policy_fn[0])
            self.sess_1.run(tf.global_variables_initializer())

        graph_2 = tf.Graph()
        self.sess_2 = tf.Session(graph=graph_2)
        with graph_2.as_default():
            self._policy_fn.append(DQN(self.sess_2, 0, 25, 26, [128, 128], **kwargs))
            saver = tf.train.import_meta_graph(policy_fn[1] + '.meta', clear_devices=True)
            saver.restore(self.sess_2, policy_fn[1])
            self.sess_2.run(tf.global_variables_initializer())

        graph_val = tf.Graph()
        self.sess_val = tf.Session(graph=graph_val)
        with graph_val.as_default():
            self._value_fn = DQN(self.sess_val, 0, 25, 26, [128, 128], **kwargs)
            saver = tf.train.import_meta_graph(value_fn + '.meta', clear_devices=True)
            saver.restore(self.sess_val, value_fn)
            self.sess_val.run(tf.global_variables_initializer())

    def playout(self, time_step, env):
        # copy to avoid to change original attributes
        node = self._root
        player_id = self._player_id
        env_cpy = copy.deepcopy(env)

        while 1:
            if node.is_leaf():
                break
            action, node = node.select()
            time_step = env_cpy.step(action)
            player_id = 0 if player_id else 1

        # expand
        cur_player = time_step.observations["current_player"]
        info_state = time_step.observations["info_state"][cur_player]
        legal_actions = time_step.observations["legal_actions"][cur_player]
        action_probs = self._policy_fn[0]._get_policy(info_state, legal_actions)
        # print(action_probs)
        time_step = env_cpy.step(action_probs[0][np.argmax(action_probs[1])])
        node.expand(action_probs, time_step)
        
        # calculate the value
        val_1 = self._value_fn._get_value(info_state, legal_actions)
        # print(np.average(val_1), np.sum(val_1))
        val_1 = np.sum(val_1)
        val_2 = self.rollout(time_step, env_cpy)
        leaf_value = (1-self.lambda_val)*val_1 + self.lambda_val*val_2

        node.update_recuesive(leaf_value)

    def rollout(self, time_step, env):
        player_id = time_step.observations["current_player"]
        env_cpy = copy.deepcopy(env)
        while not time_step.last():
            action_probs = self._rollout_fn(time_step)
            # print(action_probs)
            cur_player = time_step.observations["current_player"]
            sensible_moves = time_step.observations["legal_actions"][cur_player]
            # print(sensible_moves)
            best_action = action_probs[0][np.argmax(action_probs[1])]
            # print(best_action)
            time_step = env_cpy.step(best_action)
            player_id = 0 if player_id else 1
        
        return time_step.rewards[0]

    def get_move_probs(self, time_step, env, tmp=1e-3):
        self._player_id = time_step.observations["current_player"]
        
        for _ in range(self._n_playout):
            env_cpy = copy.deepcopy(env)
            self.playout(time_step, env_cpy)
            
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        # print(act_visits)
        if act_visits == []:
            # print("finish")
            return

        acts, visits = zip(*act_visits)
        act_probs = self.softmax(1.0/tmp*np.log(np.array(visits)+1e-10))
        # print("get_move_probs", acts)

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

NUM_ACTIONS = 26

def rollout_policy_fn(time_step):
    cur_player = time_step.observations["current_player"]
    sensible_moves = time_step.observations["legal_actions"][cur_player]
    action_probs = np.random.rand(len(sensible_moves))

    return sensible_moves, action_probs

class MCTSAgent():

    def __init__(self, value_module=None, policy_module=None, n_playout=50):
        self._value_fn = value_module
        self._policy_fn = policy_module
        self._rollout_fn = rollout_policy_fn

        self.mcts = MCTS(0, self._value_fn, self._policy_fn, self._rollout_fn, n_playout)

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
            # print('aaa')
            # print(acts)

            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)
            # print(move)
            
            return StepOutput(action=move, probs=move_probs)
        else:
            return StepOutput(action=move, probs=move_probs)
