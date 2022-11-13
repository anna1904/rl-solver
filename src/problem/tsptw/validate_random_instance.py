import sys
import os
import argparse

# sys.path.append(os.path.join(sys.path[0],'..','..','..', '..'))

import torch
import torch.nn as nn
import dgl
import numpy as np

from learning.trainer_dqn import TrainerDQN
from environment.environment import Environment
from main_training_dqn_tsptw import *
# from src.problem.tsptw.learning.trainer_dqn import TrainerDQN
from environment.vrptw import VRPTW
from src.architecture.graph_attention_network import GATNetwork
from src.architecture.graph_attention_network import Transformer

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def test_marl():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    instance = VRPTW.generate_random_instance(n_city=args.n_city, grid_size=args.grid_size,
                                              is_integer_instance=False, capacity=args.capacity,
                                              seed=args.seed)
    print("X_COORD", instance.x_coord)
    print("DEMANDS", instance.demands)
    print('time_windows', instance.time_windows)
    trainer = TrainerDQN(args)
    total_reward, action_list = evaluate_instance(trainer, instance)
    print(action_list)


def select_action(observation, available, trainer):
    """
    Select an action according the to the current model
    :param graph: the graph (first part of the state)
    :param available: the vector of available (second part of the state)
    :return: the action, following the greedy policy with the model prediction
    """
    # batched_graph = dgl.batch([observation[0], ])
    available = available.astype(bool)
    out = predict(observation[0], [observation[1]], trainer)[0].reshape(-1)

    action_idx = np.argmax(out[available])

    action = np.arange(len(out))[available][action_idx]

    return action


def evaluate_instance(trainer, instance):
    """
    Evaluate an instance with the current model
    :param idx: the index of the instance in the validation set
    :return: the reward collected for this instance
    """


    env = Environment(instance, trainer.num_node_feats, trainer.num_edge_feats, trainer.reward_scaling,
                      trainer.args.grid_size, period_size=1000)
    cur_state = env.get_initial_environment()

    total_reward = 0
    action_list = []

    while True:
        graph = env.make_nn_input(cur_state, trainer.args.mode)

        observation = [graph, [cur_state.last_visited, cur_state.cur_load, cur_state.cur_time]]

        avail = env.get_valid_actions(cur_state)

        action = select_action(observation, avail, trainer)

        action_list.append(action)

        cur_state, reward = env.get_next_state_with_reward(cur_state, action)

        total_reward += reward

        if cur_state.is_done(env.count_current_actions):
            break

    return total_reward, action_list


def predict(graph, vehicle, trainer):

    embedding = [(trainer.num_node_feats, trainer.num_edge_feats),
                  (trainer.args.latent_dim, trainer.args.latent_dim),
                  (trainer.args.latent_dim, trainer.args.latent_dim),
                  (trainer.args.latent_dim, trainer.args.latent_dim)]
    device = torch.device('cpu')
    model = Transformer(embedding=embedding, hidden_layer=trainer.args.hidden_layer, latent_dim=trainer.args.latent_dim)
    model.load_state_dict(
        torch.load('/Users/anko/Development/Imperial/rl-solver/src/problem/tsptw/result-default/iter_670_model.pth.tar',
                   map_location=device))
    model.eval()
    res = model(graph, vehicle)
    max_features = [torch.max(i, dim=1).values.detach().numpy() for i in res]
    return max_features

test_marl()