from src.problem.tsptw.environment.tsptw import *
from src.problem.tsptw.environment.state import *

import torch
import numpy as np
import dgl

class Environment:

    def __init__(self, instance, n_node_feat, n_edge_feat, reward_scaling, grid_size, period_size):
        """
        Initialize the DP/RL environment
        :param instance: a TSPTW instance
        :param n_node_feat: number of features for the nodes
        :param n_edge_feat: number of features for the edges
        :param reward_scaling: value for scaling the reward
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size] (used for normalization purpose)
        :param max_tw_gap: maximum time windows gap allowed between the cities (used for normalization purpose)
        :param max_tw_size: time windows of cities will be in the range [0, max_tw_size] (used for normalization purpose)
        """

        self.instance = instance
        self.n_node_feat = n_node_feat
        self.n_edge_feat = n_edge_feat
        self.reward_scaling = reward_scaling
        self.grid_size = grid_size
        self.period_size = period_size
        self.g = dgl.from_networkx(self.instance.graph)

        self.max_dist = np.sqrt(self.grid_size ** 2 + self.grid_size ** 2)
        # self.max_tw_value = (self.instance.n_city - 1) * (self.max_tw_size + self.max_tw_gap)
        self.ub_cost = self.max_dist * self.instance.n_city

        self.edge_feat_tensor = self.instance.get_edge_feat_tensor(self.max_dist)
        self.number_of_total_actions = 5
        self.count_current_actions = 0

    def get_initial_environment(self):
        """
        Return the initial state of the DP formulation: we are at the city 0 at time 0
        :return: The initial state
        """

        must_visit = set(range(1, self.instance.n_city))  # cities that still have to be visited.
        last_visited = 0  # the current location
        cur_time = 0  # the current time
        cur_tour = [0]  # the tour that is current done
        cur_load = 40

        return State(self.instance, must_visit, last_visited, cur_time, cur_load, cur_tour)

    def make_nn_input(self, cur_state, mode):
        """
        Return a DGL graph serving as input of the neural network. Assign features on the nodes and on the edges
        :param cur_state: the current state of the DP model
        :param mode: on GPU or CPU
        :return: the DGL graph
        """

        # g = dgl.DGLGraph()
        # # g = dgl.graph()
        # g.from_networkx(self.instance.graph)

        #node label: position,  demand, time window, duration
        #edge label: cost and time
        node_feat = []
        # node_feat.append([-1, -1, 0, 0, 0, 0, -1])

        for i in range(cur_state.instance.n_city):
            node_feat.append([self.instance.x_coord[i] / self.grid_size,  # x-coord
                          self.instance.y_coord[i] / self.grid_size,  # y-coord
                          self.instance.service_times[i],
                          self.instance.demands[i]/self.instance.capacity,
                          self.instance.time_windows[i][0] / self.period_size,  # start of the time windows
                          self.instance.time_windows[i][1] / self.period_size,  # end of the time windows
                          0 if i in cur_state.must_visit else 1,  # 0 if it is possible to visit the node
                          1 if i == cur_state.last_visited else 0  # 1 if it is the last node visited
                         ])

        node_feat_tensor = torch.FloatTensor(node_feat).reshape(self.g.number_of_nodes(), self.n_node_feat)

        # feeding features into the dgl_graph
        self.g.ndata['n_feat'] = node_feat_tensor
        self.g.edata['e_feat'] = self.edge_feat_tensor

        if mode == 'gpu':
            self.g.ndata['n_feat'] = self.g.ndata['n_feat'].cuda()
            self.g.edata['e_feat'] = self.g.edata['e_feat'].cuda()

        return self.g

    def get_next_state_with_reward(self, cur_state, action):
        """
        Compute the next_state and the reward of the RL environment from cur_state when an action is done
        :param cur_state: the current state
        :param action: the action that is done
        :return: the next state and the reward collected
        """

        new_state = cur_state.step(action)
        self.count_current_actions += 1


        # see the related paper for the reward definition
        # if (action == 0):
        #     reward = 0
        # if (action == 0):
        #     reward = -0.1
        # # else:
        # else:
        reward = self.ub_cost - self.instance.travel_time[cur_state.last_visited][new_state.last_visited]

        if new_state.is_done(self.count_current_actions):
                        #  cost of going back to the starting city (always 0)
                reward = reward - self.instance.travel_time[new_state.last_visited][0]


        reward = reward * self.reward_scaling

        return new_state, reward

    def get_valid_actions(self, cur_state):
        """
        Compute and return a binary numpy-vector indicating if the action is still possible or not from the current state
        :param cur_state: the current state of the DP model
        :return: a 1D [0,1]-numpy vector a with a[i] == 1 iff action i is still possible
        """

        available = np.zeros(self.instance.n_city , dtype=np.int)
        available_idx = np.array([x for x in cur_state.must_visit], dtype=np.int)
        available[available_idx] = 1
        # available[0] = 1
        # available[1] = 1

        return available
