import networkx as nx
import random
import heapq
import numpy as np
import torch
import pandas as pd
import numpy as np

class VRPTW: 
    def __init__(self, n_city, depot_location, travel_time, x_coord, y_coord, time_windows, service_times, demands, capacity):
        """
        Create an instance of the TSPTW problem
        :param n_city: number of cities with depot
        :param travel_time: travel time matrix between the cities
        :param x_coord: list of x-pos of the cities
        :param y_coord: list of y-pos of the cities
        :param time_windows: list of time windows of all cities ([lb, ub] for each city)
        :param service_times: list of service times for each city
        """

        self.n_city = n_city
        self.travel_time = travel_time
        self.depot_location = depot_location
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.time_windows = time_windows
        self.service_times = service_times
        self.demands = demands
        self.capacity = capacity

        self.graph = self.build_graph()

    def build_graph(self, n_dummy_nodes = 1):
        """
        Build a networkX graph representing the VRPTW instance. Features on the edges are the distances
        and 4 binary values stating if the edge is part of the (1, 5, 10, 20) nearest neighbors of a node?
        :return: the graph
        """

        g = nx.DiGraph()

        # g.add_node(0) ##TODO add more then one dummy node in a loop
        # g.add_edge(0, 0, weight=1000000, is_k_neigh_1=0, is_k_neigh_5=0,
        #            is_k_neigh_10=0, is_k_neigh_20=0)

        for i in range(self.n_city):

            cur_travel_time = self.travel_time[i][:]

            # +1 because we remove the self-edge (cost 0)
            k_min_idx_1 = heapq.nsmallest(1 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__) # find the index of 2 smallest elements (the first with duplicates)
            k_min_idx_5 = heapq.nsmallest(5 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)
            k_min_idx_10 = heapq.nsmallest(10 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)
            k_min_idx_20 = heapq.nsmallest(20 + 1, range(len(cur_travel_time)), cur_travel_time.__getitem__)

            for j in range(self.n_city):

                if i != j:
                    is_k_neigh_1 = 1 if j in k_min_idx_1 else 0
                    is_k_neigh_5 = 1 if j in k_min_idx_5 else 0
                    is_k_neigh_10 = 1 if j in k_min_idx_10 else 0
                    is_k_neigh_20 = 1 if j in k_min_idx_20 else 0

                    weight = self.travel_time[i][j]
                    g.add_edge(i, j, weight=weight, is_k_neigh_1=is_k_neigh_1, is_k_neigh_5=is_k_neigh_5,
                               is_k_neigh_10=is_k_neigh_10, is_k_neigh_20=is_k_neigh_20)
        assert g.number_of_nodes() == (self.n_city)

        return g

    def get_edge_feat_tensor(self, max_dist):
        """
        Return a tensor of the edges features.
        As the features for the edges are not state-dependent, we can pre-compute them
        :param max_dist: the maximum_distance possible given the grid-size
        :return: a torch tensor of the features
        """

        edge_feat = [[e[2]["weight"] / max_dist,
                    e[2]["is_k_neigh_1"],
                    e[2]["is_k_neigh_5"],
                    e[2]["is_k_neigh_10"],
                    e[2]["is_k_neigh_20"]]
                    for e in self.graph.edges(data=True)]

        edge_feat_tensor = torch.FloatTensor(edge_feat)

        return edge_feat_tensor

    @staticmethod
    def read_instance(path):

        data = pd.read_csv(path, header = 0, delim_whitespace=True, nrows = 20)

        n_city = len(data) 
        x_coord = data['XCOORD.'].to_numpy()
        y_coord = data['YCOORD.'].to_numpy()
        time_windows = [[i,j] for i in data['READY_TIME'] for j in data['DUE_DATE']]
        service_time = data['SERVICE_TIME'].to_numpy()

        travel_time = []
        for i in range(n_city):

            dist = [float(np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2))
                    for j in range(n_city)]

            dist = [round(x) for x in dist]

            travel_time.append(dist)
        return VRPTW(n_city, travel_time, x_coord, y_coord, time_windows, service_time, 2)

    def generate_order(self, current_time, scenario, seed):

        rand = random.Random()

        if seed != -1:
            rand.seed(seed)

        x_coord = rand.uniform(0, self.grid_size)
        y_coord = rand.uniform(0, self.grid_size)

        service_time = rand.uniform(5, 30)
        demand = rand.uniform(1, int(self.capacity / 5))

        travel_time = [round(np.sqrt((x_coord[i] - x_coord) ** 2 + (y_coord[i] - y_coord) ** 2)) for i in range(self.n_city)]

        #scenario is define either order is before noon or  afternoon
        #current time is to generate order after specific point in time

        a_sample = np.ceil(
            float(np.sqrt((self.depot_location[0] - x_coord) ** 2 + (self.depot_location[1] - y_coord) ** 2))) + 1
        b_sample = 1000 - a_sample
        a = np.ceil(np.random.uniform(a_sample, b_sample))
        epsilon = np.maximum(np.abs(np.random.normal(0, 1)), 1 / 100)
        b = min(np.floor(a + 300 * epsilon), b_sample)
        time_windows = [a, b]

        return [x_coord, y_coord, service_time, demand, travel_time, time_windows]

    @staticmethod
    def receive_dataset(size, path):

        #path is an array of instances
        dataset = []
        for i in range(size):
            new_instance = VRPTW.read_instance(path[i])
            dataset.append(new_instance)

        return dataset
    @staticmethod
    def generate_random_instance(n_city, grid_size,
                                 is_integer_instance, capacity, seed):
        """
        :param n_city: number of cities with depot
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param is_integer_instance: True if we want the distances to have integer values
        :param seed: seed used for generating the instance. -1 means no seed (instance is random)
        :return: a feasible TSPTW instance randomly generated using the parameters
        """

        rand = random.Random()

        if seed != -1:
            rand.seed(seed)

        x_coord = [rand.uniform(0, grid_size) for _ in range(n_city-1)]
        y_coord = [rand.uniform(0, grid_size) for _ in range(n_city-1)]

        service_times = [rand.uniform(5, 30) for _ in range(n_city-1)]
        demands = [rand.uniform(1, int(capacity/5)) for _ in range(n_city-1)]

        #depot as a first node
        depot_location = (round(grid_size / 2), round(grid_size / 2))
        x_coord.insert(0, depot_location[0])
        y_coord.insert(0, depot_location[1])
        service_times.insert(0, 0)
        demands.insert(0, 0)


        travel_time = []
        time_windows = np.zeros((n_city, 2))

        for i in range(n_city):

            dist = [float(np.sqrt((x_coord[i] - x_coord[j]) ** 2 + (y_coord[i] - y_coord[j]) ** 2))
                    for j in range(n_city)]

            if is_integer_instance:
                dist = [round(x) for x in dist]

            travel_time.append(dist)


        time_windows[0] = [0, 1000]
        #a0 = 0, b0 = end of the day= 1000

        # TW start needs to be feasibly reachable directly from depot

        #arxiv.org/pdf/2006.09100.pdf
        for i in range(1, n_city):
            a_sample = np.ceil(float(np.sqrt((depot_location[0] - x_coord[i]) ** 2 + (depot_location[1] - y_coord[i]) ** 2))) + 1
            b_sample = 1000 - a_sample
            a = np.ceil(np.random.uniform(a_sample, b_sample))
            epsilon = np.maximum(np.abs(np.random.normal(0,1)), 1 / 100)
            b = min(np.floor(a + 300 * epsilon), b_sample)
            time_windows[i] = [a, b]

        return VRPTW(n_city, depot_location, travel_time, x_coord, y_coord, time_windows, service_times, demands, capacity)

    @staticmethod
    def generate_dataset(size, n_city, grid_size, is_integer_instance, capacity,  seed):
        """
        Generate a dataset of instance
        :param size: the size of the data set
        :param n_city: number of cities
        :param grid_size: x-pos/y-pos of cities will be in the range [0, grid_size]
        :param is_integer_instance: True if we want the distances to have integer values
        :param seed: the seed used for generating the instance
        :return: a dataset of '#size' feasible TSPTW instance randomly generated using the parameters
        """
        dataset = []
        for i in range(size):
            new_instance = VRPTW.generate_random_instance(n_city=n_city, grid_size=grid_size,
                                                          is_integer_instance=is_integer_instance, capacity = capacity, seed=seed)
            dataset.append(new_instance)
            seed += 1

        return dataset
        

# problem = VRPTW.read_instance('instances/test.txt')
# print(problem.graph.edges(data=True))
# problem = VRPTW.generate_random_instance(5, 100, True, 20, 1234)


