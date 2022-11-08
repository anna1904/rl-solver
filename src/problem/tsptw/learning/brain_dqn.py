
import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np
import dgl

from src.architecture.graph_attention_network import Transformer


class BrainDQN:
    """
    Definition of the DQN Brain, computing the DQN loss
    """

    def __init__(self, args, num_node_feat, num_edge_feat):
        """
        Initialization of the DQN Brain
        :param args: argparse object taking hyperparameters
        :param num_node_feat: number of features on the nodes
        :param num_edge_feat: number of features on the edges
        """

        self.args = args

        self.embedding = [(num_node_feat, num_edge_feat),
                         (self.args.latent_dim, self.args.latent_dim),
                         (self.args.latent_dim, self.args.latent_dim),
                         (self.args.latent_dim, self.args.latent_dim)]

        self.model = Transformer(embedding = self.embedding, hidden_layer = self.args.hidden_layer, latent_dim = self.args.latent_dim)
        self.target_model = Transformer(embedding = self.embedding, hidden_layer = self.args.hidden_layer, latent_dim = self.args.latent_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.mode == 'gpu':
            self.model.cuda()
            self.target_model.cuda()

    def train(self, x, y):
        """
        Compute the loss between (f(x) and y)
        :param x: the input
        :param y: the true value of y
        :return: the loss
        """

        self.model.train()

        observation, _ = list(zip(*x))
        graph = [g[0] for g in observation]
        vehicle = [v[1] for v in observation]
        graph_batch = dgl.batch(graph)
        y_pred = self.model(graph_batch, vehicle)
        max_features = [torch.max(i, dim=1).values for i in y_pred]
        y_pred = torch.stack(max_features)
        y_tensor = torch.FloatTensor(np.array(y))

        if self.args.mode == 'gpu':
            y_tensor = y_tensor.contiguous().cuda()

        loss = F.smooth_l1_loss(y_pred, y_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, graph, vehicle, target):
        """
        Predict the Q-values using the current graph, either using the model or the target model
        :param graph: the graph serving as input
        :param target: True is the target network must be used for the prediction
        :return: A list of the predictions for each node
        """
        with torch.no_grad():
            if target:
                self.target_model.eval()
                res = self.target_model(graph, vehicle)
            else:
                self.model.eval()
                res = self.model(graph, vehicle)

        max_features = [torch.max(i, dim=1).values.numpy() for i in res]
        # for i in range(len(res)):
        #     res[i] = torch.max(res[i], dim=1).values
        #     res[i] = res[i].numpy()
        #[r.cpu().data.numpy().flatten() for r in res]
        return max_features

    def update_target_model(self):
        """
        Synchronise the target network with the current one
        """

        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, folder, filename):
        """
        Save the model
        :param folder: Folder requested
        :param filename: file name requested
        """

        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.model.state_dict(), filepath)
