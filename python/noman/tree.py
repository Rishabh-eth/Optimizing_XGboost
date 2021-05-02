import numpy as np
from conf import LAMBDA


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def g(y, p):
    return p - y


def h(y, p):
    return p * (1 - p)


### FIZZA #####
class Tree:
    def __init__(self, X, Y):
        self.d = X.shape[1]  ## NUM_FEATURES
        self.root = {'instance_X': X, 'instance_Y': Y, 'prediction_val': np.mean(Y), 'depth': 1}
        self.potential_split_nodes = [self.root]
        self.depth = 1

    def get_potential_split_node(self):
        if len(self.potential_split_nodes) is 0:
            return None
        node = self.potential_split_nodes[0]
        self.potential_split_nodes = self.potential_split_nodes[1:]
        return node

    def split(self, node, k, val, pred):
        d = node['depth']
        node['split_val'] = val
        node['split_feature'] = k
        instance_X = node['instance_X']
        instance_Y = node['instance_Y']
        left_index = np.argwhere(instance_X[:, k] <= val)[:, 0]
        right_index = np.argwhere(instance_X[:, k] > val)[:, 0]
        left_instance_X = instance_X[left_index]
        left_instance_Y = instance_Y[left_index]
        right_instance_X = instance_X[right_index]
        right_instance_Y = instance_Y[right_index]
        left_value = -np.sum([g(y, pred) for y in left_instance_Y]) / (
                np.sum([h(y, pred) for y in left_instance_Y]) + LAMBDA)
        right_value = -np.sum([g(y, pred) for y in right_instance_Y]) / (
                np.sum([h(y, pred) for y in right_instance_Y]) + LAMBDA)
        node['left'] = {'instance_X': left_instance_X, 'instance_Y': left_instance_Y, 'prediction_val': left_value,
                        'depth': d + 1}
        node['right'] = {'instance_X': right_instance_X, 'instance_Y': right_instance_Y, 'prediction_val': right_value,
                         'depth': d + 1}
        self.potential_split_nodes.append(node['left'])
        self.potential_split_nodes.append(node['right'])
        self.depth = max(self.depth, node['depth'] + 1)

    def print_tree(self, node=None, label_map=None):
        if node is None:
            node = self.root
        if 'left' in node:
            print('\t'.join([''] * node['depth']), 'feature: %s, val: %f, num_instances: %d' %
                  (label_map[node['split_feature']], node['split_val'], len(node['instance_Y'])))
            self.print_tree(node=node['left'], label_map=label_map)
            self.print_tree(node=node['right'], label_map=label_map)
        else:
            prob = sigmoid(node['prediction_val'])
            print('\t'.join([''] * node['depth']), 'prob: ', prob, ', num_instances:', len(node['instance_Y']))
