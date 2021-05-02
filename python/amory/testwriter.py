import json
import os
import numpy as np

import config as conf


def tree_to_json(node):
    data = {
        'depth': node.depth + 1,
    }

    # TODO: removed num instances

    if node.left is None:
        data['prediction_val'] = float(node.output)
    else:
        data['split_feature'] = node.split_feature
        data['split_val'] = float(node.split_val)
        data['left'] = tree_to_json(node.left)
        data['right'] = tree_to_json(node.right)

    return data


class TestWriter:
    test_data = {}
    exact_greedy_data = []

    @staticmethod
    def add(test, data):
        if test not in TestWriter.test_data:
            TestWriter.test_data[test] = []

        TestWriter.test_data[test].append(data)

    @staticmethod
    def add_trees(type, trees, x, y):

        data = {
            'trees': [tree_to_json(tree.root) for tree in trees],
            'n_trees': conf.N_TREES,
            'x': x.flatten().tolist(),
            'rows': x.shape[0],
            'cols': x.shape[1],
            'Y': y.flatten().tolist(),
            'max_depth': conf.MAX_TREE_DEPTH + 1,
            'lambda': conf.REGULARIZER,
            'learning_rate': conf.LEARNING_RATE,
            'gamma': conf.MIN_SPLIT_LOSS,
            'min_instances': 0
        }

        TestWriter.add(type, data)

    @staticmethod
    def add_splitfinding(type, x, g, h, feature, val, gain):
        data = {
            'lambda': conf.REGULARIZER,
            'x': x.flatten().tolist(),
            'rows': x.shape[0],
            'cols': x.shape[1],
            'gh': np.hstack((g[:, None], h[:, None])).flatten().tolist(),
            'value': float(val),
            'feature': feature,
            'gain': float(gain),
        }

        TestWriter.add(type, data)

    @staticmethod
    def write(test, path):
        with open(os.path.join(path, '{}.json'.format(test)), 'w') as outfile:
            json.dump(TestWriter.test_data[test], outfile)

    @staticmethod
    def write_all(path):
        for key in TestWriter.test_data:
            TestWriter.write(key, path)
