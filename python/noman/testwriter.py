import json
import os
import numpy as np

import conf as conf
from tree import Tree, g, h


def tree_to_json(node):
    data = {
        'depth': node['depth'],
        'num_instances': len(node['instance_Y']),
    }

    if 'left' in node:
        data['split_feature'] = node['split_feature']
        data['split_val'] = float(node['split_val'])
        data['left'] = tree_to_json(node['left'])
        data['right'] = tree_to_json(node['right'])
    else:
        data['prediction_val'] = float(node['prediction_val'])

    return data


def generate_jsonmatrix(X, Y, pred):
    grad = [g(y, pred) for y in Y]
    hess = [h(y, pred) for y in Y]
    mat_2d = np.hstack((X, np.array(grad)[:, None], np.array(hess)[:, None]))
    return mat_2d.flatten().tolist()


class TestWriter:
    test_data = {}
    exact_greedy_data = []

    @staticmethod
    def add(test, data):
        data["lambda"] = conf.LAMBDA
        data["maxdepth"] = conf.MAX_DEPTH

        if test not in TestWriter.test_data:
            TestWriter.test_data[test] = []

        TestWriter.test_data[test].append(data)

    @staticmethod
    def add_tree(type, tree, data_X, data_Y, pred):
        data = {
            'tree': tree_to_json(tree.root),
            'matrix': generate_jsonmatrix(data_X, data_Y, pred),
            'rows': data_X.shape[0],
            'cols': data_X.shape[1] + 2,
            'Y': data_Y.flatten().tolist(),
            'ini_pred': float(pred)
        }

        TestWriter.add(type, data)

    @staticmethod
    def add_splitfinding(type, node, pred, best_split, best_score):
        data = {
            'matrix': generate_jsonmatrix(node['instance_X'], node['instance_Y'], pred),
            'rows': node['instance_X'].shape[0],
            'cols': node['instance_X'].shape[1] + 2,
            'value': float(best_split[1]),
            'feature': best_split[0],
            'gain': float(best_score)
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