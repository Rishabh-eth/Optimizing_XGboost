from conf import *
from tree import Tree, g, h
import numpy as np
import pandas as pd
from testwriter import TestWriter

label_to_feature = ['I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I11', 'I12']


def split_finding_algorithm(node, n_features, pred):
    instance_X = node['instance_X']
    instance_Y = node['instance_Y']
    G = sum([g(y, pred) for y in instance_Y])
    H = sum([h(y, pred) for y in instance_Y])

    best_score = 0
    best_split = None
    for k in range(0, n_features):
        GL = 0
        HL = 0
        prev_score = 0
        prev_val = -1
        sorted_data = sorted(enumerate(instance_X), key=lambda row: row[1][k])
        for j, x in sorted_data:
            GL += g(instance_Y[j], pred)
            HL += h(instance_Y[j], pred)
            GR = G - GL
            HR = H - HL
            this_score = ((GL ** 2) / (HL + LAMBDA)) + ((GR ** 2) / (HR + LAMBDA)) - ((G ** 2) / (H + LAMBDA))
            if prev_score > best_score and x[k] != prev_val:
                best_score = prev_score
                best_split = (k, prev_val)
            prev_score = this_score
            prev_val = x[k]

    if WRITE_TEST:
        TestWriter.add_splitfinding("split_exact_greedy", node, pred, best_split, best_score)

    return best_split


def create_tree(data_X, data_Y, pred):
    tree = Tree(data_X, data_Y)
    while True:
        node = tree.get_potential_split_node()
        if node is None:
            break
        if node['depth'] >= MAX_DEPTH:
            continue
        best_split = split_finding_algorithm(node, tree.d, pred)
        if best_split is not None:
            k, val = best_split
            tree.split(node, k, val, pred)
    return tree


if __name__ == "__main__":
    data = pd.read_csv(SMALL_DATA_PATH)
    data_X = data[label_to_feature].to_numpy()
    data_Y = data['Label'].to_numpy()
    pred = np.mean(data_Y)
    tree = create_tree(data_X, data_Y, pred)
    tree.print_tree(label_map=label_to_feature)

    if WRITE_TEST:
        TestWriter.add_tree("tree_exact_greedy", tree, data_X, data_Y, pred)

    if WRITE_TEST:
        TestWriter.write_all('./')
