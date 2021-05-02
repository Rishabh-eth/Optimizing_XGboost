import numpy as np
import pandas as pd
import config as conf
import math
from testwriter import TestWriter
from scipy.special import expit

# Basic functions
def sigmoid(x):
    return expit(x)
    #return 1 / (1 + np.exp(-x))


def sigmoid_1(x):
    return 1 / (1 + math.exp(-x))


def log_odds(p):
    return np.log(p / (1-p))


class Node:

    @staticmethod
    def output_value(g, h):
        return - g.sum() / (h.sum() + conf.REGULARIZER)

    def __init__(self, x, g, h, depth):
        self.depth = depth # Distance from root node
        self.left = None # Left child
        self.right = None # Right child

        self.split_feature = None # Feature (column) we used to split
        self.split_val = None  # All values smaller than split val in left split
        self.output = None # Output value

        if self.depth < conf.MAX_TREE_DEPTH:
            # Find best split
            max_gain, best_k, best_j_idx = self.exact_greedy(x, g, h)

            if max_gain - conf.MIN_SPLIT_LOSS >= 0:
                self.split_feature = best_k
                order = x[:, self.split_feature].argsort()

                self.split_val = x[order[best_j_idx], best_k]

                # Add left and right child node
                x_left_indices = order[:best_j_idx+1] # i:j, j is exclusive -> j-1
                x_right_indices = order[best_j_idx+1:]

                self.left = Node(x[x_left_indices,:], g[x_left_indices], h[x_left_indices], depth+1)
                self.right = Node(x[x_right_indices,:], g[x_right_indices], h[x_right_indices], depth+1)
            else:
                self.output = self.output_value(g, h)
        else:
            self.output = self.output_value(g, h)

    def exact_greedy(self, x, g, h):
        gain = 0
        best_k = -1 # Feature we used for the split
        best_j_idx = 0 # Idx of val j in the sorted x column, all values up to j_idx in left split, from j_idx + 1 in right split
        best_val = -1

        G = g.sum()
        H = h.sum()
        root_score = G ** 2 / (H + conf.REGULARIZER)

        n_rows = x.shape[0]
        n_features = x.shape[1]
        for k in range(n_features):
            Gl = 0
            Hl = 0
            sort_positions = x[:,k].argsort()
            n_feature_vals = len(sort_positions)
            for idx, j in enumerate(sort_positions):
                Gl = Gl + g[j]
                Hl = Hl + h[j]
                Gr = G - Gl
                Hr = H - Hl
                new_gain = (Gl**2 / (Hl+conf.REGULARIZER)) + (Gr**2 / (Hr+conf.REGULARIZER)) - root_score
                if new_gain > gain and (idx+1 != n_rows and x[j,k] != x[sort_positions[idx+1],k]): # Only check if improves if added all same features to the gain
                    best_val = x[j,k]
                    gain = new_gain
                    best_k = k
                    best_j_idx = idx

        if conf.WRITE_TEST and 0 == 1:
            feature = best_k
            TestWriter.add_splitfinding("split_exact_greedy", x, g, h, feature, best_val, gain)

        return (gain, best_k, best_j_idx)

    def predict(self, x): # Single x vec!
        if self.output is not None: # We are in a leaf
            return self.output
        else:
            if x[self.split_feature] <= self.split_val:
                return self.left.predict(x)
            else:
                return self.right.predict(x)

    def print(self):
        #features = ['I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I11', 'I12']

        if self.left is None:
            print('{} prediction: {}'.format('\t' * self.depth, self.output))
        else:
            print('{} feature: {}, val: {}'.format('\t' * self.depth, self.split_feature, self.split_val))
            self.left.print()
            self.right.print()


class Tree:

    def __init__(self, x, g, h):
        self.root = Node(x, g, h, 0)

    def predict(self, x):
        # Run prediction on each row
        return np.array([self.root.predict(el) for el in x])

    def print(self):
        self.root.print()

class XGBoostClassifier:

    def __init__(self):
        self.trees = None

    @staticmethod
    def gradient_log_loss(p_pred, y):
        return p_pred - y  # pi - yi

    @staticmethod
    def hessian_log_loss(p_pred):
        return p_pred * (1 - p_pred)  # pi (1 - pi)

    def fit(self, x, y):
        '''
        Fits trees to the residuals in order to do predictions
        :return:
        '''
        self.trees = []

        p = np.empty(y.shape[0])
        p.fill(np.average(y))

        for i in range(conf.N_TREES):
            g = self.gradient_log_loss(p, y)
            h = self.hessian_log_loss(p)
            tree = Tree(x, g, h)
            p += conf.LEARNING_RATE * sigmoid(tree.predict(x)) # Convert log odds prediction into prob
            self.trees.append(tree)

    def compare_print(self, x, y):
        self.trees = []

        p = np.empty(y.shape[0])
        p.fill(np.average(y))
        g = self.gradient_log_loss(p, y)
        h = self.hessian_log_loss(p)
        tree = Tree(x, g, h)
        tree.print()

    def predict(self, x):
        prediction = 0

        p = np.empty(x.shape[0])
        p.fill(0.5)

        prediction += log_odds(p)

        for tree in self.trees:
            prediction += conf.LEARNING_RATE * tree.predict(x)

        return sigmoid(prediction)


def main():
    data = pd.read_csv(conf.INPUT)
    x = data[['I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I11', 'I12']].to_numpy()
    y = data['Label'].to_numpy()

    classifier = XGBoostClassifier()
    classifier.fit(x, y)

    classifier.trees[0].print()

    # if conf.WRITE_TEST:
    #    TestWriter.add_tree("tree_exact_greedy", tree, data_X, data_Y, pred)

    if conf.WRITE_TEST:
        TestWriter.add_trees("tree_exact_greedy", classifier.trees, x, y)

        TestWriter.write_all('./')



    #classifier.compare_print(x, y)

    # Don't do this! :-)
    #predicted = classifier.predict(x)
    #y_predict = np.where(predicted > 0.5, 1, 0)
    #print((y == y_predict).astype(int).sum() / y.shape[0])


    #data_test = pd.read_csv(conf.INPUT_TEST)
    #x_test = data_test[['I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I11', 'I12']].to_numpy()
    #y_test = data_test['Label'].to_numpy()
    #predicted = classifier.predict(x_test)
    #y_predict = np.where(predicted > 0.5, 1, 0)
    #print((y_test == y_predict).astype(int).sum() / y_test.shape[0])


if __name__ == "__main__":
    main()
