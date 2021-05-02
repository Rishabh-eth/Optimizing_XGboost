# Input / output
INPUT = '../../data/ten_thousand.csv'
INPUT_TEST = '../../data/test.csv'
WRITE_TEST = True

# XGBoost parameters
N_TREES = 1
MAX_TREE_DEPTH = 5

MIN_SPLIT_LOSS = 5 # gamma, tree complexity parameter # TODO: default 5
LEARNING_RATE = 0.3 # epsilon, for pred TODO: default 0.3
REGULARIZER = 10 # lambda, regularization parameter # TODO: default
