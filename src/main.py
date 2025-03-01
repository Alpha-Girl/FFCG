import Parameters
import warnings
import random
import test
from DQN import *
from read_data import *
import numpy as np
# ignore warnings
warnings.filterwarnings("ignore")

seed_ = Parameters.seed
np.random.seed(seed_)
random.seed(seed_)

FFCG = DQNAgent(env=None, capacity=Parameters.capacity, batch_size=Parameters.batch_size, epochs=Parameters.epochs,
                embedding_size=Parameters.embedding_size, cons_num_features=Parameters.cons_num_features,
                vars_num_features=Parameters.vars_num_features, learning_rate=Parameters.lr)

DATA = test.test_ffcg(FFCG, Parameters.model_index)
