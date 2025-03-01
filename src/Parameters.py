# We use this file to store the parameters of the model and the RL algorithm
# seed
seed = 2

DATA_PATH = 'data/'
RESULT_PATH = 'result/'
MODEL_PATH = 'model/'
# parameters about neural network
lr = 0.001
batch_size = 32
capacity = 20000
epochs = 5
embedding_size = 32
cons_num_features = 2
vars_num_features = 9

# parameters of RL algorithm
gamma = 0.9
epsilon = 0.05
min_epsilon = 1e-2
min_epsilon_ratio = 0.8
decaying_epsilon = True
step_penalty = 1
alpha_obj_weight = 1000  # for the reward function
beta = 0.2  # for the reward function
action_pool_size = 10  # solution pool
max_episode_num = 400
capacity = 2000
threshold = 0.5

# parameter index
model_index = 1
