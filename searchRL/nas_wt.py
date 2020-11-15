import numpy as np
import csv
import torch
from tqdm import trange
import time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# import tensorflow as tf
# from keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1.keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.datasets import cifar10
from keras.utils import to_categorical

from controller import Controller, StateSpace
#from manager import NetworkManager
#from model import Net
from model_torch import model_fn
from manager_torch import NetworkManager

import neptune
from api_config import project_name,api_token
neptune.init(project_name,api_token=api_token)
neptune.set_project(project_name)

# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = 4  # number of layers of the state space
MAX_TRIALS = 250  # maximum number of models generated

MAX_EPOCHS = 10  # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training
USE_TRAIN=False

params = {
    'logging':'neptune',
    'num_layers':NUM_LAYERS,
    'max_trials':MAX_TRIALS,
    'max_epochs':MAX_EPOCHS,
    'child_batchsize':CHILD_BATCHSIZE,
    'exploration':EXPLORATION,
    'regularization':REGULARIZATION,
    'controller_cells':CONTROLLER_CELLS,
    'embedding_dim':EMBEDDING_DIM,
    'accuracy_beta':ACCURACY_BETA,
    'clip_rewards':CLIP_REWARDS,
    'restore_controller':RESTORE_CONTROLLER,
    'model_name':'test',
    'use_train': USE_TRAIN,
    'batch_size': 32,
    'n_runs':10,
    'n_samples':100
}

model_name = params['model_name']
name_one = model_name

if(params['logging']=='neptune'):
    neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
    neptune.append_tag(model_name)
    neptune.append_tag('storing_best')

device = 'cuda'

# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='kernel', values=[1, 3])
state_space.add_state(name='filters', values=[16, 32, 64])

# print the state space being searched
state_space.print_state_space()

# prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['batch_size'],
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=params['batch_size'],
                                                 shuffle=False, num_workers=2)


times     = []
chosen    = []
acc       = []
val_acc   = []                      
topscores = []

order_fn = np.nanargmax

def get_batch_jacobian(net, x, target):
        net.zero_grad()
        x.requires_grad_(True)
        y = net(x)
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()

        return jacob, target.detach()

def eval_score(jacob, labels=None):
        corrs = np.corrcoef(jacob)
        v, _  = np.linalg.eig(corrs)
        k = 1e-5
        return -np.sum(np.log(v + k) + 1./(v + k))

runs = trange(params['n_runs'], desc='acc: ')
for N in runs:
    start = time.time()
    states = [state_space.get_random_state_space(NUM_LAYERS) for no in range(params['n_samples'])]
    scores = []

    for arch in states:

        data_iterator = iter(trainloader)
        x,target = next(data_iterator)
        x, target = x.to(device), target.to(device)

        network = model_fn(state_space.parse_state_space_list(arch),1).to(device)

        jacobs, labels= get_batch_jacobian(network, x, target)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

        try:
            s = eval_score(jacobs, labels)
        except Exception as e:
            print(e)
            s = np.nan
        scores.append(s)

    best_arch = states[order_fn(scores)]
    topscores.append(scores[order_fn(scores)])
    chosen.append(state_space.parse_state_space_list(best_arch))

    times.append(time.time()-start)

print(topscores)
print(chosen)

import json
with open("topscores.txt", "w") as fp:
    json.dump(topscores, fp)
with open("chosen.txt", "w") as fp:
    json.dump(chosen, fp)

neptune.log_artifact('topscores.txt')
neptune.log_artifact('chosen.txt')



