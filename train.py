import numpy as np
import csv

import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical

from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn


policy_sess = tf.Session()
K.set_session(policy_sess)

NUM_LAYERS = 4  
MAX_TRIALS = 250  

MAX_EPOCHS = 10  
CHILD_BATCHSIZE = 128  
EXPLORATION = 0.8  
REGULARIZATION = 1e-3 
CONTROLLER_CELLS = 32  
EMBEDDING_DIM = 20  
ACCURACY_BETA = 0.8  
CLIP_REWARDS = 0.0  
RESTORE_CONTROLLER = True


state_space = StateSpace()


state_space.add_state(name='kernel', values=[1, 3])
state_space.add_state(name='filters', values=[16, 32, 64])


state_space.print_state_space()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

dataset = [x_train, y_train, x_test, y_test] 

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():

    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)


manager = NetworkManager(dataset, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA)


state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()


controller.remove_files()


for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state)  


    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))


    reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
    print("Rewards : ", reward, "Accuracy : ", previous_acc)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)


        state = actions
        controller.store_rollout(state, reward)


        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))


        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

print("Total Reward : ", total_reward)