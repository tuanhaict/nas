import numpy as np

from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

class NetworkManager:

    def __init__(self, dataset, epochs=5, child_batchsize=128, acc_beta=0.8, clip_rewards=0.0):

        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0

    def get_rewards(self, model_fn, actions):
        with tf.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)

            model = model_fn(actions) 
            model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

            X_train, y_train, X_val, y_val = self.dataset


            model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs,
                      verbose=1, validation_data=(X_val, y_val),
                      callbacks=[ModelCheckpoint('weights/temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True)])

            model.load_weights('weights/temp_network.h5')


            loss, acc = model.evaluate(X_val, y_val, batch_size=self.batchsize)

            reward = (acc - self.moving_acc)

            if self.clip_rewards:
                reward = np.clip(reward, -0.05, 0.05)

            if self.beta > 0.0 and self.beta < 1.0:
                self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
                self.moving_acc = self.moving_acc / (1 - self.beta_bias)
                self.beta_bias = 0

                reward = np.clip(reward, -0.1, 0.1)

            print()
            print("Manager: EWA Accuracy = ", self.moving_acc)


        network_sess.close()

        return reward, acc