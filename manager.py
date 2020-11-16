import numpy as np

from keras.models import Model
# from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
import tensorflow.compat.v1 as tf
import keras


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, dataset, epochs=5, child_batchsize=128, acc_beta=0.8, clip_rewards=0.0):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            child_batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_rewards: float - to clip rewards in [-range, range] to prevent
                large weight updates. Use when training is highly unstable.
        '''
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0

    def get_rewards(self, model_fn, actions):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.
            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given actions
        '''
        with tf.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)

            # generate a submodel given predicted actions
            model = model_fn(actions)  # type: Model
            model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

            # unpack the dataset
            X_train, y_train, X_val, y_val = self.dataset
            print(X_train.shape)
            # train the model using Keras methods
            model.fit(X_train, y_train, batch_size=self.batchsize, epochs=self.epochs,
                      verbose=1, validation_data=(X_val, y_val),
                      callbacks=[ModelCheckpoint('weights/temp_network.h5',
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True)])

            # load best performance epoch in this training session
            model.load_weights('weights/temp_network.h5')

            # evaluate the model
            loss, acc = model.evaluate(X_val, y_val, batch_size=self.batchsize)

            # compute the reward
            reward = (acc - self.moving_acc)

            # if rewards are clipped, clip them in the range -0.05 to 0.05
            if self.clip_rewards:
                reward = np.clip(reward, -0.05, 0.05)

            # update moving accuracy with bias correction for 1st update
            if self.beta > 0.0 and self.beta < 1.0:
                self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
                self.moving_acc = self.moving_acc / (1 - self.beta_bias)
                self.beta_bias = 0

                reward = np.clip(reward, -0.1, 0.1)

            print()
            print("Manager: EWA Accuracy = ", self.moving_acc)

        # clean up resources and GPU memory
        network_sess.close()

        return reward, acc




    def get_rewards_wt(self, model_fn, actions):

        batch_size=16

        # generate a submodel given predicted actions
        model = model_fn(actions)  # type: Model
        #model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

        # Instantiate an optimizer.
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        # Instantiate a loss function.
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


        X_train, y_train, X_val, y_val = self.dataset

#         train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#         train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        epochs = 2
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                image=tf.Variable(X_train[0:1])
                tape.watch(image)
                logits = model(image, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                print(np.argmax(y_train[0:4],axis=1),logits.shape)
                loss_value = loss_fn(np.argmax(y_train[0:1],axis=1), logits)
                print(loss_value)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, image)
            print(grads)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            #optimizer.apply_gradients(zip(grads, model.trainable_weights))

#             # Log every 200 batches.
#             if step % 200 == 0:
#                 print(
#                     "Training loss (for one batch) at step %d: %.4f"
#                     % (step, float(loss_value))
#                 )
#                 print("Seen so far: %s samples" % ((step + 1) * 64))



        # gradients = K.gradients(model.output, model.input)

        # # Wrap the input tensor and the gradient tensor in a callable function
        # f = K.function([model.input], gradients)

        # # Random input image
        # x = np.random.rand(1, 100,100,3)

        reward=9
        acc=9
        return reward, acc