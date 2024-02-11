# basics 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# specifics
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
import os

from tqdm import tqdm 

class Trainer():
    """Training the model while keeping track of the accuracies and losses using SGD.
    ___________________________________________________________
    """

    def __init__(self, n_epochs,  arc=None, learning_rate=None, optimizer_func=Adam):
        """Initializes Trainer()-Object
        
        Args:
            arc (MLP): architecture of the NN
            n_epochs (int): iterations of training loop
            learning_rate (float): learning rate for training
            optimizer_func (tf.keras.optimizers)
        
        Attributes:
            loss_func
            test_accuracies (list(float)): keeping track of test accuracies during training
            test_losses (list(float)): keeping track of test losses during training
            train_losses (list(float)): keeping track of train losses during training
        """

        self.arc = arc

        # hyperparamers
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.loss_func = self.error_function
        self.optimizer_func = optimizer_func(learning_rate)

        # get params
        self.params = [
            self.n_epochs, self.learning_rate, self.loss_func, self.optimizer_func
        ]
        
        # for visualization of training
        # for visualization of training
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_losses = []
        self.train_losses = []

        self.v_accuracies = []
        self.v_losses = []


    def __call__(self, train_ds,  test_ds, validation_ds):
        """Calling the trainer calls training_loop()
        Args: 
            train_ds (PrefetchDataset): dataset for training
            test_ds (PrefetchDataset): dataset for testing
        Note: 
            Implemented for cleaner code and the use in the inherited 
            class neurofuzzyTrainer() 
        """
        self.training_loop(train_ds,  test_ds, validation_ds)
        self.visualize_training(self.df_name, self.arc.Name)




    def training_loop(self, train_ds_og, test_ds_og, validation_ds_og):
        """Training of the model
        Args: 
            train_ds (PrefetchDataset): dataset for training
            test_ds (PrefetchDataset): dataset for testing
        """
        # picking random batch from dataset
        test_ds =  test_ds_og
        train_ds = train_ds_og
        validation_ds = validation_ds_og
        # test_ds = self.pick_batch(test_ds_og)
        # train_ds =  self.pick_batch(train_ds_og)
        # validation_ds = self.pick_batch(validation_ds_og)

        # run model on test_ds to keep track of progress during training
        test_loss, test_accuracy = self.test(test_ds)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)

        # same thing for train_ds
        train_loss, train_acc = self.test(train_ds)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

        # same thing for validation ds
        # validation_ds_loss, validation_ds_accuracy = self.test(validation_ds)
        # self.v_losses.append(validation_ds_loss)
        # self.v_accuracies.append(validation_ds_accuracy)
       
        # training loop until self.iters 
        for epoch in range(self.n_epochs):
            print(f'Epoch: {str(epoch)} starting with \n \
            test accuracy {self.test_accuracies[-1]} \n \
            test loss {self.test_losses[-1]} \n \
            train loss {self.train_losses[-1]}')

            # in each epoch, pick a random batch
            # test_ds =  self.pick_batch(test_ds_og)
            # train_ds =  self.pick_batch(train_ds_og)

            # train and keep track
            epoch_loss_agg = []
            for input,target in train_ds:
                train_loss, train_acc = self.train_step(input,target)
                self.train_accuracies.append(train_acc)
                self.train_losses.append(train_loss)

            #track training loss
            self.train_losses.append(tf.reduce_mean(epoch_loss_agg))

            #testing, so we can track accuracy and test loss
            test_loss, test_accuracy = self.test(test_ds)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)

            # same thing for validation ds
            validation_ds_loss, validation_ds_accuracy = self.test(validation_ds)
            self.v_losses.append(validation_ds_loss)
            self.v_accuracies.append(validation_ds_accuracy)

        print("Training Loop completed")
        return 0


    def test(self, ds):
        """Forward pass of test_data
        Args:
            test_ds (TensorSliceDataset): batch of testing dataset
        Returns:
            test_loss (float): average loss from output to target
            test_accuracy (float): average accuracy of output
        """

        test_accuracy_aggregator = []
        test_loss_aggregator = []

        # iterate over batch
        for (inputs_batch, targets_batch) in ds:
            
            for input, target in (zip(tqdm(inputs_batch, desc='testing'), targets_batch)):
                
              #  print("INPUT", input)
                # forward pass to get prediction
                prediction = self.arc(input)
              #  print("Pred", prediction)

               # print("tar", target)

                # get loss
                sample_test_loss = self.loss_func(prediction, target)
                # get accuracy
                sample_test_accuracy =  target == np.round(prediction, 0)

              #  print("sample_test_accuracy", sample_test_accuracy)
                sample_test_accuracy = np.mean(sample_test_accuracy)

               # print("sample_test_accuracy np mean" , sample_test_accuracy)

                test_loss_aggregator.append(sample_test_loss)
#                test_loss_aggregator.append(sample_test_loss.numpy())


               # print("np.mean(sample_test_accuracy", np.mean(sample_test_accuracy))
                test_accuracy_aggregator.append(np.mean(sample_test_accuracy))  

        # return averages per batch
        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
        return test_loss, test_accuracy


    def train_step(self, inputs_batch, targets_batch):
        """Implements train step for batch of datasamples
        Args:
            input (tf.Tensor): input sequence of a batch of dataset
            target (tf.Tensor): output sequence of a batch of dataset
        Returns:
            loss (float): average loss before after train step
        """
        # list for losses per batch
        losses_aggregator = []
        # iterate over the batch
        for input, target in zip(inputs_batch, targets_batch):
            with tf.GradientTape() as tape:
                # forward pass to get prediction
                prediction = self.arc(input)
                # get loss
                loss = self.loss_func(prediction, target)
                losses_aggregator.append(loss)
                # get gradients
                gradients = tape.gradient(loss, self.arc.trainable_variables)

            # adapt the trainable variables with gradients 
            self.optimizer_func.apply_gradients(zip(gradients, self.arc.trainable_variables))  

        # return average loss
        loss = tf.reduce_mean(losses_aggregator)
        return loss


    def error_function(self, prediction, targets):
        """Derived error function:  
            error function: tf.reduce_mean(0.5*(prediction - targets)**2)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """

        #error_term = tf.reduce_mean(0.5*(prediction - targets)**2)

       # print("pred", prediction)
       # print("tar", targets)
        error_term = 0.5*(prediction - targets)**2
        return error_term


    def error_function_derived(self, prediction, targets):
        """Derived error function:  
            derived error function: (prediction - targets)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """
        error_term = (prediction - targets)
        return error_term


    def visualize_training(self, df_name, type_model):
        """ Visualize accuracy and loss for training and test data.
        Args:
            type_model (str): type of model that has been trained
        """
        plt.figure()
        # accuracies
        line1, = plt.plot(self.test_accuracies)
        line2, = plt.plot(self.train_accuracies)
       # line3, = plt.plot(self.val_accuracies)
        # losses
        line4, = plt.plot(self.test_losses)
        line5, = plt.plot(self.train_losses)
       # line5, = plt.plot(self.v_losses)
        plt.xlabel("Training steps")
        plt.ylabel("Loss/Accuracy")
        plt.legend((line1, line2, line4, line5),
        ("test accuracy", "training accuracy", "test losses", "training losses"))
        plt.title(f'{type_model}')
        plt.figure
       
        # get save path 
        file_name = 'Performance' + str(type_model) + '.png'
        save_path = os.path.dirname(__file__) +  f'/../results/{df_name}/figures'
        completeName = os.path.join(save_path, file_name)
        plt.savefig(completeName)
        plt.clf()
    def summary(self):
        column_names = ['n_epochs', 'learning_rate', 'loss_func', 'optimizer_func']
        d = self.params
        df = pd.DataFrame(d, column_names)
        return df