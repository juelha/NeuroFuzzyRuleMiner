# basics
from numpy import dtype, float32
import tensorflow as tf
import numpy as np

# custom
from model_pkg.Trainer import Trainer
#from neurofuzzy_pkg.fuzzyLayers import MF_gaussian_prime_a
from neurofuzzy_pkg import utils
from neurofuzzy_pkg.utils.MFs import MF_gaussian,MF_gaussian_prime_a, MF_gaussian_prime_b
from neurofuzzy_pkg.utils.MFs import MF_tri, MF_tri_prime_a, MF_tri_prime_b
from neurofuzzy_pkg.utils.math_funcs import coefficient
import neurofuzzy_pkg.utils.MFs as MFs


from tqdm import tqdm 
#from neurofuzzy_pkg.fuzzyLayers.RuleConsequentLayer import RuleConsequentLayer
   

class MyArcTrainer(Trainer):   
    """
    The Trainer() Class is:
    - inherited and 
    - train_step() is overwritten for the tuning of the MFs parameters
    ___________________________________________________________
    """

    def __init__(self, n_epochs, learning_rate):#, ite = 10):# loss_func=None, optimizer_func=Adam):
        """Initializing neurofuzzyTrainer by inheriting from Trainer
        """
        super().__init__(Trainer(n_epochs=n_epochs))#, ite)#, iters, learning_rate )
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.feature_ranges = None
      #  self.df_name = None
       # self.feature_ranges = None 


    def __call__(self, train_ds, test_ds, validation_ds):
        """Running the training loop and saving the MFs before and after 
        
        Args:
            train_ds (PrefetchDataset): dataset for training
            test_ds (PrefetchDataset): dataset for testing
        """
        # train
        self.training_loop(train_ds, test_ds, validation_ds)
        

        



    def training_loop(self, train_ds_og, test_ds_og, validation_ds_og):
        """Training of the model
        Args: 
            train_ds (PrefetchDataset): dataset for training
            test_ds (PrefetchDataset): dataset for testing
        """
        # picking random batch from dataset
        test_ds = self.pick_batch(test_ds_og)
        train_ds =  self.pick_batch(train_ds_og)
        validation_ds = self.pick_batch(validation_ds_og)
        # test_ds =  test_ds_og
        # train_ds = train_ds_og
        # validation_ds = validation_ds_og


        # run model on test_ds to keep track of progress during training
        test_loss, test_accuracy = self.test(test_ds)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)

        # same thing for train_ds
        train_loss, _ = self.test(train_ds)
        self.train_losses.append(train_loss)

        # same thing for validation ds
        validation_ds_loss, validation_ds_accuracy = self.test(validation_ds)
        self.v_losses.append(validation_ds_loss)
        self.v_accuracies.append(validation_ds_accuracy)
       
        # training loop until self.iters 
        for epoch in range(self.n_epochs):
            print(f'Epoch: {str(epoch)} starting with \n \
            test accuracy {self.test_accuracies[-1]} \n \
            test loss {self.test_losses[-1]} \n \
            train loss {self.train_losses[-1]} \n \
            validation accuracy {self.v_accuracies[-1]} \n \
            validation loss {self.v_losses[-1]} ')


            # shuffle
            # train_ds = tf.random.shuffle(train_ds)
            # test_ds = tf.random.shuffle(test_ds)
            # validation_ds = tf.random.shuffle(validation_ds)

            # in each epoch, pick a random batch
            test_ds =  self.pick_batch(test_ds_og)
            train_ds =  self.pick_batch(train_ds_og)
            validation_ds = self.pick_batch(validation_ds_og)

            # train and keep track
            epoch_loss_agg = []
            for input,target in train_ds:
                train_loss = self.train_step(input, target)
                epoch_loss_agg.append(train_loss)

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
            
            for inputs, target in (zip(tqdm(inputs_batch, desc='testing'), targets_batch)):
                
                # forward pass to get prediction
                prediction = self.arc(inputs)

                target = np.resize(target, prediction.shape)
                ones = np.ones(shape=prediction.shape)

                # get loss
              #  print(prediction)
              #  print(target)
                sample_test_loss = self.error_function(prediction, target)
               # print("sample_test_loss",sample_test_loss)
                # get accuracy
                sample_test_accuracy =  target == np.round(prediction, 0)
              #  sample_test_accuracy = ones - self.error_function(prediction, target)
                sample_test_accuracy = np.mean(sample_test_accuracy)
                test_loss_aggregator.append(sample_test_loss)
#                test_loss_aggregator.append(sample_test_loss.numpy())
                test_accuracy_aggregator.append(sample_test_accuracy)

        # return averages per batch
        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
        return test_loss, test_accuracy



    def train_step(self, inputs_batch, targets_batch):
        """Tuning the parameters of the MFs using Backpropagation
        Args:
            input (tf.Tensor): input sequence of a dataset
            target (tf.Tensor): output sequence of a dataset
            
        Returns: 
            loss (float): loss before the trainig step
        """

        train_loss_agg = []

        ## step 1: calculating gradients for each entry in batch
        # iterating over data entries of a batch
        deltas_avg = None
        assigned = False
        for inputs, targets in (zip(tqdm(inputs_batch, desc='training'), targets_batch)):

            # forward propagation
            prediction =  self.arc(inputs)

            # calculating error in outputlayer
            train_loss_agg.append(self.error_function(prediction, targets))
            errorterm = self.error_function_derived(prediction, targets)
            # print("errorterm\n\n")
            # print(errorterm)
            # print("targets\n\n")
            # print(targets)
            delta = np.array(errorterm)
            #n_rules = 495
          #  print(delta.shape[0])
            delta = np.reshape(delta, ( int(delta.shape[0]),1))
            if assigned == False: 
                deltas_avg = delta
                assigned = True
            else:
                deltas_avg = np.concatenate((deltas_avg, delta), axis=1)


        ## step 2: get averages of all entries
        train_loss_agg = np.array(train_loss_agg)
        train_loss = np.mean(train_loss_agg)
        deltas_avg = np.mean(deltas_avg,axis=1)
        centers_derived = self.calc_mf_derv_center()
        widths_der = self.calc_mf_derv_widths()

        ## step 3: adapt the parameters with average gradients
        self.adapt(self.arc.FuzzificationLayer, deltas_avg, centers_derived, widths_der)
        
        return train_loss


    def error_function(self, prediction, target):
        """Derived error function:  
            error function: tf.reduce_mean(0.5*(prediction - targets)**2)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """

        error_term = []
        target = target[0]
        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
        
            out_row = prediction[cidx] # in order to slice [:,idx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:

                
                    error =  0.5*(target[idx] - out_row[idx])**2
                    
                    error_term.append(error)
                
        return error_term



    def error_function_derived(self, prediction, target):
        """Derived error function:  
            derived error function: (prediction - targets)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """
        error_term = []
        target = target[0]
        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
        
            out_row = prediction[cidx] # in order to slice [:,idx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:

                
                    error =  -1*(target[idx] - out_row[idx])
                    
                    error_term.append(error)
                
        return error_term



    def calc_mf_derv_center(self):
        calc = MF_gaussian_prime_a(self.arc.FuzzificationLayer.inputs, self.arc.FuzzificationLayer.centers, self.arc.FuzzificationLayer.widths)
        return calc
 
    def calc_mf_derv_widths(self):
        calc = MF_gaussian_prime_b(self.arc.FuzzificationLayer.inputs, self.arc.FuzzificationLayer.centers, self.arc.FuzzificationLayer.widths)
        return calc




    def adapt(self, layer, error, centers_prime, widths_prime):
        
        

        # get those directly from antecedent layer with inputs attribute
        mus = self.arc.RuleAntecedentLayer.inputs
        
        # reshape error to match each mu 
        error = np.reshape(error, mus[0].shape)
        delta = [mu * error for mu in mus]
        delta.reverse() # the other mu for each x
        
        # x = np.array_split(centers_prime, range(3, len(centers_prime), 3))
        # centers_grided = np.meshgrid(x[0], x[1]) 
        # delta_centers = [d * c for d,c in zip(delta,centers_grided)]
        # c_delta_x1 = np.sum(delta_centers[0], axis=0)
        # c_delta_x2 = np.sum(delta_centers[1], axis=1)
        # deltas_centers = np.concatenate((c_delta_x1, c_delta_x2))
        # self.arc.FuzzificationLayer.centers =  self.arc.FuzzificationLayer.centers   + deltas_centers  * self.learning_rate


        # x = np.array_split(widths_prime, range(3, len(centers_prime), 3))
        # centers_grided = np.meshgrid(x[0], x[1]) 
        # delta_centers = [d * c for d,c in zip(delta,centers_grided)]
        # c_delta_x1 = np.sum(delta_centers[0], axis=0)
        # c_delta_x2 = np.sum(delta_centers[1], axis=1)
        # deltas_centers = np.concatenate((c_delta_x1, c_delta_x2))
        # self.arc.FuzzificationLayer.widths =  self.arc.FuzzificationLayer.widths   + deltas_centers  * self.learning_rate
        self.adapt_parameter('centers', layer, delta, centers_prime)
        self.adapt_parameter('widths', layer, delta, widths_prime)

        return 0
    
    def adapt_parameter(self, param, layer, delta, para_prime):
        
        # imitate the whole meshgrid process like in antecedent layer
        x = np.array_split(para_prime, range(3, len(para_prime), 3))
        para_gridded = np.meshgrid(x[0], x[1]) 
        delta = [d * p for d,p in zip(delta, para_gridded)]
        delta_x1 = np.sum(delta[0], axis=0)
        delta_x2 = np.sum(delta[1], axis=1)
        delta = np.concatenate((delta_x1, delta_x2))

        # self.fuzzi ...
        param_to_tune = getattr(layer, param)
        param_to_tune -= np.multiply(delta, self.learning_rate)
        setattr(layer, param, param_to_tune)