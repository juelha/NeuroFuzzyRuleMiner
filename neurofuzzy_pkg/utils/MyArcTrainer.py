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
        

    def preprocess_target(self, target, prediction):
        """
        ds (tuple) = unzips into featuers, targets
        since one target for nine rules -> resize target 
        """
        target = np.resize(target, prediction.shape) # the only difference to trainer
        return target
    
    
    def test(self, ds):
        """Forward pass of test_data
        Args:
            test_ds (TensorSliceDataset): batch of testing dataset
        Returns:
            test_loss (float): average loss from output to target
            test_accuracy (float): average accuracy of output
        """

        accuracy_aggregator = []
        loss_aggregator = []

        inputs_batch, targets_batch = ds
        
        for input, target in (zip(tqdm(inputs_batch, desc='testing'), targets_batch)):
            prediction = self.arc(input)
            target = np.resize(target, prediction.shape) # the only difference to trainer !!!!!!!!!!!!!!!!!!!!!
            loss = self.loss_func(prediction, target)
            accuracy = self.accuracy_function(prediction, target)
           # accuracy =  target == np.round(prediction, 0)
            #accuracy = np.mean(accuracy)

            loss_aggregator.append(loss)
            accuracy_aggregator.append(np.mean(accuracy))  

        # return averages per batch
        test_loss = np.mean(loss_aggregator)
        test_accuracy =  np.mean(accuracy_aggregator)
        return test_loss, test_accuracy


    def train_step(self, train_batch):
        """Implements train step for batch of datasamples
        Args:
            input (tf.Tensor): input sequence of a batch of dataset
            target (tf.Tensor): output sequence of a batch of dataset
        Returns:
            loss (float): average loss before after train step
        """
        inputs_batch, targets_batch = train_batch

        loss_agg = []
        accuracy_aggregator = []

        ## step 1: calculating gradients for each entry in batch
        # iterating over data entries of a batch
        deltas_avg = None
        assigned = False
        for inputs, targets in (zip(tqdm(inputs_batch, desc='training'), targets_batch)):

            # forward propagation
            prediction =  self.arc(inputs)
            # calculating error in outputlayer
            targets = np.resize(targets, prediction.shape) # the only difference to trainer !!!!!!!!!!!!!!!!!!!!!
            loss_agg.append(self.error_function(prediction, targets))
            # calculating accuracy
            accuracy = self.accuracy_function(prediction,targets)
            #accuracy = np.mean(accuracy)
            accuracy_aggregator.append(accuracy)
            errorterm = self.error_function_derived(prediction, targets)

            self.adapt(self.arc.FuzzificationLayer, errorterm)
            # redo this bit 
            # delta = np.array(errorterm)
            # delta = np.reshape(delta, ( int(delta.shape[0]),1))
            # if assigned == False: 
            #     deltas_avg = delta
            #     assigned = True
            # else:
            #     deltas_avg = np.concatenate((deltas_avg, delta), axis=1)


        ## step 2: get averages of all entries
      #  deltas_avg = np.mean(deltas_avg,axis=1)
        

        ## step 3: adapt the parameters with average gradients
       # self.adapt(self.arc.FuzzificationLayer, deltas_avg)
        
        # return average loss
        loss =  np.mean(loss_agg)
        acc = np.mean(accuracy_aggregator)
        return loss, acc


    def accuracy_function(self, prediction, target):
        
        accs = []
        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
            
            y = prediction[cidx] # in order to slice [:,idx]
            t = target[cidx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:

                    acc = t[idx] == np.round(y[idx], 0)
                    accs.append(acc)
        
        return acc

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
        #target = target[0]
        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
        
            out_row = prediction[cidx] # in order to slice [:,idx]
            tar = target[cidx]
          #  print("tar", at)
            for idx, number in enumerate(classweight):
                if bool(number)==True:

                
                    error =  0.5*(tar[idx] - out_row[idx])**2
                    
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
      #  target = target[0]
        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
        
            out_row = prediction[cidx] # in order to slice [:,idx]
            tar = target[cidx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:

                
                    error =  -1*(tar[idx] - out_row[idx])
                    
                    error_term.append(error)
                
        return error_term



    def calc_mf_derv_center(self):
        calc = MF_gaussian_prime_a(self.arc.FuzzificationLayer.inputs, self.arc.FuzzificationLayer.centers, self.arc.FuzzificationLayer.widths)
        return calc
 
    def calc_mf_derv_widths(self):
        calc = MF_gaussian_prime_b(self.arc.FuzzificationLayer.inputs, self.arc.FuzzificationLayer.centers, self.arc.FuzzificationLayer.widths)
        return calc




    def adapt(self, layer, error):
        

        # get those directly from antecedent layer with inputs attribute
        mus = self.arc.RuleAntecedentLayer.inputs
        
        # reshape error to match each mu 
        error = np.reshape(error, mus[0].shape)
        delta = [mu * error for mu in mus]
        #delta.reverse() # the other mu for each 
        
        centers_prime = self.calc_mf_derv_center()
        widths_prime = self.calc_mf_derv_widths()
        

        self.adapt_parameter('centers', layer, delta, centers_prime)
        self.adapt_parameter('widths', layer, delta, widths_prime)

        return 0
    
    def adapt_parameter(self, param, layer, delta, para_prime):
        """
        Args:
            param (str)

        """
        
       
        deltas = [np.sum(d, axis=(i+1)%2) for i, d in enumerate(delta)]
        deltas = np.array(deltas)
        deltas = deltas.ravel()
        deltas *= para_prime

        param_to_tune = getattr(layer, param)
        print("heh", param_to_tune)
        param_to_tune -= np.multiply(deltas, self.learning_rate)
        setattr(layer, param, param_to_tune)