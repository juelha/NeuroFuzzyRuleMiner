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

        print(ds)

        print(type(ds))
        # iterate over batch
        inputs_batch, targets_batch = ds

            
        for inp, target in (zip(tqdm(inputs_batch, desc='testing'), targets_batch)):
            
            #  print("INPUT", input)
            # forward pass to get prediction
            prediction = self.arc(inp)
            #  print("Pred", prediction)

            # print("tar", target)

            # get loss
            target = np.resize(target, prediction.shape) # the only difference to trainer
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


    def train_step(self, train_batch):
        """Implements train step for batch of datasamples
        Args:
            input (tf.Tensor): input sequence of a batch of dataset
            target (tf.Tensor): output sequence of a batch of dataset
        Returns:
            loss (float): average loss before after train step
        """
        # list for losses per batch
        losses_aggregator = []
        accuracy_aggregator = []
        # iterate over the batch
        print("inputs_batch", train_batch)
        inputs_batch, targets_batch = train_batch


        train_loss_agg = []
        accuracy_aggregator = []

        ## step 1: calculating gradients for each entry in batch
        # iterating over data entries of a batch
        deltas_avg = None
        assigned = False
        for inputs, targets in (zip(tqdm(inputs_batch, desc='training'), targets_batch)):

            # forward propagation
            prediction =  self.arc(inputs)

            # calculating error in outputlayer
            targets = np.resize(targets, prediction.shape) # the only difference to trainer
            train_loss_agg.append(self.error_function(prediction, targets))

            # calculating accuracy
            accuracy =  targets == np.round(prediction, 0)

        #  print("sample_test_accuracy", sample_test_accuracy)
            accuracy = np.mean(accuracy)
            accuracy_aggregator.append(np.mean(accuracy))
            
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
        
        # return average loss
        loss = tf.reduce_mean(train_loss_agg)
        acc = tf.reduce_mean(accuracy_aggregator)
        return loss, acc


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