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
import neurofuzzy_pkg.utils.MFs as MFs


from tqdm import tqdm 

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
        self.builder = None
        self.n_mfs = None
        self.max_vals = None

    def __call__(self, train_ds, test_ds, validation_ds, constraint_center =None, constraint_width=None, learning_rate=None, n_epochs=None):
        """Running the training loop and saving the MFs before and after 
        
        Args:
            train_ds (tuple of two numpy.ndarrays): dataset for training
            test_ds (tuple of two numpy.ndarrays): dataset for testing
        """
        self.constraint_center = constraint_center
        self.constraint_width = constraint_width
        self.learning_rate = learning_rate
        self.training_loop(train_ds, test_ds, validation_ds, n_epochs=n_epochs)
        





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
            test_ds (tuple of two numpy.ndarrays): batch of testing dataset
        Returns:
            test_loss (float): average loss from output to target
            test_accuracy (float): average accuracy of output
        """

        accuracy_aggregator = []
        loss_aggregator = []

        inputs_batch, targets_batch = ds
        accuracy = self.get_class_accuracy(inputs_batch, targets_batch)
        for input, target in (zip(tqdm(inputs_batch, desc='testing'), targets_batch)):
            prediction = self.arc(input)
            target = np.resize(target, prediction.shape) # the only difference to trainer !!!!!!!!!!!!!!!!!!!!!
            loss = self.loss_func(prediction, target)
            loss_aggregator.append(loss)
        # return averages per batch
        test_loss = np.mean(loss_aggregator)
        test_accuracy =  np.mean(accuracy)
        return test_loss, test_accuracy


    def train_step(self, train_batch):
        """Implements train step for batch of datasamples
        Args:
            train_batch (tuple of two numpy.ndarrays): batch of testing dataset
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
        accuracy = self.get_class_accuracy(inputs_batch, targets_batch)
        for inputs, targets in (zip(tqdm(inputs_batch, desc='training'), targets_batch)):
            # forward propagation
            prediction =  self.arc(inputs)
            # calculating error in outputlayer
            targets = np.resize(targets, prediction.shape) # the only difference to trainer !!!!!!!!!!!!!!!!!!!!!
            loss_agg.append(self.error_function(prediction, targets))
            errorterm = self.error_function_derived(prediction, targets)

            self.adapt(self.arc.FuzzificationLayer, errorterm)
        
        # return average loss
        loss =  np.mean(loss_agg)
        acc = np.mean(accuracy)
        return loss, acc


    def accuracy_function(self, prediction, target):
        
        accs = []
        weights = self.arc.RuleConsequentLayer.class_weights
        output = []
        # go through weights and select the max idx
        # since weights are one-hot encoded this will match the idx of the belonging class
        for i,w in enumerate(weights):
            idx_max = np.argmax(w)
            accs.append(target[i][idx_max] == np.round(prediction[i][idx_max], 0)) 
               
        accs = np.array(accs)
        accs = accs.reshape(-1, 1)
        
        return accs

    def get_class(self, input_vec, df_name=None): 
        # propagating through network
        outputs = self.arc(input_vec)
        outputs = np.sum(outputs, axis=1) # make 1d
        idx_max = np.argmax(outputs)
        classID = self.arc.RuleConsequentLayer.class_weights[idx_max]
        return classID
    

    def get_class_accuracy(self, inputs, targets, df_name =None):
        acc = []
        for input_vec, target_vec in (zip(tqdm(inputs, desc='class testing'), targets)):
            classID = self.get_class(input_vec) 
            acc.append(classID == target_vec)
        return np.mean(acc)

    
    def is_class_correct(self, classID, target):
        return classID == target
    
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
        weights = self.arc.RuleConsequentLayer.class_weights
        
        output = []
        # go through weights and select the max idx
        # since weights are one-hot encoded this will match the idx of the belonging class
        for i,w in enumerate(weights):
            idx_max = np.argmax(w)
            output.append(0.5* (target[i][idx_max]-prediction[i][idx_max])**2) 
               
        output = np.array(output)
        output = output.reshape(-1, 1)
        error = output
        return error

    def cross_entropy_loss_prime(self, p,t):
    # https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
        return  np.sum(p- t, axis=1) #/ len(t)


    def cross_entropy_loss(self, p, t):
        return - np.sum(t * np.log(p), axis=1) / len(t)


    def error_function_derived(self, prediction, target):
        """Derived error function:  
            derived error function: (prediction - targets)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """
        weights = self.arc.RuleConsequentLayer.class_weights
        output = []
        # go through weights and select the max idx
        # since weights are one-hot encoded this will match the idx of the belonging class
        for i,w in enumerate(weights):
            idx_max = np.argmax(w)
            output.append(-1* (target[i][idx_max]-prediction[i][idx_max])) 
               
        output = np.array(output)
        output = output.reshape(-1, 1)
        error = output#1-output / np.shape(target)[1]
        return error


    def calc_mf_derv_center(self):
        calc = MF_gaussian_prime_a(self.arc.FuzzificationLayer.inputs, self.arc.FuzzificationLayer.centers, self.arc.FuzzificationLayer.widths)
        return calc
 
    def calc_mf_derv_widths(self):
        calc = MF_gaussian_prime_b(self.arc.FuzzificationLayer.inputs, self.arc.FuzzificationLayer.centers, self.arc.FuzzificationLayer.widths)
        return calc


    def adapt(self, layer, error, ):
        

        mus = self.arc.RuleAntecedentLayer.inputs
   
        # reshape error to match each mu 
        error = np.reshape(error, mus[0].shape)
        deltas = []
        for i, _ in enumerate(mus):
            delta = error
            for j, mu in enumerate(mus):
                if i==j:
                    continue
                delta = delta* mu
            deltas.append(delta)
        
        centers_prime = self.calc_mf_derv_center()
        widths_prime = self.calc_mf_derv_widths()

        self.adapt_parameter('centers', layer, deltas, centers_prime)
        self.adapt_parameter('widths', layer, deltas, widths_prime)

        return 0
    
    def adapt_parameter(self, param_name, layer, delta, para_prime):
        """
        Args:
            param_name (str)

        """

        # deriv para have to be meshgridded too
        para_prime = np.array_split(para_prime, range(self.n_mfs, len(para_prime), self.n_mfs)) # hc
        para_prime = np.array(np.meshgrid(*para_prime,indexing='ij'))
        delta = [d * para_prime[i] for i, d in enumerate(delta)]

        deltas = []
        if delta[0].ndim == 2:


            x = delta[0]
            x = np.sum(x, axis=1)
            deltas.append(x)#/9* self.learning_rate)
            x = delta[1]
            x = np.sum(x, axis=0)
            deltas.append(x)#/9* self.learning_rate)

        elif delta[0].ndim == 3: 
            #print("true")
            x = delta[0]
            x = np.sum(x, axis=(1,2)) # works
            deltas.append(x* self.learning_rate)

            x = delta[1]
            x = np.sum(x, axis=0)
            x = np.sum(x, axis=1)
            deltas.append(x* self.learning_rate)

            x = delta[2]
            x = np.sum(x, axis=1)            
            x = np.sum(x, axis=0)
            deltas.append(x* self.learning_rate)

        elif delta[0].ndim == 4:
            x = delta[0]
            x = np.sum(x, axis=3)
            x = np.sum(x, axis=1)
            x = np.sum(x, axis=1)
            deltas.append(x/81* self.learning_rate)

            x = delta[1]
            x = np.sum(x, axis=0)
            x = np.sum(x, axis=1)
            x = np.sum(x, axis=1)
            deltas.append(x/81* self.learning_rate)
            
            x = delta[2]
            x = np.sum(x, axis=0) # or 1
            x = np.sum(x, axis=2)
            x = np.sum(x, axis=0)
            deltas.append(x/81* self.learning_rate)

            x = delta[3]
            x = np.sum(x, axis=1) # or 2
            x = np.sum(x, axis=0)
            x = np.sum(x, axis=0)
            deltas.append(x/81 * self.learning_rate)

        deltas = np.array(deltas)
        deltas = deltas.ravel()

        # params to tune
        para = getattr(layer, param_name)
        
        n_mfs = 3 #hc
        maxs = np.repeat(self.max_vals, n_mfs)
        mins = np.repeat(self.min_vals, n_mfs)
        for i, p in enumerate(para):
            if param_name == "centers":
                if p <= mins.iloc[i] or p >= maxs.iloc[i]: 
                    deltas[i] = 0#0.00001 # randomize todo

                elif i in [1,4,7,10] and p >= maxs.iloc[i] - (self.constraint_center* (maxs.iloc[i] - mins.iloc[i])): # hc
                    deltas[i] = 0#0.00001 # randomize todo
            if param_name == "widths":
                
                if p <= 0 or p >= (maxs.iloc[i] - mins.iloc[i])/(self.constraint_width*(n_mfs)): #hc
                    deltas[i] = 0

        para =  para - deltas 
        setattr(layer, param_name, para)