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
        self.builder = None
        self.n_mfs = None
        self.max_vals = None
    
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
        accuracy = self.get_class_accuracy(inputs_batch, targets_batch)
        for input, target in (zip(tqdm(inputs_batch, desc='testing'), targets_batch)):
       # for input, target in (zip(inputs_batch, targets_batch)):
            prediction = self.arc(input)
            target = np.resize(target, prediction.shape) # the only difference to trainer !!!!!!!!!!!!!!!!!!!!!
            loss = self.loss_func(prediction, target)
           # accuracy = self.accuracy_function(prediction, target)

           # accuracy =  target == np.round(prediction, 0)
            #accuracy = np.mean(accuracy)

            loss_aggregator.append(loss)
          #  accuracy_aggregator.append(np.mean(accuracy))  

        # return averages per batch
        test_loss = np.mean(loss_aggregator)
        test_accuracy =  np.mean(accuracy)
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
        accuracy = self.get_class_accuracy(inputs_batch, targets_batch)
        for inputs, targets in (zip(tqdm(inputs_batch, desc='training'), targets_batch)):
      #  for inputs, targets in (zip(inputs_batch, targets_batch)):
            # forward propagation
            prediction =  self.arc(inputs)
            # calculating error in outputlayer
            targets = np.resize(targets, prediction.shape) # the only difference to trainer !!!!!!!!!!!!!!!!!!!!!
            loss_agg.append(self.error_function(prediction, targets))
            # calculating accuracy
            
            #accuracy = np.mean(accuracy)
           # accuracy_aggregator.append(accuracy)
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
        acc = np.mean(accuracy)
        return loss, acc


    def accuracy_function(self, prediction, target):
        
        accs = []
        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
            
            y = prediction[cidx] # in order to slice [:,idx]
            t = target[cidx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:

                    acc = t[idx] == np.round(y[idx], 0)
                  #  print("here", t[idx])
                   # print("hopre", y[idx])
                    accs.append(acc)
        
        return accs#self.get_class_accuracy(prediction,target)


    def get_class(self, input_vec, df_name=None): 
        # propagating through network
        outputs = self.arc(input_vec)
       # print("out", outputs)
        outputs = np.sum(outputs, axis=1) # make 1d
        idx_max = np.argmax(outputs)
      # print("out after", outputs)

       # max_val = max(outputs)
       # idx_max = outputs.index(max_val)
        classID = self.arc.RuleConsequentLayer.class_weights[idx_max]
        return classID
    

    def get_class_accuracy(self, inputs, targets, df_name =None):
        acc = []
        for input_vec, target_vec in (zip(tqdm(inputs, desc='class testing'), targets)):
            classID = self.get_class(input_vec) 
            acc.append(self.is_class_correct(classID, target_vec))
        return np.mean(acc)

    
    def is_class_correct(self, classID, target):
        #print("target", target)
       # print("classid", classID)
        return classID == target
    
    def error_function(self, pred, tar):
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
       
        # losses = []
        # for p,t in zip(pred,tar):
        #     # Calculate cross-entropy loss
        #     loss = - np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))

        #     # Normalize by the number of examples
        #     num_examples = len(t)
        #     loss /= num_examples
        #     losses.append(loss)
        # for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
        
        #     out_row = prediction[cidx] # in order to slice [:,idx]
        #     tar = target[cidx]
        #   #  print("tar", at)
        #     for idx, number in enumerate(classweight):
        #         if bool(number)==True:

                
        #             error =  0.5*( tar[idx] - out_row[idx]  )**2
                    
        #             error_term.append(error)
     #   print("HM", np.shape(tar)[1])
        # epsilon = 1e-15  # Small constant to prevent log(0)

        # # Clip predicted probabilities to avoid log(0) or log(1)
        # pred = np.clip(pred, epsilon, 1 - epsilon)
        # tar = np.clip(tar, epsilon, 1 - epsilon)
      #  error = np.sum( tar * pred / np.shape(tar)[1], axis=1)
     #   error = 1-error
        error = 0.5*( tar - pred  )**2
       # print("e", error)
        return error
     #   return self.cross_entropy_loss(pred,tar)

    def cross_entropy_loss_prime(self, p,t):
    # https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
        return  np.sum(p- t, axis=1) #/ len(t)


    def cross_entropy_loss(self, p, t):
        # Calculate cross-entropy loss
      #  print("HERE", len(t))
        
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
        #return self.cross_entropy_loss_prime(prediction, target)
        error_term = []
    #   #  target = target[0]
        weights = self.arc.RuleConsequentLayer.class_weights
        
        output = []
        # go through weights and select the max idx
        # since weights are one-hot encoded this will match the idx of the belonging class
        for i,w in enumerate(weights):
            idx_max = np.argmax(w)
            output.append(-1* (target[i][idx_max]-prediction[i][idx_max])) 
               
        output = np.array(output)
        output = output.reshape(-1, 1)
        error = output
    #   #  print("error", error_term)
       # return  np.sum(-1* (target-prediction),axis=1)#error_term
       # return  np.sum(prediction- target,axis=1)#error_term
        #return self.cross_entropy_loss_prime(prediction, target)



        # # Clip predicted probabilities to avoid log(0) or log(1)
        # pred = np.clip(pred, epsilon, 1 - epsilon)
        # tar = np.clip(tar, epsilon, 1 - epsilon)
      #  error = np.sum( tar * pred / np.shape(tar)[1], axis=1)
     #   error = 1-error
        



    #     pred = np.where(prediction == 0, -1, prediction)
    #     tar = np.where(target == 0, -1, target)

    #   #  zeros = np.zeros_like(tar)

    #    # honk = np.concatenate((zeros, 1-tar*pred),axis=1)


    #     # my hinge
    #     error = np.sum( tar * pred / np.shape(tar)[1], axis=1)
    #     error = 1-error
     #   error = np.sum(1-tar*pred, axis=1)
        return error



    def calc_mf_derv_center(self):
        calc = MF_gaussian_prime_a(self.arc.FuzzificationLayer.inputs, self.arc.FuzzificationLayer.centers, self.arc.FuzzificationLayer.widths)
        return calc
 
    def calc_mf_derv_widths(self):
        calc = MF_gaussian_prime_b(self.arc.FuzzificationLayer.inputs, self.arc.FuzzificationLayer.centers, self.arc.FuzzificationLayer.widths)
        return calc




    def adapt(self, layer, error):
        

        # get those directly from antecedent layer with inputs attribute
        mus = self.arc.RuleAntecedentLayer.inputs
     #   print("MUS", mus[0])
      #  
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
       # delta = [mu * error for mu in mus]
        #delta.reverse() # the other mu for each 
     #   print("HERERE", deltas)
        
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
       # x.reverse()  # so it fits with convention 
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

          #  deltas = [np.sum(d, axis=(i+1)%2) for i, d in enumerate(delta)]

        
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
            deltas.append(x/81 * self.learning_rate)

            x = delta[1]
            x = np.sum(x, axis=0)
            x = np.sum(x, axis=1)
            x = np.sum(x, axis=1)
            deltas.append(x/81 * self.learning_rate)
            
            x = delta[2]
            x = np.sum(x, axis=0) # or 1
            x = np.sum(x, axis=2)
            x = np.sum(x, axis=0)
            deltas.append(x/81 * self.learning_rate)

            x = delta[3]
            x = np.sum(x, axis=1) # or 2
            x = np.sum(x, axis=0)
            x = np.sum(x, axis=0)
            deltas.append(x/81 * self.learning_rate)

        deltas = np.array(deltas)
        deltas = deltas.ravel()

        
    #   # print("para", para_prime)
    #     deltas = deltas * para_prime
        
      #  print("deltas", deltas)

        # self.fuzzi ...
       # param_to_tune = getattr(layer, param)
        
      #  print("Delta", delta)


        # params to tune
        para = getattr(layer, param_name)
        


      #  print("heh", para)
        n_mfs = 3 #hc
       # print("HOOONK",self.max_vals)
        hmm = np.repeat(self.max_vals, n_mfs)
        #print("honik",hmm)
        for i, p in enumerate(para):
          #  for j in n_mfs:
        #  print("p", p)
            if param_name == "centers":
                if p <= 0 or p >= hmm[i]: 
                    deltas[i] = 0#0.00001 # randomize todo
            if param_name == "widths":
                
                if p <= 0 or p >= hmm[i]/5: #hc
                    #print("yqa")
                    deltas[i] = 0
            # print("P",p)
       # print("deltas", deltas)
        para =  para - deltas #np.multiply(deltas, self.learning_rate)
        setattr(layer, param_name, para)