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
        

        


    def get_class(self, input_vec, df_name=None): 
        # propagating through network
        outputs = self.arc(input_vec)
       # print("out", outputs)
        outputs = np.sum(outputs, axis=1) # make 1d
        idx_max = np.argmax(outputs)
      # print("out after", outputs)

       # max_val = max(outputs)
       # idx_max = outputs.index(max_val)
        classID = self.arc.RuleConsequentLayer.weights[idx_max]
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
        print("DF", ds)
        for (inputs_batch, targets_batch) in ds:
            accuracy = self.get_class_accuracy(inputs_batch, targets_batch)

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
                #   sample_test_accuracy =  self.accuracy_function(prediction, target)
                #  sample_test_accuracy = ones - self.error_function(prediction, target)
                #  sample_test_accuracy = np.mean(sample_test_accuracy)
                    test_loss_aggregator.append(sample_test_loss)
    #                test_loss_aggregator.append(sample_test_loss.numpy())
                    #test_accuracy_aggregator.append(sample_test_accuracy)

        # return averages per batch
        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
        test_accuracy =  np.mean(accuracy)
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
        accuracy_aggregator = []

        ## step 1: calculating gradients for each entry in batch
        # iterating over data entries of a batch
        deltas_avg = None
        assigned = False
        accuracy = self.get_class_accuracy(inputs_batch, targets_batch)
        for inputs, targets in (zip(tqdm(inputs_batch, desc='training'), targets_batch)):
        #for inputs, targets in (zip(tqdm(inputs_batch, desc='training'), targets_batch)):

            # forward propagation
            prediction =  self.arc(inputs)

            # calculating error in outputlayer
            train_loss_agg.append(self.error_function(prediction, targets))
            errorterm = self.error_function_derived(prediction, targets)
            # calculating accuracy
            #accuracy = self.accuracy_function(prediction,targets)
            #accuracy = np.mean(accuracy)
           # accuracy_aggregator.append(accuracy)
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
        
        acc = np.mean(accuracy)
        return train_loss, acc


    def error_function(self, prediction, targets):
        """Derived error function:  
            error function: tf.reduce_mean(0.5*(prediction - targets)**2)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """
        error_term = []
        
        targets = targets#.numpy()
     #   print("TAR", targets)
     #   print("pred", prediction)
      #  print("tar", targets)
        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
          #  print("tar", targets.numpy)
            
         #   print("cd", cidx)
          #  print("out", prediction)
          #  tar_row = targets[cidx] # would need to blow up targets to vectoize 
            out_row = prediction[cidx] # in order to slice [:,idx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:

                 #   print("idx", idx)
                   # print("tar",targets[idx]  )
                  #  print("out_row[:,idx]", out_row[idx])
                    error =  0.5*(targets[idx] - out_row[idx])**2
                    
                    error_term.append(error)
                # else:
                #     error_term.append(0) # for weights that are 0 0
        #error_term = tf.reduce_mean(0.5*(prediction - targets)**2)#
      #  print("error", error_term)
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
        error_term = []
       # targets = targets[0]
      #  print("tar", targets)
      #  print("pre", prediction)

        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
         #   print("tar", targets.numpy)
            
           # print("cd", cidx)
            #print("out", prediction)
            assigned = False
          #  tar_row = targets[cidx] # would need to blow up targets to vectoize 
            out_row = prediction[cidx] # in order to slice [:,idx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:
                    #print("idx", idx)
                   # print("tar",targets[idx]  )
                   # print("out_row[:,idx]", out_row[idx])
                    error =  -1*(targets[idx] - out_row[idx])
                    error_term.append(error)
                    assigned = True
            # if assigned==False:        
            #     error_term.append(0) # for weights that are 0 0
       # print("error", error_term)
        return error_term

    def calc_mf_derv_widths(self):
         # to output
        fuzzified_inputs  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        # calculating the MF values μ "mus" per input
        for xID, x in enumerate(self.arc.FuzzificationLayer.inputs):

            # there will be n_mfs mus per input
            mus_per_x = []
            for mfID in range(self.arc.FuzzificationLayer.n_mfs):

                # calling MF 
                mu = MF_gaussian_prime_b(x, self.arc.FuzzificationLayer.centers[xID][mfID], self.arc.FuzzificationLayer.widths[xID][mfID])    
                mus_per_x.append(mu)

           # print("here", mus_per_x)
            # write to TensorArray
            fuzzified_inputs = fuzzified_inputs.write(fuzzified_inputs.size(), mus_per_x)

        # return the values in the TensorArray as a stacked tensor
        fuzzified_inputs = fuzzified_inputs.stack()
        return fuzzified_inputs

    def calc_mf_derv_center(self):
            # to output
        fuzzified_inputs  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        # calculating the MF values μ "mus" per input
        for xID, x in enumerate(self.arc.FuzzificationLayer.inputs):

            # there will be n_mfs mus per input
            mus_per_x = []
            for mfID in range(self.arc.FuzzificationLayer.n_mfs):

                # calling MF 
                mu = MF_gaussian_prime_a(x, self.arc.FuzzificationLayer.centers[xID][mfID], self.arc.FuzzificationLayer.widths[xID][mfID])    
                mus_per_x.append(mu)
        
            # write to TensorArray
            fuzzified_inputs = fuzzified_inputs.write(fuzzified_inputs.size(), mus_per_x)

        # return the values in the TensorArray as a stacked tensor
        fuzzified_inputs = fuzzified_inputs.stack()
        return fuzzified_inputs
 
    
    def accuracy_function(self, prediction, target):
        
        accs = []
        target = target[0]
      #  print("tar", targets)
      #  print("pre", prediction)

        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
         #   print("tar", targets.numpy)
            
           # print("cd", cidx)
            #print("out", prediction)
            assigned = False
          #  tar_row = targets[cidx] # would need to blow up targets to vectoize 
            out_row = prediction[cidx] # in order to slice [:,idx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:
                    #print("idx", idx)
                   # print("tar",targets[idx]  )
                   # print("out_row[:,idx]", out_row[idx])
                    acc = target[idx] == np.round(out_row[idx], 0)
                    accs.append(acc)
                    assigned = True
            # if assigned==False:        
            #     error_term.append(0) # for weights that are 0 0
       # print("error", error_term)
        return accs

    def adapt(self, layer, gradients, centers_derived, widths_der):
        
        n_rows, n_cols = layer.centers.shape

        # picking first participant of a rule 
        # by looping over rows of input 
        i = 0
        for xID1 in range(n_rows):
            for mfID1 in range(n_cols):

                # print("D", gradients)
                # print("c", layer.centers[xID1][mfID1])
                # print("w", layer.widths[xID1][mfID1])

                mu1 = self.arc.RuleAntecedentLayer.inputs[xID1,mfID1] 


                # get second participant
                # by looping over the rest of rows
                for xID2 in range(xID1+1, n_rows):
                    for mfID2 in range(n_cols):  
                        mu2 = self.arc.RuleAntecedentLayer.inputs[xID2,mfID2]

                        # adapt mu1
                        delta = float32(gradients[i])
                        delta *= mu2
                        gradient_center = delta* centers_derived[xID1][mfID1]
                        gradient_width = delta* widths_der[xID1][mfID1]
                        layer.centers[xID1][mfID1] -=  np.multiply(gradient_center, self.learning_rate)
                        layer.widths[xID1][mfID1] -= np.multiply(gradient_width, self.learning_rate)
                        
                        # adapt mu2
                        delta = float32(gradients[i])
                        delta *= mu1
                        gradient_center = delta* centers_derived[xID2][mfID2]
                        gradient_width = delta * widths_der[xID2][mfID2]

                        hmm = np.repeat(self.max_vals, 3)
                        mins = np.repeat(self.min_vals, 3)
                        #print("honik",hmm)
                       # for i, p in enumerate(para):
                        #  for j in n_mfs:
                        #  print("p", p)
                        #if param_name == "centers":

                        # centers
                        # for p in gradient_center:
                        #     if p <= mins.iloc[i] or p >= hmm.iloc[i]: 
                        #         gradient_center = 0#0.00001 # randomize todo

                        #     elif i in [1,4,7,10] and p >= hmm.iloc[i] - 1/5* (hmm.iloc[i] - mins.iloc[i]): # hc
                        #         deltas[i] = 0#0.00001 # randomize todo
                        # if param_name == "widths":
                                
                        #     if p <= 0 or p >= (hmm.iloc[i] - mins.iloc[i])/3: #hc
                        #         #print("yqa")
                        #         deltas[i] = 0


                        layer.centers[xID2][mfID2] -= np.multiply(gradient_center, self.learning_rate)
                        layer.widths[xID2][mfID2] -= np.multiply(gradient_width, self.learning_rate)
                        i += 1
        return 0