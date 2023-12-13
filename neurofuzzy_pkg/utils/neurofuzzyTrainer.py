# basics
from numpy import dtype, float32
import tensorflow as tf
import numpy as np

# custom
from model_pkg import Trainer
from neurofuzzy_pkg.fuzzyLayers import MF_gaussian_prime_a
from neurofuzzy_pkg import utils
from neurofuzzy_pkg.utils.MFs import MF_gaussian,MF_gaussian_prime_a, MF_gaussian_prime_b

from tqdm import tqdm 
#from neurofuzzy_pkg.fuzzyLayers.RuleConsequentLayer import RuleConsequentLayer
   

class neurofuzzyTrainer(Trainer):   
    """
    The Trainer() Class is:
    - inherited and 
    - train_step() is overwritten for the tuning of the MFs parameters
    ___________________________________________________________
    """

    def __init__(self):#, ite = 10):# loss_func=None, optimizer_func=Adam):
        """Initializing neurofuzzyTrainer by inheriting from Trainer
        """
        super().__init__(neurofuzzyTrainer)#, ite)#, iters, learning_rate )


    def __call__(self, train_ds, test_ds, validation_ds):
        """Running the training loop and saving the MFs before and after 
        
        Args:
            train_ds (PrefetchDataset): dataset for training
            test_ds (PrefetchDataset): dataset for testing
        """
        
        ## run once to init the parameters of the membership functions for comparison
        # picking random batch from dataset
      #  test_ds = self.pick_batch(test_ds)
       # train_ds = self.pick_batch(train_ds)

        # run model on test_ds to keep track of progress during training
       # self.test(test_ds)

        # layers to keep track of
        inputMFs = self.arc.FuzzificationLayer
        outputMFs = self.arc.RuleConsequentLayer

        # saving figs before training 
       # utils.MFs.visuMFs(outputMFs, dir="after_building", func="outputMFs", names=self.feature_names, means=self.inputs_mean)

        # train
      #  super().__call__(train_ds, test_ds)
        self.training_loop(train_ds, test_ds, validation_ds)


        # saving figs after training
        utils.MFs.visuMFs(inputMFs, dir="after_training", func="inputMFs", names=self.feature_names, means=self.inputs_mean)
        #utils.MFs.visuMFs(outputMFs, dir="after_training", func="outputMFs", names=self.feature_names, means=self.inputs_mean)

        self.visualize_training(self.arc.Name)



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
            #test_ds = tf.random.shuffle(test_ds)
            #validation_ds = tf.random.shuffle(validation_ds)

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


  

    def train_step(self, inputs_batch, targets_batch):
        """Tuning the parameters of the MFs using Backpropagation
        Args:
            input (tf.Tensor): input sequence of a dataset
            target (tf.Tensor): output sequence of a dataset
            
        Returns: 
            loss (float): loss before the trainig step
        """

        errors_average = [] # average over all errors in a batch
        gradients = {} # dict for gradients

        # setting up gradients dict 
        for layerID, layer in  enumerate(reversed(list(self.arc.internal_layers))): 

            # if layer has parameters to tune
            if layer.tunable:

                # set up dict per layer
                list_params = layer.train_params.keys()
                gradients[layerID] = {k: [] for k in list_params}

        ## step 1: calculating gradients for each entry in batch
        # iterating over data entries of a batch
        deltas_avg = None
        assigned = False
        for inputs, targets in (zip(tqdm(inputs_batch, desc='training'), targets_batch)):
        #for inputs, targets in zip(inputs_batch,targets_batch):

            # forward propagation
            prediction =  self.arc(inputs)
            print("out", prediction)
            print("tar", targets)

            # calculating error in outputlayer
            errorterm = self.error_function_derivedMyArc(prediction, targets)
            errors_average.append(self.error_function(prediction, targets))
            print("errorterm", errorterm)
            delta = np.array(errorterm)
            delta = np.reshape(delta, (495,1))
           #delta = np.ones(shape=(495,1)) 
            if assigned == False: 
                deltas_avg = delta
                assigned = True
            else:

                print("ddddelta", np.shape(delta))
                print("asfasdgf", np.shape(deltas_avg))
                deltas_avg = np.concatenate((deltas_avg, delta), axis=1)

            # backpropagation part
          #  for layerID, layer in  enumerate(reversed(list(self.arc.internal_layers))): 

 

                # if layer has parameters to tune
                #if layer.tunable:

                    # # get delta, the inforamtion about the error of a layer
                    # delta = self.get_deltaMyArc(layer, delta)

                    # # calculate gradient for each param 
                    # for param in layer.train_params:
                    #     grad = self.calc_grads(param, delta, layer)
                    #     gradients[layerID][param].append(grad)

        # ## step 2: get averages of all entries
        # # error
        # errors_average = tf.reduce_mean(errors_average)  
        # gradients
        print("del", deltas_avg)
        deltas_avg = np.mean(deltas_avg,axis=1)
        gradients= deltas_avg
        print("gradients", gradients)
        print("gradients", np.shape(gradients)) # 495
        centers_derived = self.calc_mf_derv_center()
        widths_der = self.calc_mf_derv_widths()
        # for layerID in gradients:
        #     for param in gradients[layerID]:
        #         gradients[layerID][param] = tf.stack(gradients[layerID][param])
        #         gradients[layerID][param] = tf.reduce_mean(gradients[layerID][param], axis=0)
       # print("gradients after stack", gradients)

        ## step 3: adapt the parameters with average gradients
        for layerID,layer in enumerate(reversed(list(self.arc.internal_layers))):  
        
            # if layer has parameters to tune
            if layer.tunable:
                self.adaptMyArc(layer, gradients, centers_derived, widths_der)
        
        return errors_average


    def calc_mf_derv_widths(self):
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

    def calc_mf_derv_center(self):
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
        
            # write to TensorArray
            fuzzified_inputs = fuzzified_inputs.write(fuzzified_inputs.size(), mus_per_x)

        # return the values in the TensorArray as a stacked tensor
        fuzzified_inputs = fuzzified_inputs.stack()
        return fuzzified_inputs

    def error_function_derivedMyArc(self, prediction, targets):
        """Derived error function:  
            derived error function: (prediction - targets)
        
        Args:
            prediction (tf.Tensor): output of model
            target (tf.Tensor): desired output of model 
        Returns:
            error_term (float): output of derived error function
        """
        
        targets = np.resize(targets,new_shape=(495, 2))
      #  print(targets)

     #   print("pr",prediction)
        prediction = prediction.numpy() 
     #   print("pr np", prediction)

      #  print("shapes")
     #   print(np.shape(targets))
     #   print(np.shape(prediction))
       # np.concatenate((targets,prediction))
        error_term = []
        for i in range(495):
            term = -1*np.dot(prediction[i],targets[i]) # NOTE CHANGED BC OF ONE HOT ENCODED OUTPUT VECTOR
            error_term.append([term])
        #err
        # or_term = (prediction - targets)
        return error_term

    def get_deltaMyArc(self,layer,deltas):


        return deltas

    def get_delta(self, layer, deltas):
        """Get the deltas of a layer, 
        deltas are used to backpropagate information about the error through the network

        Note: calculation of delta depends on parameters that are going to be adapted
        -> 
        layer has either mfs paras or weights 

        Args:
            layer (Custom-Class): the layer to adapt 
            deltas (tf.Tensor): shape=(n_mfs of successive layer,)
        Returns:
            deltas (tf.Tensor): shape=(n_mfs of current layer)
        """

        # if layer has weights
        if hasattr(layer, 'weights'):

          #  print("La", layer)
           # print("wei", layer.weights.shape)
            #print("de", deltas)

            # get weight shape
            weights_shape = len(layer.weights.shape)
          #  print("sha", weights_shape)


            if len(layer.weights.shape) == 1:

                if  layer.weights.shape[0] > deltas.shape[0]:

                    # broadcast

                    deltas = tf.expand_dims(deltas, 1)

                    weights = layer.weights
                    weights = tf.expand_dims(weights, 1)

                    # multiply
                    deltas = tf.multiply(deltas, layer.weights)

                    # get mean over redundant axis 
                    deltas = tf.reduce_mean(deltas, axis=1) 

                    # reshape delta to match input shape
                    deltas = tf.reshape(deltas,  layer.weights.shape[0])

                    return deltas


            

            if len(layer.weights.shape) == 2:

                if  layer.weights.shape[1] != deltas.shape[0]:

                    # flatten deltas arr
                    deltas = tf.reshape(deltas, [-1])

                    # reshape deltas to match input size 
                    input_size  = tf.reduce_prod(layer.weights.T.shape)
                    delta_size = tf.reduce_prod(deltas.shape)
                    deltas = tf.reshape(deltas, (input_size,int(delta_size/input_size)))

                    # get mean over redundant axis 
                    deltas = tf.reduce_mean(deltas, axis=1) 

                    # reshape delta to match input shape
                    deltas = tf.reshape(deltas, ( layer.weights.shape[0],  layer.weights.shape[1]))
            
            new_deltas = tf.multiply(deltas, layer.weights)

            return new_deltas 


        # if layer has tunable mf params 
        if hasattr(layer, 'n_mfs'):
            
            mf_prime  = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
                
            # calculating mu per input
            for xID, x in enumerate(layer.inputs):
                mus_per_x = []

               
                for mfID in range(layer.n_mfs):
                    mu = (MF_gaussian_prime_a(x, layer.centers[xID][mfID], layer.widths[xID][mfID]))
                    mus_per_x.append(mu)

                mf_prime = mf_prime.write(mf_prime.size(), mus_per_x)

            mf_prime = mf_prime.stack()

            deltas = tf.multiply(layer.centers, deltas )            
            deltas = tf.multiply(deltas, mf_prime) 
            return deltas 

        
            


    def calc_grads(self, param, deltas, layer):
        """Calculating the gradient for the given parameter
        -> refer to documentation for mathematical explanation 
        
        Args:
            param (str): param of layer to be adapted 
            deltas (tf.Tensor): part of gradient equation 
            layer (Custom-Class): Layer to be adapted 
        
        Returns:
            gradients (tensor): for each para one gradient, shape=(n_paras,)
        Raise:
            AssertionError: if output shape does not match parameter shape 
        """

        if param == "centers" or param == "widths":
            return deltas # !
            n_params = layer.centers.shape
            grads = self.calc_grads_mfs( deltas, layer)

        elif param == "weights":
            n_params = layer.weights.shape #len(layer.weights)
            grads = self.calc_grads_weights(deltas, layer)

        elif param == "biases":
            n_params = layer.biases.shape
            grads = self.calc_grads_bias(deltas, layer)

       # print("grads", grads)
       # print()
        assert  (grads.shape[0] == n_params[0] and grads.shape[1] == n_params[1]), f'Gradient has wrong shape \n \
        should have shape {n_params[0], n_params[1]} but has shape {grads.shape[0], grads.shape[1]} \n \
        layer: {type(layer)} \n \
        parameter: {param} \n '


        return grads


    def calc_grads_mfs(self, delta, layer):
        """Calculating the gradient for the parameter "centers" or "widths" of the MFs
        -> refer to documentation for mathematical explanation 
        
        Args:
            deltas (tf.Tensor): part of gradient equation 
            layer (Custom-Class): Layer to be adapted 
        
        Returns:
            gradients (tensor): for each para one gradient, shape=(n_paras,)
        """

        sum_centers = tf.cast(tf.reduce_sum(layer.centers),dtype=tf.float32)
        gradients = delta * sum_centers
        
        return gradients


    def calc_grads_weights(self, deltas, layer):
        """Calculating the gradient for the parameter "weights"
        -> refer to documentation for mathematical explanation 
        
        Args:
            deltas (tf.Tensor): part of gradient equation 
            layer (Custom-Class): Layer to be adapted 
        
        Returns:
            gradients (tensor): for each para one gradient, shape=(n_paras,)
        """
       
        gradients = deltas*layer.inputs

        return gradients


    def calc_grads_bias(self, deltas, layer):
        """Calculating the gradient for the parameter "biases" 
        -> refer to documentation for mathematical explanation 
        
        Args:
            deltas (tf.Tensor): part of gradient equation 
            layer (Custom-Class): Layer to be adapted 
        
        Returns:
            gradients (tensor): for each para one gradient, shape=(n_paras,)
        """

        sum_centers = tf.cast(tf.reduce_sum(layer.centers),dtype=tf.float32)
        gradients = deltas * sum_centers * (-1)

        return gradients

    def adaptMyArc(self, layer, gradients, centers_derived, widths_der):
        
        if hasattr(layer, 'n_mfs'):

            n_rows, n_cols = layer.centers.shape

            # picking first participant of a rule 
            # by looping over rows of input 
            
          #  print("delt", gradients)
            for delta in gradients:
                for xID1 in range(n_rows):
                    for mfID1 in range(n_cols):

                        # print("D", gradients)
                        # print("c", layer.centers[xID1][mfID1])
                        # print("w", layer.widths[xID1][mfID1])
                        other_mu = self.arc.RuleAntecedentLayer.inputs[xID1+1,mfID1] # get tghe other mu errror

                        delta *= other_mu

                        delta_center = delta* centers_derived[xID1][mfID1]
                        delta_widths = delta * widths_der[xID1][mfID1]
                    
                        layer.centers[xID1][mfID1] -= np.multiply(delta_center, self.learning_rate)
                        layer.widths[xID1][mfID1] -= np.multiply(delta_widths, self.learning_rate)
                        

                        # get second participant
                        # by looping over the rest of rows
                        for xID2 in range(xID1+1, n_rows):
                            for mfID2 in range(n_cols):  
                                other_mu = self.arc.RuleAntecedentLayer.inputs[xID1,mfID1]
                                delta *= other_mu
                                delta_center = delta* centers_derived[xID2][mfID2]
                                delta_widths = delta * widths_der[xID2][mfID2]
                               # print("delta," ,)
                                layer.centers[xID2][mfID2] -= np.multiply(delta_center, self.learning_rate)
                                layer.widths[xID2][mfID2] -= np.multiply(delta_widths, self.learning_rate)
            print("centers", layer.centers)
            print("widths", layer.widths)

    def adapt(self, layer, gradients):
        """Adapt the parameters using the gradients from calc_grads
        Args:
            gradients (dict): dict of gradients to add to respective parameters
        """
        # if layer has tunable mf paras 
        if hasattr(layer, 'n_mfs'):
            for param in gradients:
                if param == "centers":
                    layer.centers -= np.multiply(gradients[param], self.learning_rate)
                if param == "widths":
                    layer.widths -= np.multiply(gradients[param], self.learning_rate)
                elif param == "weights":
                    layer.weights -= np.multiply(gradients[param], self.learning_rate)
                elif param == "biases":
                    layer.biases -= np.multiply(gradients[param], self.learning_rate)
        else:
            for param in gradients:
                if param == "weights":
                    layer.weights -= np.multiply(gradients[param], self.learning_rate)

        return 0


