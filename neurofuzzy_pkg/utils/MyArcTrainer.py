# basics
from numpy import dtype, float32
import tensorflow as tf
import numpy as np

# custom
from model_pkg import Trainer
#from neurofuzzy_pkg.fuzzyLayers import MF_gaussian_prime_a
from neurofuzzy_pkg import utils
from neurofuzzy_pkg.utils.MFs import MF_gaussian,MF_gaussian_prime_a, MF_gaussian_prime_b
from neurofuzzy_pkg.utils.MFs import MF_tri,MF_tri_prime_a, MF_tri_prime_b


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


    def __call__(self, train_ds, test_ds, validation_ds):
        """Running the training loop and saving the MFs before and after 
        
        Args:
            train_ds (PrefetchDataset): dataset for training
            test_ds (PrefetchDataset): dataset for testing
        """


        utils.MFs.visuMFs(self.arc.FuzzificationLayer, dir="before_training", func="inputMFs", names=self.feature_names)
       
        # train
        self.training_loop(train_ds, test_ds, validation_ds)

        # saving figs after training
        utils.MFs.visuMFs(self.arc.FuzzificationLayer, dir="after_training", func="inputMFs", names=self.feature_names)

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
               # print(target)
                sample_test_loss = self.error_function(prediction, target)
               # print("sample_test_loss",sample_test_loss)
                # get accuracy
                sample_test_accuracy =  target == np.round(prediction, 1)
              #  sample_test_accuracy = ones - self.error_function(prediction, target)
                sample_test_accuracy = np.mean(sample_test_accuracy)
                test_loss_aggregator.append(sample_test_loss)
#                test_loss_aggregator.append(sample_test_loss.numpy())
                test_accuracy_aggregator.append(np.mean(sample_test_accuracy))  

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
            n_rules = 9
            delta = np.reshape(delta, (n_rules,1))
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
        targets = targets[0]
        for cidx,classweight in enumerate(self.arc.RuleConsequentLayer.weights):
          #  print("tar", targets.numpy)
            
         #   print("cd", cidx)
          #  print("out", prediction)
          #  tar_row = targets[cidx] # would need to blow up targets to vectoize 
            out_row = prediction[cidx] # in order to slice [:,idx]
            for idx, number in enumerate(classweight):
                if bool(number)==True:

                 #   print("idx", idx)
                  #  print("tar",targets[idx]  )
                   # print("out_row[:,idx]", out_row[idx])
                    error =  0.5*(targets[idx] - out_row[idx])**2
                    error_term.append(error)
                # else:
                #     error_term.append(0) # for weights that are 0 0
        #error_term = tf.reduce_mean(0.5*(prediction - targets)**2)
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
        targets = targets[0]
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
                        layer.centers[xID2][mfID2] -= np.multiply(gradient_center, self.learning_rate)
                        layer.widths[xID2][mfID2] -= np.multiply(gradient_width, self.learning_rate)
                        i += 1
        return 0