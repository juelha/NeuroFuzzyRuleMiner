# basics
import tensorflow as tf
import numpy as np

# custom
import neurofuzzy_pkg.utils.MFs as MFs


class RuleConsequentLayerMam():
    """
    The RuleConsequentLayer()-Class is:
    - mapping the rule strengths calculated in RuleAntecedentLayer to the MFs of the output
    THEN-Neuron for the MFs Low, and High:
                μ_L(μ_i(x_1,x_2))
                /
    μ_i(x_1,x_2) 
                \ 
                 μ_H(μ_i(x_1,x_2))
    ___________________________________________________________
    """

    def __init__(self, mf_type=MFs.MF_gaussian, n_mfs=2):
        """Initializes RuleConsequentLayer()-Object
        
        Args:
            n_mfs (int): number of MFs, hardcoded as 2 -> binary classification
            mf_type (callable): type of MFs

        Attributes:
            tunable (boolean): if parameters of layers can be tuned during training
            inputs (list): inputs of layer
            outputs (list): outputs of layer 
            rulesTHEN (dict): used for rule extraction (THEN-Part)
            threshold (float): defines wether a rule will be counted to MF_low or MF_high
        """

        # for MFs
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.built = False

        # for training
        self.tunable = True 
        self.inputs = []
        self.outputs = []
        
        # for rule extraction
        self.rulesTHEN = {}


    def build(self, inputs):
        """Initializes trainable parameters
        Args:
            inputs (tf.Tensor): inputs
        Attr:
            centers 
            widths
            weights (nd.Array
        """
        # build centers and widths

        # since we compare firing strength of all together we build on the domain of all rule strengths
        #self.domain_input = max(inputs)

        # since firing strengths are [0,1] -> domain is
        
        domain_rules = np.full(inputs.shape, 1.0, dtype=np.float32)

        self.centers = np.asarray(MFs.center_init(self.n_mfs, domain_rules), dtype=np.float32)
        self.widths = np.asarray(MFs.widths_init(self.n_mfs, self.centers, inputs.shape[0]), dtype=np.float32)
       
        ruleIDs = ['Rule ' + str(i) for i in range(inputs.shape[0])]
        MFs.visuMFs(self, dir="after_building", func="OutputMFs", names=ruleIDs, means=inputs)

        # build weights 
        # self.weights = np.ones((self.n_mfs , input_shape[0]), dtype=np.float32)

        # build biases
        self.biases = np.full((inputs.shape[0], self.n_mfs),0.5, dtype=np.float32)

        self.train_params = {'centers':self.centers, 'widths':self.widths, 'biases':self.biases}
        self.built = True

        # call self
        return self(inputs)


    def __call__(self, inputs):
        """Calculating the MF degree of the IF-Part of a rule
        
        Args:
            inputs (tf.Tensor): rule strengths from RuleAntecedentLayer
        
        Returns:
            (tuple): can be unzipped into: 
                - out (tensor): membership degree of rules to output membership functions,
                                row per output-MFs, column per rule, has shape=(n_rules, n_mfs)
                - layer (RuleConsequentLayer): needed for defuzzification, 
                                               since we need the number of MFs and the centers of the MFs 
        """

        # check if trainable params have been built
        assert self.built, f'Layer {type(self)} is not built yet'

        self.inputs = inputs # saved for training 

        # to output
        fuzzified_outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)# []

        # for rule extraction
        self.rulesTHEN = {}
        # set up rule dict
        for mfID in range(self.n_mfs):
            self.rulesTHEN[mfID] = []
        #ruleID = 1

        # iterate over inputs (here Tnorms)
        for ruleID, x in enumerate(inputs):
            
            # calculating the MF values "mus" per input
            mus_per_x = []
            for mfID in range(self.n_mfs):

                #weighted_sum = tf.reduce_sum( tf.multiply(self.weights[mfID][i],x))
            
                # map the weighted sum to the output MFs
                act = self.mf_type(x, self.centers[ruleID][mfID], self.widths[ruleID][mfID])

                # if act above treshhold, rule gets saved to dict
                act -= self.biases[ruleID][mfID]
                mus_per_x.append(act)
                    
                if act >= 0:
                    self.rulesTHEN[mfID].append(ruleID)

              #  ruleID += 1
                
            # activation saved to output to be defuzzified in next layer
            fuzzified_outputs = fuzzified_outputs.write(fuzzified_outputs.size(), mus_per_x)


        fuzzified_outputs = fuzzified_outputs.stack()


        # check if resulting tensor has the correct shape
        assert fuzzified_outputs.shape == (inputs.shape[0], self.n_mfs), f'Output of FuzzificationLayer has wrong shape \n \
        should have shape {inputs.shape[0], self.n_mfs} but has shape {fuzzified_outputs.shape}'        
  
        self.outputs = fuzzified_outputs # saved for training 

        return fuzzified_outputs, self # returning layer as well for defuzzication

    