# basics
import numpy as np
import matplotlib.pyplot as plt
import os 

"""
Collection of 
- Membership functions and 
- the functions needed for the initialization of their parameters
"""



def center_init(x, n_mfs):
    """Initializes the centers of MFs by partitioning the domain of the features

    Args:
        x (numpy.ndarray): the max values for each feature, i.e. the domain
        n_mfs (int): number of MFs in Fuzzification Layer
        
    Returns: 
        numpy.ndarray: initalized widths with the shape (x.size,)
    """
    n_inputs = x.size // n_mfs
    # multiplicator = np.tile(np.arange(1, n_mfs + 1), n_inputs)
    # cetnters = (x / (n_mfs + 1)) * multiplicator


    multiplicator = np.tile(np.arange(0, n_mfs ), n_inputs)
    cetnters = (x / (n_mfs -1)) * multiplicator
  #  print(cetnters)
    return cetnters


def widths_init(x, n_mfs):
    """Initializes the widths of MFs by partitioning the domain of the features

    Args:
        x (numpy.ndarray): the max values for each feature, i.e. the feature's domain
        n_mfs (int): number of MFs in Fuzzification Layer

    Returns: 
        numpy.ndarray: initalized widths with the shape (x.size,)
    """
    return x/(2*n_mfs+1)


def MF_gaussian(x, center, width):
    """ Gaussian membership function
    Args:
        x (tensor): input to fuzzify, shape=() 
        center (numpy.ndarray): centers of MF
        width (numpy.ndarray): widths of MF

    Returns:
        mu (numpy.ndarray): degrees of membership of x 
        
    Raises:
        AssertionError: if output is outside bounds
    """   
    mu = np.exp(-0.5*(((x-center)/width)**2))
    assert (mu.any() <= 1) & (mu.any() >= 0), f'Degree of membership is outside bounds, \n\
    refer to formal def of mf: µA:X → [0,1] \n\
    inputs: x: {x}; center: {center}; width: {width}; mu: {mu}'
    return mu



def MF_gaussian_prime_a(x, a, b):
    """Derived Gaussian membership function with respective to a 
    Args:
        x (tensor): input to fuzzify
        a (float): center of MF
        b (float): width of MF
    Returns:
        mu (float): degree of membership of x
    """    
    expo = -0.5*(((x-a)/b)**2)
    factor = (a-x)/(b**2)
    mu =  factor*np.exp(expo)
    return mu


def MF_gaussian_prime_b(x, a, b):
    """Derived Gaussian membership function with respective to b 
    Args:
        x (tensor): input to fuzzify
        a (float): center of MF
        b (float): width of MF
    Returns:
        mu (float): degree of membership of x
    """    
    expo = -0.5*(((x-a)/b)**2)
    factor = (a-x)**2/(b**3)
    mu =  factor*np.exp(expo)
    return mu


def MF_tri(x, a, b): 
    """Triangular formed membership function
    based on paper "A Learning Method of Fuzzy Inference Rules by Descent Method" (Nomura et al., 1992) 
    Args:
        x (tensor): input to fuzzify
        a (float): center of MF
        b (float): width of MF
    Returns:
        mu (float): degree of membership of x
    
    Raises:
        AssertionError; if output is outside bounds
    """
    mu = 1- ((2*abs(x-a)/b))
    # assert (mu <= 1 )& (mu >= 0), 'Degree of membership is outside bounds, \
    #                                 refer to formal def of mf: µA:X → [0,1]'
    return mu

def MF_tri_prime_a(x,a,b):
    mu = (2*(x-a))/(b*abs(x-a))
    return mu

def MF_tri_prime_b(x, a, b):
    mu = (2*abs(a-x)/b**2)
    return mu

def visuMFs(layer, dir, df_name, max_vals, mf_names=["low", "middle", "high"]):
    """Visualizing the current MFs
visuMFs(inputMFs, self.arc, dir="before_training", func="inputMFs")
inputMFs.mf_type, inputMFs.n_mfs, inputMFs.centers, inputMFs.widths, inputMFs.domain_input, 
Args:
        type_mf (callable): MF
        n_mfs (int): number of MFs
        centers (tensor): centers of MFs
        widths (tensor): widths of MF
        domain_input (int): upper boundary of the domain of the input 
        file_name (str): name of plot that will be saved in analysis folder 
    """
    n_mfs = 3
    
    feature_names = max_vals.keys().values.tolist()
    c = np.array_split(layer.centers, 2)
    c = np.array_split(c, range(n_mfs, len(c), n_mfs))
    w = np.array_split(layer.widths,2)
    w = np.array_split(w, range(n_mfs, len(w), n_mfs))

    c = c[0]
    w = w[0]
   
   # print("c", c)
    #print("w", w)
  #  length_inputs = tf.shape(layer.centers)[0]
    # for each input see what the mfs mean -> domain input might be diff
    for xID, max_value in enumerate(max_vals):
    #for name, input in zip(names,layer.inputs):
#        x = np.arange(0, means[xID] , (means[xID]*0.01))
        x = np.arange(0, max_value, 0.01)
        y = {}

        for mfID in range(layer.n_mfs):
            y[mfID] = []

#            print(layer.centers[j+i*layer.n_mfs])
            for bleh in x:
                y[mfID].append(layer.mf_type(bleh,c[xID][mfID],w[xID][mfID]))

            plt.plot(x, y[mfID], label=mf_names[mfID])

          #  print("hui", c[xID][mfID])
            plt.axvline(c[xID][mfID],0,1, c=plt.gca().lines[-1].get_color(), ls='--')


        plt.legend()

        plt.title('Membership Functions')
        plt.ylabel('Degree of Membership')
        plt.xlabel(feature_names[xID])

        
        # get save path 
        file_name =  '_MFs_' + feature_names[xID] + '.png'
        save_path = os.path.dirname(__file__) +  f'/../../results/{df_name}/figures/' + dir
        completeName = os.path.join(save_path, file_name)

        plt.savefig(completeName)
        plt.clf()