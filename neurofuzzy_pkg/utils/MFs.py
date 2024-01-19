# basics
import numpy as np
import matplotlib.pyplot as plt
import os 

"""
Collection of 
- Membership functions and 
- the functions needed for the initialization of their parameters
"""

def center_init_con(n_mfs, feature_ranges):
    """Calculating the centers of the MFs by dividing domain of input equally
    Args:
        n_mfs (int): number of MFs
        domain_input (int): upper boundary of the domain of the input 
    Returns:
        centers (list(float)): the centers for the respective MFs
    """
    centers = []
    
    for i in range(n_mfs):
        centers.append((feature_ranges/(n_mfs+1))*(i+1))
    return centers

def center_init(n_mfs, feature_ranges):
    """Calculating the centers of the MFs by dividing domain of input equally
    Args:
        n_mfs (int): number of MFs
        domain_input (int): upper boundary of the domain of the input 
        inputs (tf.Tensor): per input one mf
    Returns:
        centers (list(float)): the centers for the respective MFs
    """
   # print("\n")
    centers = []
   # print("inputs", inputs)

    for x in feature_ranges:
       # print("X", x)
        centers_per_x = []
       # print("\n")
        for i in range(n_mfs):
            center = (x/(n_mfs+1))*(i+1)
           # print("center", center)
            centers_per_x.append(center)
        centers.append(centers_per_x)

    # centers = np.asarray(centers, dtype=np.float32)

    # centers = centers.T # get shape (n_mfs, n_inputs)

    return centers



def widths_init(n_mfs, centers, n_inputs):
    """
    """
    widths = []
    counter = 0
   # print("c", centers)

    for xID in range(n_inputs):
        widths_per_x = []
       # print("\n")
        for i in range(n_mfs):
            
            widths_per_x.append(centers[xID][0])
           # print("center", center)

        widths.append(widths_per_x)

    # widths = np.asarray(widths, dtype=np.float32)

    # widths = widths.T # get shape (n_mfs, n_inputs)
    
 #   print("hE",widths)
    
    return widths

def MF_gaussian(x, a, b):
    """ Gaussian membership function
    Args:
        x (tensor): input to fuzzify, shape=() 
        a (float): center of MF
        b (float): width of MF
    Returns:
        mu (float): degree of membership of x
        
    Raises:
        AssertionError; if output is outside bounds
    """    
    expo = -0.5*(((x-a)/b)**2)
    mu = np.exp(expo)

    # assert (mu <= 1 ) & (mu >= 0), f'Degree of membership is outside bounds, \n\
    # refer to formal def of mf: µA:X → [0,1] \n\
    # inputs: x: {x}; a: {a}; b: {b}; mu: {mu}'
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

def visuMFs(layer, dir, func, max_vals):
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
    
    feature_names = max_vals.keys().values.tolist()
    print(layer.centers)
  #  length_inputs = tf.shape(layer.centers)[0]
    # for each input see what the mfs mean -> domain input might be diff
    for xID, max_value in enumerate(max_vals):
    #for name, input in zip(names,layer.inputs):
#        x = np.arange(0, means[xID] , (means[xID]*0.01))
        x = np.arange(0, max_value, 0.01)
        y = {}

        for mfID in range(layer.n_mfs):

#            print(layer.centers[j+i*layer.n_mfs])
            y[mfID] = []
            for bleh in x:
                y[mfID].append(layer.mf_type(bleh,layer.centers[xID][mfID],layer.widths[xID][mfID]))

            plt.plot(x, y[mfID])
            plt.axvline(layer.centers[xID][mfID],0,1, c=plt.gca().lines[-1].get_color(), ls='--')



 

        plt.title('Membership Functions')
        plt.ylabel('Degree of Membership')
        plt.xlabel(feature_names[xID])

        
        # get save path 
        file_name = func + '_MFs_' + feature_names[xID] + '.png'
        save_path = os.path.dirname(__file__) +  '/../../results/figs/' + dir
        completeName = os.path.join(save_path, file_name)

        plt.savefig(completeName)
        plt.clf()
