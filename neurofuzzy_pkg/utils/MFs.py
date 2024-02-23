# basics
import numpy as np
import matplotlib.pyplot as plt
import os 

"""
Collection of 
- Membership functions and 
- the functions needed for the initialization of their parameters
"""



def center_init(feature_mins, feature_maxs, n_mfs):
    """Initializes the centers of MFs by partitioning the domain of the features

    Args:
        x (numpy.ndarray): the max values for each feature, i.e. the domain
        n_mfs (int): number of MFs in Fuzzification Layer
        
    Returns: 
        numpy.ndarray: initalized widths with the shape (x.size,)
    """
    n_inputs = feature_maxs.size // n_mfs
    # multiplicator = np.tile(np.arange(1, n_mfs + 1), n_inputs)
    # cetnters = (x / (n_mfs + 1)) * multiplicator
    centers = []
    for i, _ in enumerate(feature_maxs):    
    # centers.append(np.arange(start=y[i], stop=x[i] + (x[i]-y[i])/(n_mfs-1),step=(x[i]-y[i])/(n_mfs-1)))
        centers.append(np.linspace(start=feature_mins[i], stop=feature_maxs[i], num=n_mfs, endpoint=True))
    centers = np.array(centers)
    #centers.ravel()
    centers = centers.ravel()
    return centers


def widths_init(feature_mins, feature_maxs, n_mfs):
    """Initializes the widths of MFs by partitioning the domain of the features

    Args:
        x (numpy.ndarray): the max values for each feature, i.e. the feature's domain
        n_mfs (int): number of MFs in Fuzzification Layer

    Returns: 
        numpy.ndarray: initalized widths with the shape (x.size,)
    """
    x = np.repeat(feature_maxs-feature_mins, n_mfs)

    
    return x/(2*(n_mfs+1))


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

def visuMFs(layer, dir, df_name, max_vals,min_vals, mf_names=["low", "middle", "high"]):
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
    plt.rc("font", family="serif")
    n_mfs = 3
    
    feature_names = max_vals.keys().values.tolist()
    
    c = np.array_split(layer.centers, len(feature_names)) # hc
   # print("HERE", c)
 
    w = np.array_split(layer.widths, len(feature_names)) # hc
 
    #print("max", max_vals)
   
   # print("c", c)
    #print("w", w)
  #  length_inputs = tf.shape(layer.centers)[0]
    # for each input see what the mfs mean -> domain input might be diff
    for xID, max_value in enumerate(max_vals):
    #for name, input in zip(names,layer.inputs):
#        x = np.arange(0, means[xID] , (means[xID]*0.01))
        x = np.arange(min_vals.iloc[xID], max_value, max_value/1000)
        y = {}

        for mfID in range(layer.n_mfs):
            y[mfID] = []

#            print(layer.centers[j+i*layer.n_mfs])
         #   for bleh in x:
              #  print(c[xID])
              #  print(c[xID][mfID])
            y[mfID] =layer.mf_type(x,c[xID][mfID],w[xID][mfID])

           # print("x", x)
            #print("y,", y[mfID])
            plt.plot(x, y[mfID], label=mf_names[mfID])

          #  print("hui", c[xID][mfID])
            plt.axvline(c[xID][mfID],0,1, c=plt.gca().lines[-1].get_color(), ls='--')


        # plt.legend( title="Fuzzy Labels",# bbox_to_anchor=( 1, 0.2), 
        #        fontsize=18,  title_fontsize=18)
        

        plt.legend(loc=(0.0, -0.4), title="Fuzzy Labels",# bbox_to_anchor=( 1, 0.2), 
                mode="expand", borderaxespad=0, ncol=3, fontsize=18,  title_fontsize=18)


        plt.title(f'MFs for {feature_names[xID]}', fontsize=20)
        plt.ylabel('µ', fontsize=18,  rotation='horizontal', ha='right')
        plt.xlabel(f' {feature_names[xID]} (cm)', fontsize=18)

        
        # get save path 
        file_name =  'MFs_' + feature_names[xID] + '.png'
        save_path = os.path.dirname(__file__) +  f'/../../results/{df_name}/figures/' + dir
        completeName = os.path.join(save_path, file_name)

        plt.savefig(completeName)
        plt.clf()