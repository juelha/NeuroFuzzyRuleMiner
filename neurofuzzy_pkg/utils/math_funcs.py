import numpy as np


"""
Collection of useful math functions to reduce the amount of needed pkgs
"""

def coefficient(n,k):
    """
    using Multiplicative formula
    """
    c = 1.0
    for i in range(1, k+1):
        c *= float((n+1-i))/float(i)
    return c


def mean(lst):
    return sum(lst) / len(lst)


def sigmoid(x): 
    return 1 / (1 + np.exp(-x))


def sigmoidprime(x):
    return sigmoid(x)*(1-sigmoid(x))


def normalize(mus):
    """Normalizing a vector of mus

    Args:
        mus (list(float)): mus calculated by applying MFs to an input
    """
    mus_normalized = []

    if sum(mus) == 0.0:
        return mus

    assert sum(mus) != 0.0, f'Sum of vector can never be zero, input: {mus}'
    for mu in mus:
       mu /= sum(mus)
       mus_normalized.append(mu)
    return mus_normalized
