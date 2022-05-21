# -*- coding: utf-8 -*-
"""
myAnalisys

Library of functions for analizyng sound (as np.array)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import copy

def ceiledPow(x):
    """
    

    Parameters
    ----------
    x : np.array
        array to .

    Returns
    -------
    int
        Nearest power of 2 rounded to the ceil.

    """
    return 2**math.ceil(math.log(x,2))


def normalizza(Data):
    """
    

    Parameters
    ----------
    Data : np.array
        array to normalize.

    Returns
    -------
    normalized array within 0. and 1.

    """
    return(Data / max(abs(Data)))


def peakFind(Data, delta):
    """
    

    Parameters
    ----------
    Arr : np.array
        Array where to find the peaks
    delta : threshold

    Modifies
    --------
    Arr : if a value is < of delta, this value is set to 0
    
    Returns
    -------
    I : array of Index of the peaks

    """
    Arr = copy.deepcopy(Data)
    
    for i in range(len(Arr)):
        if Arr[i] < delta: #thresholding
            Arr[i] = 0
    
    Arr = np.concatenate(([0],Arr,[0])) # Add 0 in front and to the end of Arr to simplify conditions
    I = np.array([])
    
    for i in range(1,len(Arr)-1):
            if (Arr[i] > Arr[i+1]) and (Arr[i] > Arr[i-1]):
                    I = np.append(I,i-1)
    
    return I

