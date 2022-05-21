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

def FFT(Data):
    """
    

    Parameters
    ----------
    Data : np.array
        Audio to analyse in fft.

    Returns
    -------
    X_mod : np.array
        FFT magnitude array normalized in [0, 1] range

    """
    pad = np.zeros(ceiledPow(len(Data))-len(Data))

    X_pad = np.append(Data,pad)



    #FFT
    X_fft = np.fft.rfft(X_pad)
    X_mod = normalizza(np.abs(X_fft))
    
    return X_mod


def quadPeakFind(Data, delta, sr):
    """
    

    Parameters
    ----------
    Data : np.array
        FFT array where to find the peaks
    
    delta : threshold
    
    sr :
        Sampling rate of the FFT array

    
    Returns
    -------
    F : 2D np.array
        2D array of the peaks [frequency, amplitude]

    """
    Arr = copy.deepcopy(Data)
    
    for i in range(len(Arr)):
        if Arr[i] < delta: #thresholding
            Arr[i] = 0
    
    Arr = np.concatenate(([0],Arr,[0])) # Add 0 in front and to the end of Arr to simplify conditions
    I = np.array([])
    F = np.array([])
    A = np.array([])
    
    #finding local max
    for i in range(1,len(Arr)-1):
            if (Arr[i] > Arr[i+1]) and (Arr[i] > Arr[i-1]):
                    I = np.append(I,i-1)
    
    
    #Quadratic interpolation for finding frequency
    for i in range(len(I)):
        j = int(I[i])
        val = Data[j]
        lval = Data[j-1]
        rval = Data[j+1]        

        
        k = val+(0.5*(lval-rval)/(lval-2*val+rval))    #frame interpolation

        f = (sr*k)/len(Data)                           #converting frame into frequency
        a = val-0.25*(lval-rval)*(k-val)               #calculating amplitude
        
        #toAdd = np.array([f,a])
        
        F = np.append(F,f)
        A = np.append(A,a)
    
    ToReturn = np.stack((F,A),axis=1)
    
    #print(ToReturn)
        
    return ToReturn

