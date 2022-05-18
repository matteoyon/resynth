#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:22:04 2022

@author: matteo
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def readFile(path):
    """
    

    Parameters
    ----------
    path : string
        path to the file to load.

    Returns
    -------
    sr : int
        Sample Rate of the file.
    X : np.array
        representation of the audio file in values between 0. and 1.

    """
    
    # USARE MONO 16 BIT WAV !!!!!
    sr, X_int = wavfile.read(path)

    print ("sr :", sr)

    # trasforma in formato "normale" tra -1.0 e 1.0
    X = X_int/32768

    #print('max abs amp input:', max(abs(X)))
    
    return sr, X


####### UTILITA' VARIE #################
def ms2smpls(t_ms, sr):
  """
    Input:
        t_ms:  tempo in ms

    Output:
        numero di campioni equivalenti
  """
  return round(sr * t_ms/1000)

def normalizza(Data):

    return(Data / max(abs(Data)))
  
         
def disegna(Data, sr):
    """
    Input:
        myArray: np array a 64 bit 
        
    Output:
        disegna il grafico dell'array
    """
    numSmpls = len(Data)    
    
    #sintassi linspace start, stop, numero di punti
    # genera asse dei tempi
    Time = np.linspace(0, numSmpls/sr, numSmpls)
    
    # crea in memoria il disegno con asse x Time ed y myArray
    plt.plot(Time, Data)
    
    # aggiungi griglia
    plt.grid()
    
    # mostra video il disegno
    plt.show()
    
    return

#  Utilities varie #######################

def lowPass(DataIn, sr, fc):
    
    """
	I° order low pass filter
	
    Inputs:
        DataIn: input numpy array
        sr: sampling rate
        fc: filter frequency cut (@ -3dB)
    Output:
        DataOut: filter output
    """
    # init output buffer
    DataOut = np.zeros(len(DataIn))
    
    # filter coefs - Pirkle pag 165
    thetaC = 2 *  np.pi*fc / sr
    gamma = 2 - np.cos(thetaC)
    b1 = np.sqrt(gamma ** 2 - 1) - gamma 
    a0 = 1 + b1
    
    # init memories
    y1 = 0

    for i in range(len(DataIn)):       
        x = DataIn[i]
        
        # eq LP filter    
        # y[n] = a0 * x[n]  - b1 * y[n-1] 
        y = a0 * x  - b1 * y1 

        # update memories
        y1 = y
 
        DataOut[i] = y
        
    return DataOut
    

def comb(DataIn, sr, delayMs, gain, delayMaxMs, combType):
    
    """
    COMB FILTER FIR / IIR
	
    Inputs:
        DataIn: input numpy array
        sr: sampling rate
        delayMs: gain in ms
        gain: reflection gain
        delayMaxMs: max delay line length line
        combType: comb FIR / IIR  [0, 1] 
    
    Output:
        DataOut: comb filter output
	"""
    
    if (delayMs > delayMaxMs):
        print("requested delay > max delay")
        return(DataIn)

    # init output buffer
    DataOut = np.zeros(len(DataIn))
    	
	# from ms to samples
    delaySmpls = ms2smpls(delayMs, sr) 
    delayMaxSmpls = ms2smpls(delayMaxMs, sr) 
   
    # init delay line
    tmpBuf = np.zeros(delayMaxSmpls) 

    # loop on input samples
    for i in range(len(DataIn)):
       
		# input
        x = DataIn[i]
        
        # comb 
        y = x + gain * tmpBuf[-delaySmpls]
        
		# output
        DataOut[i] = y

        # insert sample into the delay line (FIFO)
        if (combType == 0): 
            # COMB FIR    
            tmpBuf = np.append(tmpBuf, x) # insert input into the delay line (feedforward)
        else:
            # COMB IIR
            tmpBuf = np.append(tmpBuf, y) # insert output into the delay line (feedback)
      
        # delete sample from the delay line (FIFO)
        tmpBuf =  np.delete(tmpBuf, 0)

    return(DataOut)



def writeFile(X,sr):
    """
    

    Parameters
    ----------
    X : np.array
        audio representation to write in audio file
    sr : int
        Sample Rate.

    Returns
    -------
    None.

    """
    
    ####### NON MODIFICARE ###########
    # NORMALIZZAZIONE ... paracadute
    Y = normalizza(X)
    
    #  stampa la massima ampiezza in valore assoluto
    print('max abs amp output:', max(abs(Y)))
    
    """
    # trasforma in 16 bit (opzionale, se non la metto salva a 32 bit float)
    Y_int = np.round(Y * 32768)
    Y_int = Y_int.astype(int)
    wavfile.write("out.wav", sr, Y_int)
    """
    
    # scrivi File su disco float 32
    wavfile.write("out.wav", sr, Y)
    
    # disegna output (va posizionato qui altrimenti matplot lib blocca la prosecuzione in repl)
    disegna(Y, sr)
    return

