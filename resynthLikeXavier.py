#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 00:09:44 2022

@author: matteo
"""

# LETTURA E SEMPLICE ELABORAZIONE DI UN FILE WAV

########## NON MODIFICARE #######
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math
import copy

# USARE MONO 16 BIT WAV !!!!!
sr, X_int = wavfile.read('in.wav')

print ("sr :", sr)

# trasforma in formato "normale" tra -1.0 e 1.0
X = X_int/32768

print('max abs amp input:', np.amax(abs(X)))

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

#  ESEMPI #######################

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

# FINE ESEMPI ################################ 


# DEFINZIONE DELLE VOSTRE FUNZIONI


def ceiledPow(x):
    """
    

    Parameters
    ----------
    x : int

    Returns
    -------
    int
        Nearest power of 2 rounded to the ceil.

    """
    return 2**math.ceil(math.log(x,2))



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
    
    Arr = copy.deepcopy(Data)
    
    pad = np.zeros(ceiledPow(len(Arr))-len(Arr))

    X_pad = np.append(Arr,pad)



    #FFT
    X_fft = np.fft.fft(X_pad)
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

        
        k = val+(0.5*(lval-rval)/(lval-(2*val)+rval))    #frame interpolation


        f = (k*sr)#(sr*k)/len(Data)            #converting frame into frequency
        a = val-0.25*(lval-rval)*(j-val)       #calculating amplitude
        
        #toAdd = np.array([f,a])
        
        F = np.append(F,f)
        A = np.append(A,a)
        
    ToReturn = np.stack((F,A),axis=1)
    
    #print(ToReturn)
        
    return ToReturn

def peakDetection(mX, t):
    #detect spectral peak locations
    #mx: magnitude spectrum
    #t: treshold
    #returns ploc: peak locations
    
    tresh = np.where(mX[1:-1]>t, mX[1:-1], 0);              #locations above treshold
    next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     #locations higher than the next one
    prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    #locations higher than the previous one
    ploc = tresh*next_minor*prev_minor                      #locaions with all the 3 criterias above
    ploc = ploc.nonzero()[0] + 1                            #compensation
    return ploc

def peakInterp(mX,ploc):
    val = mX[ploc]
    lval = mX[ploc-1]
    rval = mX[ploc+1]
    iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)
    ipmag = val - 0.25*(lval-rval)*(iploc-ploc)
    return iploc, ipmag
    

#################################
# CHIAMATE LE VOSTRE FUNZIONI


N = ceiledPow(len(X))
t = 0.1

pad = np.zeros(ceiledPow(len(X))-len(X))
X_pad = np.append(X,pad)
fftX = np.fft.fft(X_pad)
mX = abs(fftX)
mXnorm = normalizza(mX)

ploc = peakDetection(mXnorm,t)

iploc, ipmag = peakInterp(mXnorm, ploc)

freqaxis = sr*np.arange(N/2)/float(N)
plt.plot(freqaxis[:int(len(freqaxis))], mXnorm[:int((N/2))])
plt.plot(sr*iploc[:int(len(iploc)/(2))]/float(N), ipmag[:int(len(ipmag)/2)], marker='x', linestyle = '')

plt.show()

NonIModes = np.array([sr*ploc[:int(len(ploc)/2)]/float(N),normalizza(ipmag[:int(len(ipmag)/2)])]);

Modes = np.array([sr*iploc[:int(len(iploc)/2)]/float(N),normalizza(ipmag[:int(len(ipmag)/2)])]);

####### NON MODIFICARE ###########
# NORMALIZZAZIONE ... paracadute
#Y = normalizza(X)

#  stampa la massima ampiezza in valore assoluto
#print('max abs amp output:', np.amax(abs(Y)))

"""
# trasforma in 16 bit (opzionale, se non la metto salva a 32 bit float)
Y_int = np.round(Y * 32768)
Y_int = Y_int.astype(int)
wavfile.write("out.wav", sr, Y_int)
"""

# scrivi File su disco float 32
#wavfile.write("out.wav", sr, Y)

# disegna output (va posizionato qui altrimenti matplot lib blocca la prosecuzione in repl)
#disegna(Y, sr)

#################################