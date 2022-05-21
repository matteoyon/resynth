#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:09:08 2022

@author: matteoyon

"""

"""
Main steps:
    Record
    Analyze
    Model
    Resynth
    
"""

"""
Problema: non capisco perchè (ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all())
"""
import myUtilities as myU
import myAnalisys as myA

"""
def main():
"""  
sr, X = myU.readFile('in.wav')

Y = myA.FFT(X)
Y = myA.quadPeakFind(Y, 0.01, sr)

myU.writeFile(X, sr)
    
"""    
    return

if __name__ == "__main__":
    main()
    """