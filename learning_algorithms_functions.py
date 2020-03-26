# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:31:55 2020

@author: lackent
"""

import numpy as np

def naive_recruit(b,p):
    '''
    '''
    m, L=len(b), len(b)
    for i in range(L):
        if i<p*L:
            if b[i]<m:
                m=b[i]
        else:
            if b[i]<m:
                return i
    return -1
