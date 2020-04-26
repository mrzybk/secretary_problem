# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:31:55 2020

@author: lackent
"""

def is_prime(x):
    n = 2
    if x < n:
        return False
    else:    
        while n < x:
            if x % n == 0:
                return False
                break
            n = n + 1
        else:
            return True