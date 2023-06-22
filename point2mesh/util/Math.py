'''
Created on Jul 25, 2022

@author: ggutow
'''
import numpy as np
from numba import njit

@njit
def singleton_clip(a,minimum,maximum):
    if a<minimum:
        return minimum
    if a>maximum:
        return maximum
    return a

@njit
def norm(x):
    '''
    norm(x) is a numba compiled 2-norm function for a (n,) numpy array
    
    @param x: (n,) np array
    @return float length
    
    on a (12,) vector it is approximately 10x as fast as np.linalg.norm
    '''
    return np.sqrt(np.sum(x*x))

@njit
def normsquared(x):
    return np.sum(x*x)

@njit
def massnorm(x):
    '''
    compute 2-norm for a collection of np array vectors (numba compiled)
    
    @param x: (m,...,n) np array
    @return (m,...,) numpy array of 2 norms taken along the last axis of x
    
    on 100x3 entry, it is >20x as fast as a list comprehension using norm(x)
    on a single entry, it is 3x slower than norm(x)
    this works with a (n,) array too!
    
    from https://github.gatech.edu/ggutow3/UncertainMotionPrimitives/blob/master/src/Util/Math.py
    '''
    return np.sqrt(np.sum(x*x,-1))
    
@njit
def stable_triangle_area(vertices):
    '''
    given the vertices of a triangle, compute its area robustly to numerical problems following a formulat given by W. Kahan in "Miscalculating Area and Angles of a Needle-like Triangle"

    Parameters: vertices : (3,n) float array
                    the vertices of the triangle
    '''
    one=norm(vertices[1]-vertices[0])
    two=norm(vertices[2]-vertices[0])
    three=norm(vertices[2]-vertices[1])
    if one<two:
        a=two
        b=one
    else:
        a=one
        b=two
    if b<three:
        c=b#b may not be two anymore
        b=three
        if a<b:
            temp=a
            a=b
            b=temp
    else:
        c=three
    return stable_triangle_area_from_edges(a,b,c)

@njit
def stable_triangle_area_from_edges(a,b,c):
    '''
    given non-increasing edge lengths a>=b>=c stably compute triangle area
    '''
    return np.sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4