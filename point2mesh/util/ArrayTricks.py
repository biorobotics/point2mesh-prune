'''
Created on Jul 28, 2022

@author: ggutow
'''
import numpy as np
from numba import njit

def cartesian_product_simple_transpose(arrays):
    '''
    n-d cartesian product of an iterable of np arrays
    
    @param arrays: length la iterable of np arrays
    @returns cartesian product of the arrays (product of lengths,la)
    
    very fast, credit to senderle and Paul Panzer
    https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    '''
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

@njit
def index_to_point(index,shape):
    '''
    convert 1d flat index to nd array position. Inverse of point_to_index
    
    arguments: index, geometry of array
    '''
    stride=np.concatenate((np.cumprod(shape[-1:0:-1])[::-1],np.array([1],dtype=shape.dtype)))
    output=np.empty_like(shape)
    for i in range(len(shape)):
        output[i]=index//stride[i]
        index%=stride[i]
    return output

@njit
def point_to_index(point,shape):
    '''
    convert nd array position to 1d flat index. Inverse of index_to_point
    '''
    return (point[-1]+np.sum(point[:-1]*np.cumprod(shape[:0:-1])[::-1]))%np.prod(shape)

@njit
def threeD_multi_to_1D(xid,yid,zid,shape):
    '''
    convert indices into a 3D array into the 1D offset into the array (C-order)
    '''
    return zid+shape[-1]*(yid+shape[-2]*xid)