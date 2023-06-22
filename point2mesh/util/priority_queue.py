'''
Provides functions that implement a numba-compatible min-heap based priority queue
'''

import heapq
from numba import njit
from collections import namedtuple

PriorityQueue=namedtuple("PriorityQueue",["heap","counter","entries"])
@njit
def make_queue(first_item,first_priority):
    heap=[(first_priority,0,first_item)]
    heapq.heapify(heap)
    counter=1
    return PriorityQueue(heap,counter,1)
@njit
def add_item(pq,item,priority):
    heapq.heappush(pq.heap,(priority,pq.counter,item))
    return PriorityQueue(pq.heap,pq.counter+1,pq.entries+1)
@njit
def get_item(pq):
    priority,_,item=heapq.heappop(pq.heap)
    return PriorityQueue(pq.heap,pq.counter,pq.entries-1),priority,item
@njit
def is_empty(pq):
    return pq.entries==0
