from numba import cuda
def njit(foo=None,*,parallel=False):
    def dec(func):
        return func
    if foo is None:
        return dec
    else:
        return foo
def List(bar=None):
    if bar is None:
        return list()
    return bar

def prange(first,second=None):
    if second is not None:
        return range(first,second)
    else:
        return range(first)