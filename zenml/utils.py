import math


def is_nan(x):
    try:
        return math.isnan(float(x))
    except:
        return False
    
    
def values_equal(a, b):
    if a == b and a not in ('', None) and is_nan(a) == False:
        return True
    else:
        return False