from scipy.special import log_softmax
import numpy as np

##Softmax Function used for selecting next action
def softmax(x,axis=None):
    """Compute softmax values for each sets of scores in x."""
    filtered_x = np.nan_to_num(x-x.max()) 
    return np.exp(log_softmax(filtered_x,axis=axis))