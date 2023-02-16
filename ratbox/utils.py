'''CUSTOM UTILS'''

from pygame.image import load
from scipy.special import log_softmax
import numpy as np

## Load images from asset folder for rendering
def load_sprite(name, with_alpha = True):
    ## create path to image
    path = f'ratbox/envs/assets/{name}.png'
    loaded_sprite = load(path)

    ## convert image to a format that better fits the screen
    if with_alpha:
        return loaded_sprite.convert_alpha()
    else:
        return loaded_sprite.convert()
    
##Softmax Function used for selecting next action
def softmax(x,axis=None):
    """Compute softmax values for each sets of scores in x."""
    filtered_x = np.nan_to_num(x-x.max()) 
    return np.exp(log_softmax(filtered_x,axis=axis))



