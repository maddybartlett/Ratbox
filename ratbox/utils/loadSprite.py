from pygame.image import load
import os

## Load images from asset folder for rendering
def load_sprite(name, with_alpha = True):
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    ## create path to image
    #path = f'assets\\{name}.png'
    path = os.path.join(__location__, f'assets\\{name}.png')
    loaded_sprite = load(path)

    ## convert image to a format that better fits the screen
    if with_alpha:
        return loaded_sprite.convert_alpha()
    else:
        return loaded_sprite.convert()