from pygame.image import load

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