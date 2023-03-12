import matplotlib.animation as animation
import matplotlib.pyplot as plt

def rend(env):
    '''
    Render rgb_array by plotting
    '''
    return plt.imshow(env.render())

def save_gifs(figure, images, filename, directory):
    '''
    Save render images as a gif.
    '''
    ani = animation.ArtistAnimation(figure, images, interval=50, blit=True,
                                    repeat_delay=1e5)
    writergif = animation.PillowWriter(fps=5)
    ani.save(directory+'\\'+filename+'.gif', writer=writergif)