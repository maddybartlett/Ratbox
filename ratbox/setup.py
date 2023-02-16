import os
from setuptools import setup

# Helper function to load up readme as long description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    
            name='ratbox', 
            version='0.0.1',  
            author = 'Maddy Bartlett, Nicole Dumont, Michael Furlong, Terry Stewart', 
            packages = [
            'steering',
            'envs',
        ],
            
            install_requires=[
                'Gymnasium==0.26.3', 
                'numpy==1.22.4', 
                'pygame==2.1.0', 
                'scipy==1.7.3', 
                'setuptools==67.1.0', 
] )