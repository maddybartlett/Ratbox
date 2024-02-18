import os
from setuptools import setup, find_packages

# Helper function to load up readme as long description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    
            name='ratbox', 
            version='1.0.0',  
            author = 'Maddy Bartlett, Nicole Dumont, Michael Furlong, Terry Stewart', 
            packages=find_packages(),
            include_package_data=True,
            package_data = {
                "": ["*.png"]
            },
            
            install_requires=[
                'Gymnasium', 
                'numpy', 
                'pygame', 
                'scipy', 
                'setuptools', 
] )