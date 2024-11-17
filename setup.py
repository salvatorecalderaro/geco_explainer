# setup.py

from setuptools import setup, find_packages

setup(
    name='geco_explainer',
    version='0.1.0',
    description='GECo method to explain GNNs.',
    author='Salvatore Calderaro',
    author_email='salvatore.calderaro01@unipa.it',
    packages=find_packages(),
    install_requires=[
        'networkx',
        'torch',
        'numpy',
        'cdlib',
        'torch_geometric',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
