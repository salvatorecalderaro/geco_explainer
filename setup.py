from setuptools import setup, find_packages

# Legge il file README.md per la descrizione su PyPI
with open("description.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='geco_explainer',
    version='0.2.3',
    description='GECo method to explain GNNs.',
    long_description=long_description,  # 👈 aggiungi questa riga
    long_description_content_type="text/markdown",  # 👈 e questa
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