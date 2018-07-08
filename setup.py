from setuptools import setup, find_packages


setup(
    name='minuet',
    version='0.0.1',
    author='Luiz Felix',
    author_email='lzcfelix@gmail.com',
    description=('Library to perform NLP sequence tagging tasks.'),
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
        'numpy==1.13.3',
        'gensim',
        'sklearn',
        'h5py',
        'keras==2.2.0',
        'tensorflow==1.8.0',
    ],
    dependency_links=[
        'git@github.com:keras-team/keras-contrib.git',
    ]
)