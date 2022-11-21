from setuptools import setup, find_packages


setup(
    name='minuet',
    version='0.0.2',
    author='Luiz Felix',
    author_email='lzcfelix@gmail.com',
    description=('An easy to use library to perform NLP sequence tagging tasks'),
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
        'numpy==1.13.3',
        'gensim',
        'sklearn',
        'h5py',
        'keras==2.2.0',
        'tensorflow==2.9.3',
        'cloudpickle==0.5.3',
    ],
    dependency_links=[
        'git@github.com:keras-team/keras-contrib.git',
    ]
)

