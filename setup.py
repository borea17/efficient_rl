from distutils.core import setup
from setuptools import find_packages

INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'networkx',
    'gym',
    'matplotlib',
    'prettytable'
]

setup(
    name='efficient_rl',
    version='1.0',
    license='MIT',
    description='Reimplementation of Diuks "object-oriented representation for efficient RL".',
    author='Markus Borea',
    author_emal='borea17@protonmail.com',
    url='https://github.com/borea17/efficient_rl',
    download_url='https://github.com/borea17/efficient_rl/archive/v_1.0.tar.gz',
    keywords=['Reinforcement Learning', 'Efficient RL', 'OO-MDP'],
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
