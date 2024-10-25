from setuptools import setup, find_packages

setup(
    name='ImmunoAlign',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pytorch-lightning',
        # add other dependencies here
    ],
)