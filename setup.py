from setuptools import setup, find_packages

setup(
    name='multiview_manipulation',
    version='1.0.0',
    description='Package for Multiview Manipulation work published at IROS 2021.',
    author='Trevor Ablett',
    author_email='trevor.ablett@robotics.utias.utoronto.ca',
    license='MIT',
    packages=find_packages(),
    install_requires=['manipulator_learning @ git+https://git@github.com/utiasSTARS/manipulator_learning@master#egg=manipulator_learning',
                      'tensorflow-gpu==2.*',
                      'tensorflow-probability',
                      'numpy',
                      'scipy',
                      'gym',
                      'transforms3d',
                      'matplotlib']
)
