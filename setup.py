from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
setup(
    name='MUMPSpy',
    description='MUMPS for Python',
    long_description='A python wrapper for the parallel sparse direct solver.',
    version='1.0.0',
    url='https://github.com/vlukes/mumpspy',
    author='Vladimír Lukeš',
    author_email='vlukes@kme.zcu.cz',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',  # 3 - Alpha, 4 - Beta, 5 - Stable
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],

    keywords='sparse solver',
    packages=find_packages(),
    install_requires=['mpi4py'],
)
