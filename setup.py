import os
import datetime as dt

from setuptools import setup, find_packages, find_namespace_packages
from sentinel2 import __version__

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='gdal-boots',
    version=__version__,
    author='Alexander Verbitsky',
    author_email='habibutsu@gmail.com',
    maintainer='Alexander Verbitsky',
    maintainer_email='habibutsu@gmail.com',
    description='python friendly wrapper over GDAL',
    long_description=read('readme.md'),
    keywords='GDAL, raster, spatial',
    url='https://github.com/gdal-wrapper',
    packages=find_packages() + find_namespace_packages(include=['gdal_boots.*']),
    test_suite='test',
    python_requires='>=3.6',
    install_requires=(
        'GDAL',
        'affine',
    ),
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Utilities',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
    ],
)