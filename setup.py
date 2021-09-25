import os
import re

from setuptools import find_namespace_packages, find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as fd:
        return fd.read()


def get_version():
    with open(os.path.join('gdal_boots', '__init__.py')) as f:
        return re.search(r'__version__ = ["\'](.*?)[\'"]', f.read()).group(1)


setup(
    name='gdal-boots',
    version=get_version(),
    author='Alexander Verbitsky',
    author_email='habibutsu@gmail.com',
    maintainer='Alexander Verbitsky',
    maintainer_email='habibutsu@gmail.com',
    description='python friendly wrapper over GDAL',
    long_description=read('readme.md'),
    keywords='GDAL, raster, spatial',
    url='https://github.com/habibutsu/gdal-boots',
    packages=find_packages() + find_namespace_packages(include=['gdal_boots.*']),
    test_suite='test',
    python_requires='>=3.6',
    install_requires=tuple(read('requirements.txt').split()),
    extras_require={
        'test': read('requirements-dev.txt').split(),
    },
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Utilities',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
    ],
)
