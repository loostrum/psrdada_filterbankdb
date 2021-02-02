#!/usr/bin/env python3

import os
import glob
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open(os.path.join('dada_fildb', '__version__.py')) as version_file:
    version = {}
    exec(version_file.read(), version)
    project_version = version['__version__']

setup(name='dada_fildb',
      version=project_version,
      description='Read multibeam filterbank data into a PSRDADA ringbuffer',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='http://github.com/loostrum/psrdada_filterbankdb',
      author='Leon Oostrum',
      author_email='l.oostrum@esciencecenter.nl',
      license='Apache2.0',
      packages=find_packages(),
      install_requires=['numpy>=1.17',
                        'astropy'],
      entry_points={'console_scripts':
                    ['dada_fildb=dada_fildb.dada_fildb:main']},
      classifiers=['License :: OSI Approved :: Apache Software License',
                   'Programming Language :: Python :: 3',
                   'Operating System :: OS Independent'],
      python_requires='>=3.6'
      )
