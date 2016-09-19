#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Sep 19, 2016

@author: Dirk Toewe
'''

from distutils.core import setup

setup(
  name='SLeEPy',
  version='1.0',
  description='Pure Python extension for Scikit-Learn.',
  author='Dirk Toewe',
  url='https://github.com/DirkToewe/sleepy',
  package_dir = {'': 'src'},
  packages=[
    'sleepy',
    'sleepy.regression',
    'sleepy.regression.nonlinear'
  ]
)