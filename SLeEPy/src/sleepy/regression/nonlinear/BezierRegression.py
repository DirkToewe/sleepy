#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Dirk Toewe
#
# This file is part of SLeEPy.
#
# Game of Pyth is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Game of Pyth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Game of Pyth. If not, see <http://www.gnu.org/licenses/>.

from numpy import amax, amin, identity as I, einsum, isinf, newaxis, power as pow
from numpy.linalg import LinAlgError, lstsq
from scipy.special import binom as binomCoef
from sklearn.base import RegressorMixin

import numpy as np
import numpy.linalg as la


def _factors(x,size):
  rows = np.arange(size)[:,newaxis]
  return (
    binomCoef(size-1,rows)
    * pow( (1+x)/2, rows[::+1] )
    * pow( (1-x)/2, rows[::-1] )
  )

class BezierRegression(RegressorMixin):
  '''
  Regularized least squares Bezier Regression. Class is designed to be used
  as a Regression model in Scikit-Learn.

  Parameters
  ----------
  coef_shape : tuple of n_feature ints
    Determines the shape of the Bezier point matrix along each feature axis.

  λ : float, optional, default 0
    Regularization parameter used during least square fitting
    the Bezier Curve/(Hyper-)Surface. `λ`=0 means no Regularization.
    `λ`→∞ means no fitting (_coefs=0). See: https://youtu.be/I-VfYXzC5ro
    or https://en.wikipedia.org/wiki/Regularization_(Mathematics).

  Attributes
  ----------
  _coefs : array, shape = (n_targets,*coef_shape)
    Estimated coefficients for the linear Bezier regression problem.

  n_features: int
    Number of inputs.

  n_targets: int
    Number of outputs. 
  '''

  def __init__(self, coef_shape, λ=0 ):
    assert 0 <= λ
    self.n_features = len(coef_shape)
    self.coef_shape = coef_shape
    self.λ = λ

  def fit( self, X, y ):
    '''
    Fit Bezier model.

    Parameters
    ----------
    X : numpy array, shape = (n_samples, n_features)
        Training data

    y : numpy array, shape = (n_samples, n_targets)
        Target values

    Returns
    -------
    self : returns an instance of self.
    '''
    assert 2 == X.ndim
    assert 2 == y.ndim
    assert self.n_features == X.shape[1]
    self.n_targets  = y.shape[1]
    n_samples  = X.shape[0]
    assert X.shape[0] == y.shape[0]

    # set X to be between -1 and +1
    _min,_max = ( f(X,axis=0) for f in (amin,amax) )
    self._X_off   = 0.5 * (_max +_min)
    self._X_scale = 2.0 / (_max -_min)
    X = X - self._X_off # <- safety copy!
    X *= self._X_scale

    # set y to be between -1 and +1
    _min,_max = ( f(y,axis=0) for f in (amin,amax) )
    self._y_off   = (_max +_min) * 0.5
    self._y_scale = (_max -_min) / 2.0
    y = y - self._y_off # <- safety copy!
    y /= self._y_scale
 
    # put all Bezier factors into (?,n_targets) shape.
    a = np.array([[1]])
    for col in range(self.n_features):
      a = a[:,newaxis,:] * _factors( X[:,col], self.coef_shape[col] ).T[:,:,newaxis]
      a = a.reshape(( n_samples, -1 )) # <- -1 means the reshape() infers that remaining axis length
    # solve least square error coefficients
    assert 0 <= self.λ
    if isinf(self.λ):
      self._coefs = np.zeros(( self.n_targets, *self.coef_shape[::-1] ))
    else:
      if 0 == self.λ:
        coefs, *_ = lstsq(a,y)
      else:
        # see: https://youtu.be/I-VfYXzC5ro?t=2100
        coefs = la.solve(
          a.T @ a  +  self.λ * I(a.shape[1]),
          a.T @ y
        )
      self._coefs = coefs.T.reshape(( self.n_targets, *self.coef_shape[::-1] ))

    return self

  def predict(self,X):
    '''
    Predict using the Bezier model.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        Samples.

    Returns
    -------
    C : array, shape = (n_samples, n_targets)
        Returns predicted values.
    '''
    assert X.ndim == 2
    assert X.shape[1] == self.n_features
    X = X - self._X_off # <- safety copy!
    X *= self._X_scale
    result = self._coefs[newaxis,...]
    for col in range(self.n_features):                                                        #       # the following code could be faster than the einsum but is a lot uglier (TODO: benchmark!)
      result = einsum( 'j...k,kj->j...', result, _factors( X[:,col], self.coef_shape[col] ) ) #       result = (
    return self._y_off + self._y_scale * result                                               #         result @ _factors( X[:,col], self.coef_shape[col] )
                                                                                              #           .reshape(len(X),*(1 for _ in range(col+1,self.n_features)),self.coef_shape[col],1)
  @property                                                                                   #       ).squeeze( axis=-1 )
  def jac(self):
    '''
    Returns
    -------
    numpy array, shape = (n_samples, n_targets, n_features) :
      The jacobi matrix of the Bezier model for the given samples.
    '''
    # TODO: consider using weakref
    if hasattr(self,'_jac'):
      return self._jac
    def partDerivs():
      for feature in range(self.n_features):
        cShape = np.array(self.coef_shape)
        cShape[feature] -= 1
        if 0 == cShape[feature]:
          yield lambda X: np.zeros(( len(X), self.n_targets ))
        else:
          partDeriv = BezierRegression( coef_shape=tuple(cShape) )
          # https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Derivative
          partDeriv._coefs = (
              np.delete(self._coefs,  0, axis=-1-feature)
            - np.delete(self._coefs, -1, axis=-1-feature)
          ) * (
            # keep in mind that we have substituted t by (x-1)/2. d( f((x-1)/2) )/dx = 0.5*f'( f((x-1)/2) )
            0.5 * cShape[feature] * self._X_scale[feature]
          )
          partDeriv._y_off = np.zeros(self.n_targets) # <- _y_offset does not influence derivative
          for k,v in self.__dict__.items():
            if k not in partDeriv.__dict__:
              partDeriv.__dict__[k] = v
          yield partDeriv.predict
    partDerivs = tuple( partDerivs() )
    self._jac = lambda X: np.concatenate( [ f(X)[:,:,newaxis] for f in partDerivs ], axis=-1 )
    return self._jac