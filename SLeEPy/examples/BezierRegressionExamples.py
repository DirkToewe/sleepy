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

'''
A collection of visualized experiments/examples used for validation and
understanding of the BezierRegression.

Created on Sep 16, 2016

@author: Dirk Toewe
'''
from functools import partial
from itertools import chain
from math import cos

from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy import NaN, logical_not, isnan, pi, vstack, sin, newaxis, double
import plotly

from sleepy.regression.nonlinear import BezierRegression
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go


def fitCurve():
  X = np.linspace(-0.75, +0.9, 21, dtype=double)
  y = np.fromiter( map( lambda x: (x+0.7)*(x+0.3)*x*(x-0.5)*(x-0.8), X ), dtype=double )

  input = go.Scatter(
    x = X, y = y,
#     mode='lines', line = dict(color='#FF0000', width=2),
    mode='markers', marker = dict(color='#0000FF', size=16),
  )

  bez = BezierRegression( coef_shape=(16,), λ = 999e12 )
  bez.fit(X[:,newaxis], y[:,newaxis])

  y_fit = np.fromiter( map( lambda x: bez.predict(np.array([[x]])), X ), dtype=double )

  regression = go.Scatter(
    x = X, y = y_fit,
    mode='lines', line = dict(color='#FF0000', width=2)
  )

  layout = go.Layout(
    title='Fit Curve',
    scene = dict(
      aspectratio = dict(x=1, y=1, z=1),
      aspectmode = 'data'
    )
  )
  fig = go.Figure(
    data = [input,regression],
    layout = layout
  )
  plotly.offline.plot(fig,filename='fit_curve.html')

def fitSurface():
  uRange = np.linspace(-0.75,+0.9, 11)
  vRange = np.linspace(-0.9, +0.75,11)

  input = scatterSurface3d(
    uRange, vRange,
    lambda u,v: ( u,v, u**3 * v**1 ),
    mode='markers', marker = dict(color='#0000FF', size=2),
  )

  not_NaN = logical_not(isnan(input.x))
  X = np.column_stack((
    input.x,
    input.y
  ))[not_NaN,:]

  y = np.column_stack((
    input.x,
    input.y,
    input.z
  ))[not_NaN,:]
  bez = BezierRegression( coef_shape = [4,2] )
  bez.fit(X,y)
  regression = scatterSurface3d(
    uRange, vRange,
    lambda *uv: bez.predict( np.array([uv]) ),
    mode='lines', line = dict(color='#FF0000', width=2)
  )

  layout = go.Layout(
    title='Fit Surface',
    scene = dict(
      aspectratio = dict(x=1, y=1, z=1),
      aspectmode = 'data'
    )
  )
  fig = go.Figure(
    data = [input,regression],
    layout = layout
  )
  plotly.offline.plot(fig,filename='fit_surface.html')

def fitHeighmap():
  uRange = np.linspace(-0.75,+0.9, 11)
  vRange = np.linspace(-0.9, +0.75,11)

  input = scatterSurface3d(
    uRange, vRange,
    lambda u,v: ( u,v, u**3 * v**1 ),
    mode='markers', marker = dict(color='#0000FF', size=2),
  )

  not_NaN = logical_not(isnan(input.x))
  X = np.column_stack((
    input.x,
    input.y
  ))[not_NaN,:]

  y = input.z[not_NaN,newaxis]
  bez = BezierRegression( coef_shape = [4,2] )
  bez.fit(X,y)
  regression = scatterSurface3d(
    uRange, vRange,
    lambda u,v: (u,v,bez.predict( np.array([[u,v]]) )),
    mode='lines', line = dict(color='#FF0000', width=2)
  )

  layout = go.Layout(
    title='Fit Heightmap',
    scene = dict(
      aspectratio = dict(x=1, y=1, z=1),
      aspectmode = 'data'
    )
  )
  fig = go.Figure(
    data = [input,regression],
    layout = layout
  )
  plotly.offline.plot(fig,filename='fit_heightmap.html')

def fitVolume():
  uRange = np.linspace(-0.75,+0.9, 11)
  vRange = np.linspace(-0.9, +0.75,21)
  wRange = np.linspace(-2,   +2,    6)

  def features():
    for u in uRange:
      for v in vRange:
        for w in wRange:
          yield u,v,w

  def targets():
    for u,v,w in features():
      x = u
      y = v + sin(u*2*pi)
      z = w + x*y
      yield x,y,z

#   X = column_stack((uRange,vRange,wRange))
  X = vstack(features())
  y = vstack(targets())

  data = [
    go.Scatter3d(
      x = y[:,0], y = y[:,1], z = y[:,2],
      mode='markers', marker = dict(color='#0000FF', size=2)
    )
  ]

  bez = BezierRegression( coef_shape=[7,3,2] )
  bez.fit(X,y)

  for w in wRange:
    data.append(
      scatterSurface3d(
        uRange, vRange,
        lambda u,v: bez.predict( np.array([[u,v,w]]) ),
        text = partial( 'u: {:.3f} v: {:.3f} w: {w:.3f}'.format, w=w ),
        mode='lines', line = dict(color='#FF0000', width=2)
      )
    )

  layout = go.Layout(
    title='Fit Volume',
    scene = dict(
      aspectratio = dict(x=1, y=1, z=1),
      aspectmode = 'data'
    )
  )
  fig = go.Figure(
    data = data,
    layout = layout
  )
  plotly.offline.plot(fig,filename='fit_volume.html')

def quiverCurve():
  X = np.linspace(-10, +10, 128, dtype=double)
  y = np.vstack( map( lambda x: (cos(x),sin(x),x/10), X ) )

  bez = BezierRegression( coef_shape=(16,) )
  bez.fit(X[:,newaxis], y)
 
  y_fit = np.vstack( map( lambda x: bez.predict(np.array([[x]])), X ) )
  dir = np.vstack( map( lambda x: bez.jac(np.array([[x]])), X ) )[:,:,0]

  fig = plt.figure( tight_layout=True )
  ax3d = fig.add_subplot(1,1,1, projection='3d')
  ax3d.scatter( *( y[:,i] for i in range(3) ) )
  ax3d.quiver(
    *chain(
      ( y_fit[:,i] for i in range(3) ),
      (   dir[:,i] for i in range(3) )
    ),
    pivot='tail', length=0.1, color='red', arrow_length_ratio = 0.3
  )

def quiverSurface():
  uRange = np.linspace(-0.75,+0.9, 11)
  vRange = np.linspace(-0.9, +0.75,11)

  input = scatterSurface3d(
    uRange, vRange,
    lambda u,v: ( u,v, u**3 * v**1 ),
    mode='markers', marker = dict(color='#0000FF', size=2),
  )

  not_NaN = logical_not(isnan(input.x))
  X = np.column_stack((
    input.x,
    input.y
  ))[not_NaN,:]

  y = np.column_stack((
    input.x,
    input.y,
    input.z
  ))[not_NaN,:]
  bez = BezierRegression( coef_shape = [4,2] )
  bez.fit(X,y)

  y_fit = bez.predict(X)
  tangents = bez.jac(X)
  assert np.isclose(
    tangents,
    np.vstack( bez.jac(row[newaxis,:]) for row in X )
  ).all()
  normals = np.cross( tangents[:,:,0], tangents[:,:,1] )

  fig = plt.figure( tight_layout=True )
  ax3d = fig.add_subplot(1,1,1, projection='3d')
  ax3d.scatter( *( y    [:,i] for i in range(3) ) )
#   ax3d.plot   ( *( y_fit[:,i] for i in range(3) ), color='red' )
  # PLOT NORMALS
  ax3d.quiver(
    *chain(
      (  y_fit[:,i] for i in range(3) ),
      (normals[:,i] for i in range(3) )
    ),
    pivot='tail', length=0.1, color='red', arrow_length_ratio = 0.3
  )
  # PLOT TANGENT ARROWS
  for j in range(2):
    ax3d.quiver(
      *chain(
        (   y_fit[:,i]   for i in range(3) ),
        (tangents[:,i,j] for i in range(3) )
      ),
      pivot='tail', length=0.1, color='red', arrow_length_ratio = 0.3
    )

def scatterSurface3d( uRange, vRange, f, text = 'u: {:.3f} v: {:.3f}'.format, **kwargs ):
  for style in ['line','marker']:
    if style in kwargs:
      style = kwargs[style]
      if 'color' in style:
        color = style['color']
        if callable(color):

          def color_gen():
            for u in uRange:
              yield NaN # <- NaN functions as a 'line break'
              for v in vRange: yield color(u,v)
            for v in vRange:
              yield NaN
              for u in uRange: yield color(u,v)
  
          color_gen = color_gen()
          next(color_gen)
          style['color'] = tuple(color_gen)

  def data_gen():
    for u in uRange:
      yield NaN,NaN,NaN # <- NaN functions as a 'line break'
      for v in vRange: yield f(u,v)
    for v in vRange:
      yield NaN,NaN,NaN
      for u in uRange: yield f(u,v)

  data = data_gen()
  next(data)
  data = np.vstack(data)

  if callable(text):
    def text_gen():
      for u in uRange:
        yield '???'
        for v in vRange: yield text(u,v)
      for v in vRange:
        yield '???'
        for u in uRange: yield text(u,v)
  
    txt = text_gen()
    next(txt)
    text = tuple(txt)

  return go.Scatter3d(
    x = data[:,0], y = data[:,1], z = data[:,2],
    text = text, **kwargs
  )  

if __name__ == '__main__':
  fitCurve()
  fitHeighmap()
  fitSurface()
  fitVolume()
  quiverCurve()
  quiverSurface()
  plt.show()