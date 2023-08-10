Welcome to GPEX's documentation!
===================================


.. contents:: Table of Contents
   :depth: 2
   :local:


.. currentmodule:: gpex

Introduction
------------------------
GPEX is a tool for performing knowledge distillation between Gaussian processes(GPs) and artificial neural networks (NNs).
It takes in an arbitrary pytorch module, and replaces one neural-network submodule to be replaced by GPs. 


.. image:: tgpframeworkv.png
    :width: 400px
    :height: 200px
The pytorch module can be quite general (as depicted above), with a few requirements:
    - It has to have on ANN submodule.
    - The ANN submodule has to take in one input tensor and one output tensor. The input has to be of dimension [Nx*], where *
means any number of dimensions, but the output has to be of shape [NxD].

Base Modules
------------------------

dsd fs df sd fsd fs dfs fd sdf sdf. sdfs dfsdf
d d s df sdf sdf. 

.. autoclass:: GPEXModule
   :members:


Modules
------------------------

dddd
