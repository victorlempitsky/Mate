#Maté

**Maté** is a MATLAB library for ConvNets (Convolutional Neural Networks).
Maté is derived from **MatConvNet** (http://www.vlfeat.org/matconvnet/) and uses MatConvNet routines for several time-critical operations. 

Maté's goal is to simplify prototyping and experimentation with non-standard layers and architectures.

Maté continues the tradition of naming ConvNet libraries after caffeine-related drinks (http://en.wikipedia.org/wiki/Mate_%28beverage%29).

##Functionality

Maté provides functionality for:
* Easy definition of new layers (just define forward and backward operators, which often takes little effort in MATLAB).
* Easy definition of non-chain networks (any directed acyclic graph can be defined with ease) using MATLAB scripts.
* Easy GPU mode thanks to MATLAB Parallel Toolbox and MatConvNet routines. For many (most?) new layers same MATLAB code can be reused for CPU and GPU modes.
* Easy visualization of the internal state of a network during training.

##Installation 
Install MatConvNet (http://www.vlfeat.org/matconvnet/), clone this repository, and run setup.m (which simply adds subfolders to MATLAB path).

##Status
The project is in a very early prerelease state, use at your own risk.


