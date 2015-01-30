#Maté

**Maté** is a MATLAB library for ConvNets (Convolutional Neural Networks).
Maté is derived from [**MatConvNet**](http://www.vlfeat.org/matconvnet/) and uses MatConvNet routines for several time-critical operations. 

Maté is object-oriented and has the goal to simplify prototyping and experimentation, in particular
to minimize the efforts needed to try new non-standard layers and new non-standard network graphs.

Maté continues the tradition of naming a ConvNet library after a caffeine-related [drink](http://en.wikipedia.org/wiki/Mate_%28beverage%29).
Other such libraries include [Decaf](https://github.com/UCB-ICSI-Vision-Group/decaf-release) (Python, CPU) and 
[Caffe](http://caffe.berkeleyvision.org/) (C++ with interfaces, CPU/GPU).

##Functionality

Maté provides functionality for:
* Easy definition of new layers (just define forward and, if needed, backward procedures; it often takes couple of lines in MATLAB).
* Easy definition of non-chain networks (any directed acyclic graph can be defined with ease) within MATLAB code.
* Easy GPU mode thanks to MATLAB Parallel Toolbox and MatConvNet routines. For many (most?) new layers same MATLAB code can be reused for CPU and GPU modes.
* Easy visualization of the internal state of a network during training.

##Installation 
Install MatConvNet (http://www.vlfeat.org/matconvnet/), clone this repository, and run setup.m (which simply adds subfolders to MATLAB path).

##How-to
* [Build a network (+other basics)](docs/network.md)
* [Train a network](docs/training.md)
* [Layer catalog](docs/catalog.md)
* [Define a new layer](docs/layer.md)


##Status
The project is in a very early prerelease state, use at your own risk.


