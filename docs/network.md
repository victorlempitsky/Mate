### General concepts

## Defining a network

A Maté **network** (*MateNet* class) contains a cell array of **layers**. 
A network is initialized with a cell array of layers:

```

net = MateNet( { MateLayerA(....)
                 MateLayerB(....)
                 MateLayerC(....)
               });
```

Later, you can access layers using *net.layers* or *net.getLayer('layerName')*. 

## Initializing a layer

Each layer is derived from the *MateLayer* class. Each layer has a constructor
that takes a number of obligatory parameters and a number of optional parameters
passed in the *('paramName',paramValue)* way, e.g.:
```
  MateConvLayer(single(0.01)*randn(5,5,1,20, 'single'), zeros(1, 20, 'single'), ...
                'name','conv1','stride', 1, 'pad', 0)
```
Here, an instance of *MateConvLayer* is created, whereas the two obligatory 
parameters correspond to filters and biases, and optional parameters include the layer' name,
stride and padding values.

## Blobs


## Naming system

## Input to the network




