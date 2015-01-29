## Defining a network/General concepts

### Defining a network

A Maté **network** (`MateNet` class) contains a cell array of **layers**. 
A network is initialized with a cell array of layers:

```

net = MateNet( { MateLayerA(....)
                 MateLayerB(....)
                 MateLayerC(....)
               });
```

Later, you can access layers using `net.layers` or `net.getLayer(layerName)`. 

### Initializing a layer

Each layer is derived from the `MateLayer` class. Each layer has a constructor
that takes a number of obligatory parameters and a number of optional parameters
passed in the *('paramName',paramValue)* way, e.g.:
```
  MateConvLayer(single(0.01)*randn(5,5,1,20, 'single'), zeros(1, 20, 'single'), ...
                'name','conv1','stride', 1, 'pad', 0)
```
Here, an instance of `MateConvLayer` is created; the two obligatory 
parameters are the initial values of filters and biases, and optional parameters include the layer name,
stride and padding values.

### Blobs and the network graph

Blobs in Maté refer to numeric arrays, which serve as inputs and outputs to layers during forward and backward propagation.
For each blob, a network contains the array of values (`x`) and the array of partial derivatives (`dzdx`). 
A network can be seen as a directed graph which connects vertices corresponding to blobs and vertices corresponding to layers.

Each layer vertex points to the blob vertices corresponding to blobs produced by this layer during forward propagation.
Each blob vertex points to the layer vertices corresponding to layers that take this blob during forward propagation.
A valid graph needs to be directed acyclic. The graph is defined via the naming system and the `takes` attribute.

### Defining the network graph

Each layer and each blob in the network have its names. The name of each layer 
can be specified during construction using the `'name'` attribute (as in the example above).
Otherwise a name will be assigned automatically (e.g. `'MateConvLayer_004'`).

The naming of blobs is automatic. The name of each blob is derived from the name of a layer that produces it. 
E.g. if a layer called `'splitlayer1'` produces three blobs, then these blobs will be called: `'splitlayer1:1'`, `'splitlayer1:2'`,`'splitlayer1:3'`.
You can omit the trailing `':1'` part if you are refering to the first blob 
produced by the layer (e.g. blob `'conv1'` is the same as `'conv1:1'` and corresponds to the first blob
produced by the layer with name `'conv1'`.

The network graph is defined by the `'takes'` attribute that can be passed to 
the constructor of any layer, e.g.:
```
  MateNNAccuracyLayer('name','nnAccuracy',...
                'takes',{'distances','input:2'})
```
The `'takes'` attribute should be set to a cell array of blob names that should serve as inputs to the layer.
If a single blob serves as an input than the curly brackets can be omitted.
Finally, by default it is assumed that the layer takes the first blob produced by
the preceding layer in the cell array (thus a chain network graph can be specified 
without specifying `'takes'` attributes.

### Input to the network

Almost every network requires some input to run on. The input in Maté is assumed to be
a bunch of numeric arrays, hence they are also treated as blobs. These blobs have special
names: `'input:1'`, `'input:2'`, `'input:3'`, etc. E.g. `'input:1'` can contain data, and 
`'input:2'` can contain labels to be used during training.

### CPU/GPU

A network can be moved to GPU using `net.move('gpu')` and back to CPU using `net.move('cpu')`. 




