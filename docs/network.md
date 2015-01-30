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

### Blobs, layers, and the network graph

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
produced by the layer with name `'conv1'`).

The network graph is defined by the `'takes'` attribute that can be passed to 
the constructor of any layer, e.g.:
```
  MateNNAccuracyLayer('name','nnAccuracy',...
                'takes',{'distances','input:2'})
```
The `'takes'` attribute should be set to a cell array of blob names that should serve as inputs to the layer.
If a single blob serves as an input then the curly brackets can be omitted.
Finally, by default it is assumed that the layer takes the first blob produced by
the preceding layer in the cell array (thus chains of layers can be specified 
without specifying `'takes'` attributes explicitly).

At each moment of time after net construction, it is possible to get access to layers and blobs by their names:
```
l = net.getLayer('layerA');
[x,dzdx] = net.getBlob('layerB:2');
```

### Input to the network
Almost every network requires some input to run on. The input in Maté is assumed to be
a bunch of numeric arrays, hence they are also treated as blobs. These blobs have special
names: `'input:1'`, `'input:2'`, `'input:3'`, etc. E.g. `'input:1'` can contain data, and 
`'input:2'` can contain labels to be used during training and validation.

### Applying network to data
Applying the network to data is easy, e.g.:
```
batchData = ... %load some data from somewhere
batchLabels = ... %associated labels
net = net.makePass( {batchData; batchLabels} );
prediction = net.getBlob('prediction');
loss = net.getBlob('loss');

```
Here, it is assumed that `net` requires two input blobs (i.e. that different blobs in the
network take `'input:1'` and `'input:2'`). Numeric arrays `batchData` and `batchLabels` will be used
as such inputs.

The code above also assumes that backward pass is not needed (otherwise use `net.makePass(x,true)`),
and that the layer named `'prediction'` produces predictions (hence the blob named `'prediction'`
contains its output). Likewise it is assumed that the layer named `'loss'` computes the loss over
the batch.

### Deriving new networks
Once the network `net` is modified, e.g. trained, a new network `net2` can be defined by taking a subset of the 
layers of the old ones. E.g. suppose the original network had two last layers that computed the loss and the error
assuming the ground truth labels are provided. Now, let us define a testtime network that does not rely on the availability
of labels and simply provides predictions:
```
net2 = MateNet( net.layers(1:end-2) );
```
Similar tricks can be used to e.g. pretrain some parts of the big network within smaller networks, etc.

### Sharing parameters between layers
Some architectures requires tying gother (sharing) learnable parameters, this can be done using
`'shareWith'` attribute during layer construction:
```
net = MateNet( {
          ...
          MateBlahLayer(W1,W2,'name','Blah1')
          ...
          ...
          MateBlahLayer([],[],'name','Blah2','shareWith','Blah1');
          ...
          )};
```

### CPU/GPU
Assuming, GPU is present and supported by MATLAB (you can create a gpuArray),
a network can be moved to GPU using `net.move('gpu')` and back to CPU using `net.move('cpu')`.





