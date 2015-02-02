##Training a network

To train a network you need to follow the following four steps.

1. [Build the network](network.md).
2. Create the `dataset` variable (e.g. a structure) that describes the training and the validation datasets.
This variable is first passed to the main training function (discussed below) and then passed back to the batch provider every time
it is called. All passes are by reference. For a small dataset, the whole data can be stored as a field, for bigger ones
some kind of iterators over the disk data can be used.
3. Write a batch provider function that must have the following nomenclature:
```
[x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
```
The input variables are: whether a train or a validation batch is requested (`istrain`), the number of the batch in 
the epoch (`batchNo`) and, finally, the dataset (`dataset`). 
The first output should return the network input `x` that will be used as `input:1`,`input:2`,etc. blobs.
The second output should be a boolean variable whether this batch ends the epoch. E.g. use the following
fragment to make all training epochs contain 100 batches and all validation epochs contain 10 batches.
```
[x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
......
if istrain
  eoe = batchNo == 100;
else
  eoe = batchNo == 10;
end
......
```
4. Call the training function:
```
[net, info, dataset] = trainNet( net, @getBatch, dataset, .......)
```
The call has a large number of options passed using `'optsName',optsValue` format.
They are given in the table:

 Option name (=default) | Description 
-----------------------| ----------- 
`numEpochs=100` | Training duration 
`learningRate=0.001` | A scalar, specifying learning rate for SGD 
`continue=false` | Whether to load the snapshot with the highest iteration number from disk 
`expDir=[]` | Export dir where snapshots and progress plots are saved to (and loaded from if `continue == true`). No saving/loading happening if empty 
`sync=verLessThan('matlab', '8.4')` | Whether to synchronize calls in GPU mode. Speeds things up (according to matconvnet authors) in R2014a and before. Slows things down in R2014b.
`momentum=0.9` | Momentum in SGD
`monitor={}` | A cell array with the names of the blobs to be "monitored" during training and evaluations. These blobs must be scalar (perhaps produced by some loss or error layer). The training process then shows their value at each iteration, records them into the info structure (returned by training), plots them as training progress (starting from iteration 2), ans saves plots to disk (to `expDir`).
`showBlobs={}`| A cell array with the names of the blobs to be visualized after each epoch
`showLayers={}`| A cell array with the names of the layers, whose weights will be visualized after each epoch
`showTimings=true`| Whether to show forward and backward timings for each layer in a separate figure after each epoch
`snapshotFrequency=30`| The process saves the current state to disk after every epoch and then erases the previous state (to save disk space). If `snapshotFrequency` divides the epoch number, the snapshot is not erased and is kept on the disk.
`onEpochEnd = []`| The callback that can be used for extra monitoring and adjustments
---

The `onEpochEnd` can be set to a pointer to the function with the following nomenclature:
```
[net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)
```
Inside it, the function can do arbitrary things to the network, the dataset, and the learning rate.
