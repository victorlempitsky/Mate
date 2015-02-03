## Using a pretrained network

Maté can load, apply, and manipulate networks from the [MatConvNet repository](http://www.vlfeat.org/matconvnet/pretrained/)
using `readMatConvNet` functions that returns a cell array of Maté layers (and optionally a MatConvNet structure with 
classnames and descriptions). The cell array can be then used to construct the network:

```matlab
[layers,classes] = readMatConvNet(modelfilename); 
net = MateNet(layers); %create MateNet
```

Note that the first layer in the cell array is of class `MatePreprocessImageLayer`.
It is applicable to single images only, and implements the necessary preprocessing
procedure. A network that is applicable to batches can be built by striping the 
first layer and using it separately to preprocess each image:

```matlab
images = { imread('im1.png'), imread('im2.png', imread('im3.png') };
layers = readMatConvNet(modelfilename); 

%preparing the batch
batch = zeros([layers{1}.imageSize numel(images)]});
for i=1:numel(images)
  batch(:,:,:,i) = layers{1}.forward(images{i});
end

%building the network
net = MateNet(layers(2:end));

%applying the network to the batch
net = net.makePass(batch);

%getting the output of the last layer
output = net.getBlob(layers{end}.name);
```

Obviously, one can build new networks using parts of the repository network:
```matlab
layers = readMatConvNet(modelfilename);
net = MateNet([layers(1:5); newTop]); %newTop is a cell array with new top layers to be trained