pathto = '/media/storage/vilem/Downloads/';
fname = 'imagenet-caffe-ref.mat';
webloc = 'http://www.vlfeat.org/matconvnet/models/';

if ~exist([pathto fname],'file')
  websave([pathto fname], [webloc fname]);
end

[layers,classes] = readMatConvNet([pathto fname]);


%trying on peppers
net = MateNet( cat(1,layers, {MateArgmaxLayer('name','class')}) );

im = imread('peppers.png');
net = net.makePass(im);

imshow(im);
fprintf('This image is classified as "%s".\n', classes.description{net.getBlob('class')});

%testing speed in batch mode
net = MateNet(layers(2:end)); %same network without preprocessor and class selector
batchSz = 256;
im = layers{1}.forward(im);
batch = repmat(im,[1 1 1 batchSz]);

net.makePass(batch);
%now measuring
tic
net.makePass(batch);
tm = toc;
fprintf('\n Batch forward-passed on CPU in %f sec', tm);


%let us measure on GPU
batch = gpuArray(batch);
net = net.move('gpu');
%warm-up run
net.makePass(batch);
%now measuring
tic
net.makePass(batch);
tm = toc;
fprintf('\n Batch forward-passed on GPU in %f sec', tm);

