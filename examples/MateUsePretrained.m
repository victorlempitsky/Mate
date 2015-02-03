pathto = '/media/storage/vilem/Downloads/';
fname = 'imagenet-caffe-alex.mat';
webloc = 'http://www.vlfeat.org/matconvnet/models/';

if ~exist([pathto fname],'file')
  disp('Loading the model from the MatConvNet repository...');
  websave([pathto fname], [webloc fname]);
  disp('Done...');
end

[layers,classes] = readMatConvNet([pathto fname]);


%trying on peppers

%using MatConvNet model + an argmax layer
net = MateNet( [ layers; {MateArgmaxLayer('name','classifier')} ] );

im = imread('peppers.png');
net = net.makePass(im);

imshow(im);
fprintf('This image is classified as "%s".\n',...
          classes.description{net.getBlob('classifier')});

%testing speed in batch mode
net = MateNet(layers(2:end)); %same network without preprocessor and argmax
batchSz = 256;
im = layers{1}.forward(im); %preprocess an image and duplicate into a batch
batch = repmat(im,[1 1 1 batchSz]); 

net = net.makePass(batch); %warm-up
%now measuring
tic
net = net.makePass(batch);
tm = toc;
fprintf('Batch forward-passed on CPU in %f sec.\n', tm);


%let us measure on GPU
batch = gpuArray(batch);
net = net.move('gpu');

net = net.makePass(batch);%warm-up
%now measuring
tic
for i=1:10
  net = net.makePass(batch);
end
tm = toc;
fprintf('Batch forward-passed on GPU in %f sec.\n', tm/10);

