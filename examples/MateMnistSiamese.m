function MateMnistSiamese
%trains a Siamese architecture on MNIST data using Mate

[thispath,~,~] = fileparts(mfilename('fullpath'));
dataDir = [thispath '/data/mnist'] ;
expDir = [thispath '/data/expSiamese'] ;
useGpu = false;

dataset = struct;
dataset.imdb = getMnistImdb(dataDir) ;


% Define the network

f=1/100 ;

%first encoder
encoder1 = {
  MateConvLayer(f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single'), ...
                'name','conv1','stride', 1, 'pad', 0)
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)
  MateConvLayer(f*randn(5,5,20,50, 'single'), zeros(1, 50, 'single'), ...
                'name','conv2','stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)  
  MateReluLayer
  MateConvLayer(f*randn(4,4,50,500, 'single'), zeros(1, 500, 'single'), ...
                'name','conv3','stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])
  MateReluLayer
  MateConvLayer(f*randn(1,1,500,10, 'single'), zeros(1, 10, 'single'), ...
                'name','conv4','stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])  
};

%second encoder (note parameter sharing with the first encoder
encoder2 = {
  MateConvLayer([], [], ...
                'shareWith','conv1', 'takes', 'input:2', 'stride', 1, 'pad', 0)
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)
  MateConvLayer([], [], ...
                'shareWith','conv2','stride', 1, 'pad', 0)
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)  
  MateReluLayer
  MateConvLayer([], [], ...
                'shareWith','conv3','stride', 1, 'pad', 0)
  MateReluLayer
  MateConvLayer([], [], ...
                'shareWith','conv4','stride', 1, 'pad', 0,'name','conv4_2')                  
};

%computing the distance and defining the loss
top = {
  MatePairwiseDistLayer('takes',{'conv4','conv4_2'},'name','dist')
  MateMetricHingeLossLayer('takes',{'dist','input:3'},...
        'name','loss', 'margin', 1)
};

%nearest neighbor evaluator
nnEval = {
  MateSqueezeLayer('takes','conv4','skipBackward',true)
  MateAllDistLayer('name','allDist1','skipBackward',true)
  MateNNAccuracyLayer('name','nnAccuracy', 'takes',{'allDist1','input:4'})
};

net = MateNet( [ encoder1; encoder2; top; nnEval ]);

%subtract the mean
dataset.imdb.images.data = bsxfun(@minus, dataset.imdb.images.data,...
                              mean(dataset.imdb.images.data,4)) ;
                            
%move to GPU                            
if useGpu
  net = net.move('gpu');
  dataset.imdb.images.data = gpuArray(dataset.imdb.images.data) ;
end

%finalizing the dataset definition
dataset.train = find(dataset.imdb.images.set == 1);
[~,ix] = sort(dataset.imdb.images.labels(dataset.train));
dataset.sortedTrain = dataset.train(ix); %needed to sample positive pairs
dataset.val = find(dataset.imdb.images.set == 3);
[~,ix] = sort(dataset.imdb.images.labels(dataset.val));
dataset.sortedVal = dataset.val(ix); %needed to sample positive pairs
dataset.batchSize = 100;

[net,info,dataset] = net.trainNet(@getBatch,dataset,...
    'numEpochs', 4, 'monitor', {'loss','nnAccuracy'}, 'showLayers', 'conv1',...
    'sync',false,'continue', false, 'expDir', expDir,...
    'onEpochEnd', @onEpochEnd, 'learningRate', 0.001, 'momentum',0.9) ;


% --------------------------------------------------------------------
function [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
% produce training and validation batches
% the first half of each batch is random (hence mostly negative pairs)
% the second half of each batch is mostly positive pairs
eoe=false;
if istrain
  batchStart = (batchNo-1)*dataset.batchSize+1;
  batchEnd = batchNo*dataset.batchSize;
  if batchEnd >= numel(dataset.train)
    batchEnd = numel(dataset.train);
    eoe = true;
  end
  sample = randi(numel(dataset.sortedTrain)-1,[dataset.batchSize/2 1]);
  sortedSample = [dataset.sortedTrain(sample) dataset.sortedTrain(sample+1)]; 
  batch = dataset.train(batchStart:batchEnd);
  
else
  batchStart = (batchNo-1)*dataset.batchSize+1;
  batchEnd = batchNo*dataset.batchSize;
  if batchEnd >= numel(dataset.val)
    batchEnd = numel(dataset.val);
    eoe = true;
  end
  sample = randi(numel(dataset.sortedVal)-1,[dataset.batchSize/2 1]);
  sortedSample = [dataset.sortedVal(sample) dataset.sortedVal(sample+1)];  
  batch = dataset.val(batchStart:batchEnd); 
end
fullBatch1 = [batch(1:end/2) sortedSample(1:end/2)];
fullBatch2 = [batch(end/2+1:end) sortedSample(end/2+1:end)];
x{1} = dataset.imdb.images.data(:,:,:,fullBatch1) ;
x{2} = dataset.imdb.images.data(:,:,:,fullBatch2) ;
x{3} = single(dataset.imdb.images.labels(1,fullBatch1) == ...
            dataset.imdb.images.labels(1,fullBatch2));          
%needed for NN accuracy evaluation:         
x{4} = single(bsxfun(@eq,dataset.imdb.images.labels(1,fullBatch1),...
            dataset.imdb.images.labels(1,fullBatch1)'));

%-----------------------------------------
function [net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)
%-----------------------------------------
dataset.train = dataset.train(randperm(numel(dataset.train)));

%display pairs from the last batch ordered by similarity
[~,order] = sort(getBlob(net,'dist'));

inp1 = squeeze(getBlob(net,'input:1'));
inp1 = inp1(:,:,order);
inp2 = squeeze(getBlob(net,'input:2'));
inp2 = inp2(:,:,order);  

figure(5000);
imagesc(getPlate(cat(2,inp1,inp2)));
title('Last batch column-wise ordered by within-pair distance');  
drawnow



