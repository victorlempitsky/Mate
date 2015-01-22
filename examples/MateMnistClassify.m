function MateMnistClassify
%demonstrates Mate on MNIST
%example is derived from the analogous MatConvNet example

[thispath,~,~] = fileparts(mfilename('fullpath'));
dataDir = [thispath '/data/mnist'] ;
expDir = [thispath '/data/exp'] ;
useGpu = true;

dataset = struct;
dataset.imdb = getMnistImdb(dataDir) ;

% Define the network

f=1/100 ;
net = MateNet( {
  MateConvLayer(f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single'), ...
                'stride', 1, 'pad', 0)
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)
  MateConvLayer(f*randn(5,5,20,50, 'single'), zeros(1, 50, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)  
  MateConvLayer(f*randn(4,4,50,500, 'single'), zeros(1, 500, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])  
  MateReluLayer
  MateConvLayer(f*randn(1,1,500,10, 'single'), zeros(1, 10, 'single'), ...
                'name','prediction', 'stride', 1, 'pad', 0, ...
                'weightDecay', [0.005 0.005])   
  MateSoftmaxLossLayer('name','loss',...
                'takes',{'prediction','input:2'})
  MateMultilabelErrorLayer('name','error',...
                'takes',{'prediction','input:2'})
  } );


%subtract mean
dataset.imdb.images.data = bsxfun(@minus, dataset.imdb.images.data,...
            mean(dataset.imdb.images.data,4)) ;
          
%move to GPU         
if useGpu
  net = net.move('gpu');
  dataset.imdb.images.data = gpuArray(dataset.imdb.images.data) ;
  dataset.imdb.images.labels = gpuArray(dataset.imdb.images.labels) ;
end

dataset.train = find(dataset.imdb.images.set == 1);
dataset.val = find(dataset.imdb.images.set == 3);
dataset.batchSize = 100;

[net,info,dataset] = net.trainNet(@getBatch, dataset,...
     'numEpochs',100, 'continue', true, 'expDir', expDir,...
     'learningRate', 0.001,'monitor', {'loss','error'}) ;

%----------------------------------------------------------%

function [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
eoe=false;
if istrain
  batchStart = batchNo*dataset.batchSize+1;
  batchEnd = (batchNo+1)*dataset.batchSize;
  if batchEnd >= numel(dataset.train)
    batchEnd = numel(dataset.train);
    eoe = true;
  end
  batch = dataset.train(batchStart:batchEnd);
else
  batchStart = batchNo*dataset.batchSize+1;
  batchEnd = (batchNo+1)*dataset.batchSize;
  if batchEnd >= numel(dataset.val)
    batchEnd = numel(dataset.val);
    eoe = true; %end of epoch
  end
  batch = dataset.val(batchStart:batchEnd); 
end
x{1} = dataset.imdb.images.data(:,:,:,batch) ;
x{2} = dataset.imdb.images.labels(1,batch) ;

