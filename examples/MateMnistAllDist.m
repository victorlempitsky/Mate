function MateMnistAllDist
%demonstrates Mate on MNIST
%example is derived from the analogous MatConvNet example

[thispath,~,~] = fileparts(mfilename('fullpath'));
dataDir = [thispath '/data/mnist'] ;
expDir = [thispath '/data/expAllDist'] ;
useGpu = true;

dataset = struct;
dataset.imdb = getMnistImdb(dataDir) ;

% Define the network

f=1/100 ;
net = MateNet( {
  MateConvLayer(f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single'), ...
                'stride', 1, 'pad', 0, 'name', 'conv1')
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)
  MateConvLayer(f*randn(5,5,20,50, 'single'), zeros(1, 50, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)  
  MateConvLayer(f*randn(4,4,50,500, 'single'), zeros(1, 500, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])  
  MateReluLayer
  MateFlattenLayer
  MateFullLayer(f*randn(10,500, 'single'), zeros(10,1, 'single'),... 
                'weightDecay', [0.005 0.005], 'name','full') 
  MateAllDistLayer('name','distances')  
  MateMetricHingeLossLayer('takes',{'distances','input:2'},...
                'name','loss', 'positiveWeight', 10)
  MateNNAccuracyLayer('name','nnAccuracy',...
                'takes',{'distances','input:2'})
 % MateRankErrorLayer('takes',{'distances','input:2'},'invert',true,'name','rankError')
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
dataset.train = dataset.train(randperm(numel(dataset.train)));
dataset.val = dataset.val(randperm(numel(dataset.val)));

[net,info,dataset] = net.trainNet(@getBatch, dataset,...
     'numEpochs',100, 'continue', false, 'expDir', expDir,...
     'learningRate', 0.001,'monitor', {'loss','nnAccuracy'},...
     'showLayers', 'conv1', 'showBlobs', 'distances',...
     'onEpochEnd', @onEpochEnd, 'learningRate', 0.0001) ;
   
end

%----------------------------------------------------------%

function [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
eoe=false;
batchStart = batchNo*dataset.batchSize+1;
batchEnd = (batchNo+1)*dataset.batchSize;

if istrain
  if batchEnd >= numel(dataset.train)
    batchEnd = numel(dataset.train);
    eoe = true; %end of epoch
  end
  batch = dataset.train(batchStart:batchEnd);
  labels = dataset.imdb.images.labels(1,batch) ;
else
  if batchEnd >= numel(dataset.val)
    batchEnd = numel(dataset.val);
    eoe = true; %end of epoch
  end
  batch = dataset.val(batchStart:batchEnd); 
  labels = dataset.imdb.images.labels(1,batch) ;
  [labels,order] = sort(labels);
  batch = batch(order);
end

x{1} = dataset.imdb.images.data(:,:,:,batch) ;
x{2} = single(bsxfun(@eq,labels',labels));
%repmat(labels,[numel(labels),1])

end

%--------------------------------------------------------------%
function [net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)
dataset.train = dataset.train(randperm(numel(dataset.train)));

end

