function [net, info, dataset] = trainNet( net, getBatch, dataset, varargin )
% getBatch signature:
% [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
% onEpochEnd signature:
% [net,dataset] = onEpochEnd(net,dataset)

opts.numEpochs = 100;
opts.learningRate = 0.001;
opts.continue = false;
opts.expDir = [];
opts.sync = verLessThan('matlab', '8.4');
opts.momentum = 0.9 ;
opts.monitor = {};
opts.showBlobs = {};
opts.showLayers = {};
opts.onEpochEnd = [];

opts = vl_argparse(opts, varargin) ;

if ~iscell(opts.monitor)
  assert(ischar(opts.monitor));
  opts.monitor = {opts.monitor};
end

if ~iscell(opts.showBlobs)
  assert(ischar(opts.showBlobs));
  opts.showBlobs = {opts.showBlobs};
end

if ~iscell(opts.showLayers)
  assert(ischar(opts.showLayers));
  opts.showLayers = {opts.showLayers};
end

if ~isempty(opts.expDir) && ~exist(opts.expDir), mkdir(opts.expDir) ; end

rng(0) ;
info.train = containers.Map(opts.monitor,cell(size(opts.monitor)));
info.val = containers.Map(opts.monitor,cell(size(opts.monitor)));

lr = 0 ;
modelPath = [];
modelFigPath = [];
for epoch=1:opts.numEpochs
  prevLr = lr ;
  lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  if ~isempty(opts.expDir)
    % fast-forward to where we stopped
    modelPath = fullfile(opts.expDir, 'net-epoch-%d.mat') ;
    modelFigPath = fullfile(opts.expDir, 'net-train') ;
    if opts.continue
      if exist(sprintf(modelPath, epoch),'file'), continue ; end
      if epoch > 1
        fprintf('resuming by loading epoch %d\n', epoch-1) ;
        load(sprintf(modelPath, epoch-1), 'net', 'info') ;
      end
    end
  end

  trainBatchNo = 0;
  
  localinfo.train = containers.Map(opts.monitor,...
    num2cell(zeros(1,numel(opts.monitor),'single')));

  % reset momentum if needed
  if prevLr ~= lr
    fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
    for l=1:net.updateSchedule
      for j = 1:numel(net.layers{l}.weights.momentum)
        net.layers{l}.weights.momentum{j}(:) = 0;
      end
    end
  end
  
  eoe = false;
  
  while ~eoe
    trainBatchNo = trainBatchNo+1;
    batch_time = tic ;
    fprintf('training (epoch %02d, batch %03d): ', epoch, trainBatchNo) ;
    [x, eoe, dataset] = getBatch(true, trainBatchNo, dataset);
    batch_size = size(x{1},ndims(x{1}));

    if strcmp(net.mode,'gpu')
      for i=1:numel(x)
        x{i} = gpuArray(x{i}) ;
      end
    end
    
    %resetting 
    for i=net.updateSchedule
      for j=1:numel(net.layers{i}.weights.dzdw)
        net.layers{i}.weights.dzdw{j}(:) = 0;
      end 
    end    

    % backprop
    net = makePass(net, x, true, 'sync', opts.sync);
    
    % gradient step
    for l=net.updateSchedule
      net.layers{l} = update(net.layers{l},lr, opts.momentum, batch_size);
    end

    % print information
    batch_time = toc(batch_time) ;
    speed = batch_size/batch_time ;

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    for t=opts.monitor
      v = gather(net.x{net.blobsId(t{1})});
      localinfo.train(t{1}) = localinfo.train(t{1})+v; 
      fprintf([';  ' t{1} ' :  %f'], v);
    end
    fprintf('\n') ;

  end % next batch

 
  localinfo.val = containers.Map(opts.monitor,...
    num2cell(zeros(1,numel(opts.monitor),'single')));
  
  valBatchNo = 0;
  eoe = false;
  
  % evaluation on validation set
  while ~eoe
    batch_time = tic ;
    valBatchNo = valBatchNo+1;

    fprintf('validation (epoch %02d, batch %03d): ', epoch, valBatchNo) ;
    
    [x,eoe,dataset] = getBatch(false, valBatchNo, dataset);
    batch_size = size(x{1},ndims(x{1}));

    if strcmp(net.mode,'gpu')
      for i=1:numel(x)
        x{i} = gpuArray(x{i}) ;
      end
    end
    
    net = makePass(net,x, false, 'sync', opts.sync);

    % print information
    batch_time = toc(batch_time) ;
    speed = batch_size/batch_time ;

    fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    for t=opts.monitor
      v = gather(net.x{net.blobsId(t{1})});
      localinfo.val(t{1}) = localinfo.val(t{1})+v; 
      fprintf([';  ' t{1} ' :  %f'], v);
    end
    fprintf('\n') ;
  end
  
  for t=opts.monitor
    info.train(t{1}) = [info.train(t{1})  localinfo.train(t{1})/trainBatchNo];
    info.val(t{1}) = [info.val(t{1})  localinfo.val(t{1})/valBatchNo];
  end  
 
  %dispTimes(net); drawnow;
  
  if epoch > 1 && ~isempty(opts.monitor)
    for t = opts.monitor
      figure(net.blobsId(t{1}));
      plot(1:epoch, info.train(t{1}), 'k', 1:epoch, info.val(t{1}), 'r');
      xlabel('training epoch');
      ylabel(t{1});
      h=legend('train', 'val') ;
      set(h,'color','none');   
      drawnow ;
      if ~isempty(modelFigPath)
        print(gcf, [modelFigPath '_' t{1} '.pdf'], '-dpdf') ;
      end
    end
  end
  
  dispTimes(net);
  if strcmp(net.mode,'gpu') && ~opts.sync
    warning('Unsynced GPU modes: layer timings are inaccurate.');
  end
  
  for t=opts.showBlobs
    dispBlob(net,t{1});
  end
  for t=opts.showLayers
    dispLayer(net,t{1});
  end      
  drawnow

  
  
  if ~isempty(opts.onEpochEnd)
    [net,dataset] = opts.onEpochEnd(net,dataset);
  end
  
  % save
  if ~isempty(modelPath)
    save(sprintf(modelPath,epoch), 'net', 'info') ;
  end  

end

