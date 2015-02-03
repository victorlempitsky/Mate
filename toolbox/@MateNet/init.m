function net = init(net,layers)
% builds a network starting from the cell array of layers

  net.layers = layers;
  N = numel(net.layers);

  net.layersId = containers.Map({'input'},{0});
  net.blobsId = containers.Map;

  demand = containers.Map;
  for i=1:N
    if isempty(net.layers{i}.name)
      net.layers{i}.name = [class(net.layers{i}) num2str(i,'%02d')];
    end 

    if strcmpi(net.layers{i}.name, 'input')
      error('Name "input" reserved for the external input to the net');
    end

    net.layersId(net.layers{i}.name) = i;
    if ~iscell(net.layers{i}.takes) && strcmp(net.layers{i}.takes,'preceding') %%
      if i == 1
        net.layers{i}.takes='input:1';
      else
        net.layers{i}.takes = [net.layers{i-1}.name ':1'];
      end
    end
    if ~iscell(net.layers{i}.takes)
      net.layers{i}.takes={net.layers{i}.takes};
    end
    net.layers{i}.takes = net.layers{i}.takes(:)';
    for t = net.layers{i}.takes   
      [layerName, blobNo] = parseid(t{1});
      if isKey(demand, layerName)
        demand(layerName) = [demand(layerName) blobNo];
      else
        demand(layerName) = blobNo;
      end
    end
  end

  net.updateSchedule = [];
  for i=1:N
    if ~isempty(net.layers{i}.shareWith)
      net.layers{i}.weights = ...
        net.layers{net.layersId(net.layers{i}.shareWith)}.weights;
    elseif ~isempty(net.layers{i}.weights.w)
      net.updateSchedule(end+1) = i;
    end
  end

  net.expectIn = 0;
  noutputs = ones(1,N); %we expect that each layer has at least one output
    
  for n=net.layersId.keys
    n = n{1};
    if ~isKey(demand, n)
      continue;
    end

    d = demand(n);
    if any(d < 1) || any( d ~= uint8(d))
      error(['Not a positive integer output index requested for layer ' n]);
    end
%     if numel(unique(d)) < numel(d) && ~strcmpi(n, 'input')
%       warning(['One of the outputs of layer ' n ' is reused at least twice. Mind potential overwrite of the derivatives during backprop.']);
%     end
    if numel(unique(d)) < max(d)
      warning(['At least one of the outputs of layer ' n ' is unused.']);
    end
    if strcmpi(n,'input')
      net.expectIn = max(d);
    else
      noutputs(net.layersId(n)) = max(1,max(d));
    end
  end


  %assigning output blobs to id, putting them to the blobsId map
  for j=1:net.expectIn
    net.blobsId(['input:' num2str(j)])=j;
  end
  if net.expectIn > 0
    net.blobsId('input')=1;
  end
  
  for i=1:N
    while ~net.layers{i}.canProduce(noutputs(i))
      noutputs(i) = noutputs(i)+1;
      assert(noutputs(i) < 100, ['The layer ' net.layers{i}.name...
          ' cannot produce the required number of blobs']); %assuming >100 blobs cannot be requested
    end
  end
  
  cs = [net.expectIn cumsum(noutputs)+net.expectIn];
  for i=1:N
    net.blobsId([net.layers{i}.name]) = cs(i)+1;
   
    for j=1:noutputs(i)
      net.blobsId([net.layers{i}.name ':' num2str(j)]) = cs(i)+j;
    end
  end

  %creating schedule
  net.forwardSchedule = struct('layer',num2cell(1:N),'in',cell(1,N),'out',cell(1,N));
  for i=1:N
    net.forwardSchedule(i).layer = i;
    net.forwardSchedule(i).out = cs(i)+1:cs(i+1);
    l = net.layers{i};
    for t = l.takes
      assert(isKey(net.blobsId,t{1}), ...
        ['Blob "' t{1} '" requested by the layer "' l.name ...
        '" not found in the network during initialization']);
      net.forwardSchedule(i).in(end+1) = net.blobsId(t{1});
    end
    assert(net.layers{i}.canTake(numel(net.forwardSchedule(i).in)),...
      ['Layer "' net.layers{i}.name '" cannot take '...
       num2str(numel(net.forwardSchedule(i).in)) ' blob(s).']);
    assert(net.layers{i}.canProduce(numel(net.forwardSchedule(i).out)),...
      ['Layer "' net.layers{i}.name '" cannot produce '...
       num2str(numel(net.forwardSchedule(i).out)) ' blob(s).']);
  end  

  %sorting and verifying schedule
  computed = {'input'};
  for j=1:net.expectIn
    computed{end+1} = ['input:' num2str(j)];
  end

  todo = [1:N];
  seq = [];

  while ~isempty(todo)
    progress = false;
    pos = 1;
    while pos <= numel(todo)
      l = net.layers{todo(pos)};
      ready = true;
      for t = l.takes 
        if ~any(strcmpi(computed,t{1}))
          ready = false;
        end
      end    

      if ~ready
        pos = pos+1;
        continue;
      end

      seq(end+1) = todo(pos);
      progress = true;
      computed{end+1} = l.name;
      for j=1:noutputs(todo(pos))
        computed{end+1} = [l.name ':' num2str(j)];
      end
      todo(pos) = [];
    end
    if ~progress
      error('Wrong schedule. Perhaps the graph is not directed acyclic.');
    end 
  end
  net.forwardSchedule = net.forwardSchedule(seq);

  net.backwardSchedule = fliplr(net.forwardSchedule);      
  eraseBackward = zeros(1,numel(net.backwardSchedule));  
  for i=1:numel(net.backwardSchedule)
    eraseBackward(i) = net.layers{net.backwardSchedule(i).layer}.skipBackward;
  end
  net.backwardSchedule(eraseBackward == true) = [];
  
  %checking backward schedule for overlaps
  
  net.nBlobs = max(cell2mat(net.blobsId.values));  
  
  updateBackward = zeros(1,net.nBlobs);
  for i = 1:numel(net.backwardSchedule)
    updateBackward(net.backwardSchedule(i).in) = updateBackward(net.backwardSchedule(i).in)+1;
  end  
  if any(updateBackward > 1)
    error('At least one blob needs to be updated twice during backprop. Consider using a split layer.');
  end

  net.x = cell(net.nBlobs,1);
  net.dzdx = cell(net.nBlobs, 1);
  
  for i=1:N
    net.layers{i} = net.layers{i}.init;
  end
end

function [layerName, blobNo] = parseid(str)
  [layerName, blobNo] = strtok({str},':');
  assert(numel(layerName) == 1);
  layerName = layerName{1};
  blobNo = blobNo{1};

  if isempty(blobNo)
    blobNo = 1;
  else
    blobNo = str2double(blobNo(2:end));
    assert(blobNo > 0);
  end
end
