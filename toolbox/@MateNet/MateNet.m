classdef MateNet 
  properties
    layers = {};
    nBlobs = 0;
    expectIn = 0;
    layersId = containers.Map;
    blobsId = containers.Map;
    forwardSchedule = struct;
    backwardSchedule = struct;
    updateSchedule = [];
    mode = 'cpu';
    
    x = {};
    dzdx = {};
  end
  
  methods (Static)
    function net = loadobj(snet)
      net = snet;
      net = net.init(net.layers);
      if strcmp(net.mode,'gpu')
        net.mode = 'cpu';
        try
          net = net.move('gpu');
        catch
          warning('Unable to move the network to GPU');
          net.mode = 'gpu';
          net = net.move('cpu');
        end
      end        
    end    
  end
  
  methods
    function net = MateNet(layers)
      assert(nargin > 0 && iscell(layers),...
        'The network must be initialized with a cell array of layers');
      net = net.init(layers);
    end
    
   
    function snet = saveobj(net)
      snet = net;
      snet = snet.clearBlobs;
%       if strcmp(snet.mode,'gpu')
%         %deep copy of learnable parameters needed
%         for i=1:numel(snet.layers)
%           for j=1:numel(snet.layers{i}.weights.w)
%             if isempty(snet.layers{i}.weights.w)
%               continue;
%             end
%             tmp = snet.layers{i}.weights;
%             save('tmpmate.mat','tmp','-v6');
%             load('tmpmate.mat');
%             delete('tmpmate.mat');
%             snet.layers{i}.weights = tmp;            
%           end
%         end      
%         snet = snet.move('cpu');
%         snet.mode = 'gpu';
%       end
    end
    
    function net = clearBlobs(net)
      net.x = cell(net.nBlobs,1);
      net.dzdx = cell(net.nBlobs,1);
    end
      
 
    net = init(net,layers)
    net = makePass(net, xin, dobackward, varargin)
    [net, info, dataset] = trainNet( net, getBatch, dataset, varargin )
    net = move(net, destination)
    l = getLayer(net, layerName)
    [x, dzdx] = getBlob(net, blobName)
    dispTimes(net)
    dispBlob(net,blobNames) %displays one or multiple blobs 
    disp(net)
  end
  
end


