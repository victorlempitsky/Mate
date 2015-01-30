classdef MateLayer
% The abstract class for all layers in Mate
  properties
    name = {};
    takes = 'preceding';
    weightDecay = [];
    learningRate = [];
    forwardTime = 0;
    backwardTime = 0;
    skipBackward = false;
    weights = MateParams;
    shareWith = [];
  end
    
  methods (Abstract)
    [y,obj] = forward(obj,x)
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn == 1;
    end
    
    function t = canProduce(nOut)
      t = nOut == 1;
    end  
  end
  
  
  methods
    function [dzdx,obj] = backward(obj, x, dzdy, y)
       error('Not defined'); %this stub would do for layers that are not involved in backward pass
    end
    
    function obj = MateLayer(params)
      obj.weights = MateParams;
      for i=1:2:numel(params)
        obj = setfield(obj, params{i}, params{i+1}); 
      end
    end
    
    function obj = initParams(obj,params)
      for i=1:2:numel(params)
        obj = setfield(obj, params{i}, params{i+1}); 
      end
    end
    
    function obj = update(obj, lr, mr, batchSz) 
      for j = 1:numel(obj.weights.w)
        obj.weights.momentum{j} = mr*obj.weights.momentum{j}-....
          lr*obj.learningRate(j)*obj.weightDecay(j)*obj.weights.w{j}-...
          lr*obj.learningRate(j)/batchSz*obj.weights.dzdw{j};
        obj.weights.w{j} = obj.weights.w{j}+obj.weights.momentum{j};
      end
    end
    
    function obj = move(obj, destination)
      switch destination
        case 'gpu'
          moveop = @(x) gpuArray(x) ;
        case 'cpu'
          moveop = @(x) gather(x) ;
        otherwise, error('Unknown desitation ''%s''.', destination) ;
      end   
      for j = 1:numel(obj.weights.w)
        obj.weights.w{j} = moveop(obj.weights.w{j});
        obj.weights.dzdw{j} = moveop(obj.weights.dzdw{j});
        obj.weights.momentum{j} = moveop(obj.weights.momentum{j});       
      end
    end
    
    function obj = init(obj)
      for j = 1:numel(obj.weights.w)
        obj.weights.dzdw{j} = zeros(size(obj.weights.w{j}),...
                                  'like',obj.weights.w{j});
        obj.weights.momentum{j} = zeros(size(obj.weights.w{j}),...
                                  'like',obj.weights.w{j});
      end  
      if numel(obj.weightDecay) ~= numel(obj.weights.w)
        obj.weightDecay = zeros(1,numel(obj.weights.w));
      end
      if numel(obj.learningRate) ~= numel(obj.weights.w)
        obj.learningRate = ones(1,numel(obj.weights.w));
      end
    end
    
    function obj = displayWeights(obj)
      if numel(obj.weights.w) == 0
        warning('The layer does not have parameters to display');
        return
      end
      x = obj.weights.w{1};

      if ndims(x) > 4 || isscalar(x)
        warning('Scalars and (>4)-dimensional blobs are not supported in MateLayer.display');
        return
      end

      if ndims(x) == 1
        subplot(2,1,1); plot(x); title('W');
        subplot(2,1,2); plot(sort(x)); title('W (sorted)');
      end

      if ndims(x) == 2
        if max(size(x)) > 200
          imagesc(x);
        else
          pcolor(x);
        end
      end

      if ndims(x) == 3 || ndims(x) == 4
        imagesc(getPlate(x));  
      end      
    end
    
  end 
  
end