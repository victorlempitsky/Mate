classdef MateConvLayer < MateLayer
  properties
    pad = 0;
    stride = 1;
  end
  methods
    function obj = MateConvLayer(filters,biases,varargin)
      obj@MateLayer(varargin);
      if isempty(obj.shareWith)
        obj.weights.w{1} = filters;
        obj.weights.w{2} = biases;
      end
      %if isempty(obj.learningRate) obj.learningRate = single([1.0 1.0]); end
      %if isempty(obj.weightDecay) obj.weightDecay = single([0.0 0.0]); end
    end
    
    function [y,obj] = forward(obj,x)
      switch ndims(x)
        case 4
          y = vl_nnconv(x, obj.weights.w{1}, obj.weights.w{2},...
           'pad', obj.pad, 'stride', obj.stride);
        case 2 
          y = squeeze(vl_nnconv(reshape(x,[1 1 size(x)]), obj.weights.w{1}, obj.weights.w{2},...
            'pad', obj.pad, 'stride', obj.stride));
        otherwise error('Input to a conv layer should have either 2 or 4 dimensions');
      end      
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      if ndims(x) == 4
        [dzdx, dzdf, dzdb] = vl_nnconv(...
                x, obj.weights.w{1}, obj.weights.w{2}, ...
                dzdy, 'pad', obj.pad, 'stride', obj.stride) ;  
      else
        [dzdx, dzdf, dzdb] = vl_nnconv(...
                    reshape(x,[1 1 size(x)]), obj.weights.w{1}, obj.weights.w{2}, ...
                    reshape(dzdy,[1 1 size(dzdy)]), 'pad', obj.pad, 'stride', obj.stride) ;
        dzdx = squeeze(dzdx);
%        dzdf = squeeze(dzdf);
%        dzdb = squeeze(dzdb);
      end
      obj.weights.dzdw{1} = obj.weights.dzdw{1}+dzdf;
      obj.weights.dzdw{2} = obj.weights.dzdw{2}+dzdb;
    end
  end  
end