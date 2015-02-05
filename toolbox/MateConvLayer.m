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
    end
    
    function [y,obj] = forward(obj,x)
        y = vl_nnconv(x, obj.weights.w{1}, obj.weights.w{2},...
           'pad', obj.pad, 'stride', obj.stride);  
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      [dzdx, dzdf, dzdb] = vl_nnconv(...
              x, obj.weights.w{1}, obj.weights.w{2}, ...
              dzdy, 'pad', obj.pad, 'stride', obj.stride) ;  
      obj.weights.dzdw{1} = obj.weights.dzdw{1}+dzdf;
      obj.weights.dzdw{2} = obj.weights.dzdw{2}+dzdb;
    end
  end  
end