classdef MateSoftmaxLayer < MateLayer
  properties
    gamma = 1;
  end
  methods
    function obj = MateSoftmaxLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      dim = ndims(x)-1;
      y = exp(bsxfun(@minus,x,max(x,[],dim)).*obj.gamma);
      y = bsxfun(@rdivide,y,sum(y,dim));
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dim = ndims(x)-1;
      dzdx = y .* bsxfun(@minus, dzdy, sum(dzdy .* y, dim)) .* obj.gamma ;
    end
  end  
end
