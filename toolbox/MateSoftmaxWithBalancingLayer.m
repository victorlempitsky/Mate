classdef MateSoftmaxWithBalancingLayer < MateLayer
  properties
    gamma = single(1);
    epsilon = single(0.01);
  end
  methods
    function obj = MateSoftmaxWithBalancingLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      dim = ndims(x)-1;
      y = exp(bsxfun(@minus,x,max(x,[],dim)).*obj.gamma);
      y = bsxfun(@rdivide,y,sum(y,dim));
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dim = ndims(x)-1;
      dzdy = bsxfun(@plus, dzdy, (sum(y,dim+1)-...
            single(size(y,dim+1)/size(y,dim))).*obj.epsilon);
      dzdx = y .* bsxfun(@minus, dzdy, sum(dzdy .* y, dim)) .* obj.gamma ;
    end
  end  
end
