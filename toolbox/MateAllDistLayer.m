classdef MateAllDistLayer < MateLayer
  methods
    function obj = MateAllDistLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      assert(ismatrix(x));
      len = sum(x.*x);
      y = bsxfun(@plus, len, len').*single(0.5) - x'*x;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)   
      dzdx = bsxfun(@times, x, sum(dzdy,1));
    end
  end  
end
