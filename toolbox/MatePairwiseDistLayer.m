classdef MatePairwiseDistLayer < MateLayer
  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MatePairwiseDistLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = single(0.5)*sum((x{1}-x{2}).^2, ndims(x{1})-1);
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx{1} = bsxfun(@times,x{1}-x{2},dzdy);
      dzdx{2} = bsxfun(@times,x{2}-x{1},dzdy);
    end
  end  
end