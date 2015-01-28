classdef MateBilinearAllDistLayer < MateLayer

  methods
    function obj = MateBilinearAllDistLayer(W,varargin)
      obj@MateLayer(varargin);
      if isempty(obj.shareWith)
        obj.weights.w{1} = W;
      end      
    end
    
    function [y,obj] = forward(obj,x)
      assert(ismatrix(x));
      y = x'*obj.weights.w{1}*x;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = obj.weights.w{1}*x*dzdy.*single(2);
      obj.weights.dzdw{1} = x*dzdy*x';
    end
    
  end
  
end