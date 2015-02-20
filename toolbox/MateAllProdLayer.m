classdef MateAllProdLayer < MateLayer
  methods
    function obj = MateAllProdLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = x'*x;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)   
      dzdx = x*dzdy;
    end
  end  
end
