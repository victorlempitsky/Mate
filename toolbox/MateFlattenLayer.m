classdef MateFlattenLayer < MateLayer
  methods
    function obj = MateFlattenLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = reshape(x,[size(x,1)*size(x,2)*size(x,3) size(x,4)]);
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = reshape(dzdy,size(x));
    end
  end  
end