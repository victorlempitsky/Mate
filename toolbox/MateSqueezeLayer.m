classdef MateSqueezeLayer < MateLayer
  methods
    function obj = MateSqueezeLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = squeeze(x) ;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = reshape(dzdy,size(x));
    end
  end  
end