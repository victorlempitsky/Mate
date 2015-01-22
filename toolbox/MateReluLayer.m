classdef MateReluLayer < MateLayer
  properties

  end
  methods
    function obj = MateReluLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = max(x, single(0));
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = dzdy.*(x > 0);
    end
  end  
end