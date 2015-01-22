classdef MateDropoutLayer < MateLayer
  properties
    rate = 0.5;
    mask = [];
    freeze = false;
    disable = false;
  end
  methods
    function obj = MateDropoutLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function freezeMask(obj,t)
      if nargin >= 2
        obj.freeze = t;
      else
        obj.freeze = true;
      end
    end
    
    function disableDropout(obj,t)
      if nargin >= 2
        obj.disable = t;
      else
        obj.disable = false;
      end
    end    
      
    
    function [y,obj] = forward(obj,x)
      if obj.disable
        y = x ;
      elseif obj.freeze && ~isempty(mask)
        [y, obj.mask] = vl_nndropout(x, 'rate', obj.rate, 'mask', obj.mask);
      else
        [y, obj.mask] = vl_nndropout(x, 'rate', obj.rate);
      end
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
        if obj.disable
          dzdx = dzdy ;
        else
          dzdx = vl_nndropout(x, dzdy, 'mask', obj.mask) ;
        end
    end
  end  
end