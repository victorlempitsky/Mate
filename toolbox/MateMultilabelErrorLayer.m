classdef MateMultilabelErrorLayer < MateLayer
  properties
    weight = 1;
  end

  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MateMultilabelErrorLayer(varargin)
      obj@MateLayer(varargin);
      obj.skipBackward = true;
    end
    
    function [y,obj] = forward(obj,x)
      [~,pred] = max(x{1},[],3);
      y = sum(single(pred(:) ~= x{2}(:)));
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      %will be skipped
    end
  end  
end