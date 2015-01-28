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
      [~,pred] = max(x{1},[],ndims(x{1})-1);
      [~,gt] = max(x{2},[],ndims(x{1})-1);
      y = sum(gt(:)~=pred(:))/numel(gt);
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      %will be skipped
    end
  end  
end