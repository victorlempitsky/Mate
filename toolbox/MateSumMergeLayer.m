classdef MateSumMergeLayer < MateLayer
  properties
    nIn = 2;
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn >= 2;
    end 
  end
  
  methods
    function obj = MateSumMergeLayer(varargin)
      obj@MateLayer(varargin);
      obj.nIn = numel(obj.takes);
    end
    
    function [y,obj] = forward(obj,x)
      y = x{1};
      obj.nIn = numel(x);
      for i = 2:obj.nIn
        y = y+x{i};
      end
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = cell(1,obj.nIn);
      for i=1:obj.nIn
        dzdx{i} = dzdy;
      end
    end
    
  end
  
end