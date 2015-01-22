classdef MateSplitLayer < MateLayer
  properties
    nCopies = 2;
  end
  
  methods (Static)
    function t = canProduce(nOut)
      t = nOut >= 2;
    end 
  end
  
  methods
    function obj = MateSplitLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = cell(1,obj.nCopies);
      for i = 1:obj.nCopies
        y{i} = x;
      end
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = dzdy{1};
      for i=2:obj.nCopies
        dzdx = dzdx+dzdy{i};
      end
    end
    
  end
  
end