classdef MateSoftmaxLossLayer < MateLayer
  properties
    weight = single(1);
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MateSoftmaxLossLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = vl_nnsoftmaxloss(x{1}, x{2}).*obj.weight;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx{1} = vl_nnsoftmaxloss(x{1}, x{2}, obj.weight) ;
      dzdx{2} = [] ;
    end
  end  
end