classdef MateLogisticLossLayer < MateLayer
  %implements binary logistic loss layer
  %the second input should be {-1,1}-label blob
  properties
    weight = single(1);
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MateLogisticLossLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = sum(log(single(1)+exp(-x{1}.*x{2})),ndims(x{1}))*obj.weight;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      E = exp(-x{1}.*x{2});
      dzdx{1} = x{2}.*E./(E+single(1)).*(-obj.weight/numel(x{1},ndims(x{1})));
      dzdx{2} = [] ;
    end
  end  
end