classdef MateWeightedLogisticLossLayer < MateLayer
  %implements binary logistic loss layer with ellementwise weights (third
  %blob)
  %the second input should be {-1,1}-label blob
  properties
   ex = [];
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn == 3;
    end 
  end
  
  methods
    function obj = MateWeightedLogisticLossLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      obj.ex = exp(-x{1}(:).*x{2}(:));
      y = sum(log(single(1)+obj.ex).*x{3}(:));
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx{1} = x{2}.*reshape(obj.ex./(obj.ex+single(1)),size(x{1})).*x{3};
      dzdx{2} = [];
      dzdx{3} = [];
    end
  end  
end