classdef MateLogisticLossLayer < MateLayer
  %implements binary logistic loss layer
  %the second input should be {-1,1}-label blob
  properties
    weight = single(1);
    ex = [];
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
      obj.ex = exp(-x{1}(:).*x{2}(:));
      y = sum(log(single(1)+obj.ex))*obj.weight;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx{1} = x{2}(:).*obj.ex./(obj.ex+single(1)).*(-obj.weight/numel(x{1}));
      dzdx{1} = reshape(dzdx{1}, size(x{1}));
      dzdx{2} = [] ;
    end
  end  
end