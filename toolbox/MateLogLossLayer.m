classdef MateLogLossLayer < MateLayer
  %implements multinomial logistic loss layer
  %the second input should be {0,1}-label blob marking the correct label
  %with 1
  
  %NOT TESTED
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
    function obj = MateLogLossLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = -sum(sum(log(x{1}).*x{2},ndims(x{1})-1),ndims(x{1})-1).*obj.weight;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx{1} = -x{2}.*(single(1)./x{1}).*obj.weight;
      dzdx{2} = [] ;
    end
  end  
end

