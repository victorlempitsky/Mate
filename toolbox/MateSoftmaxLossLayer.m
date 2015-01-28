classdef MateSoftmaxLossLayer < MateLayer
  properties
    weight = single(1);
    c = []
    ex = []
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
      dim = ndims(x{1})-1;
      xmax = max(x{1},[],dim);
      obj.ex = exp(bsxfun(@minus, x{1}, xmax)) ;      
      t =  xmax+log(sum(obj.ex,dim))-sum(x{1}.*x{2},dim);
      y = sum(t,ndims(t));
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx{1} = (bsxfun(@rdivide, obj.ex, sum(obj.ex,ndims(x{1})-1))-x{2}).*(obj.weight/numel(x{1},ndims(x{1}))) ;
      dzdx{2} = [] ;
    end
  end  
end

