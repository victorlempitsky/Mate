classdef MateDoubleMarginLossLayer < MateLayer
  properties
    positiveWeight = single(1);
    negativeWeight = single(1);
    positiveMargin = single(1);
    negativeMargin = single(0.1);
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MateDoubleMarginLossLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = sum(max(obj.positiveMargin-x{1}(:),0).*x{2}(:).*obj.positiveWeight+...
        max(x{1}(:)-obj.negativeMargin,0).*(1-x{2}(:)).*obj.negativeWeight);
    end
    
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx{1} = zeros(size(x{1}),'like',x{1}) ;
      dzdx{1}(:) = -(obj.positiveWeight)*x{2}(:).*(x{1}(:) < obj.positiveMargin)+...
              (obj.negativeWeight)*(single(1)-x{2}(:)).*(x{1}(:) > obj.negativeMargin);
      dzdx{2} = [] ;
    end
  end  
end