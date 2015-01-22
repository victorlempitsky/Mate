classdef MatePairwiseLossLayer < MateLayer
  properties
    positiveWeight = single(1);
    negativeWeight = single(1);
    margin = single(1);
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MatePairwiseLossLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = sum(x{1}(:).*x{2}(:).*obj.positiveWeight+...
            (1-x{2}(:)).*max(obj.margin-x{1}(:),0).*obj.negativeWeight);    
%     y = sum(squeeze(x{1}).*x{2}')*obj.positiveWeight+...
%            sum((gpuArray(1)-x{2}').*max(obj.margin-squeeze(x{1}),gpuArray(0)))*obj.negativeWeight;    
          
    end
    
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx{1} = zeros(size(x{1}),'like',x{1}) ;
      dzdx{1}(:) = obj.positiveWeight*x{2}(:)-...
              obj.negativeWeight*(single(1)-x{2}(:)).*(x{1}(:) < obj.margin);
      %dzdx{1}(~x{2}(:) & squeeze(x{1}) < obj.margin) = -;
      dzdx{2} = [] ;
    end
  end  
end