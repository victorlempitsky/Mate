classdef MateBilinearLayer < MateLayer
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MateBilinearLayer(W,varargin)
      obj@MateLayer(varargin);
      if isempty(obj.shareWith)
        obj.weights.w{1} = W;
      end      
    end
    
    function [y,obj] = forward(obj,x)
      assert(ismatrix(x));
      y = sum((obj.weights.w{1}*x{1}).*x{2});
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
%       for i=1:size(x{1},2)
%         m = x{1}(:,i)*x{2}(:,i)'*(dzdy(i)*single(0.5));
%         obj.weights.dzdw{1} = obj.weights.dzdw{1}+m+m';
%       end
      m = x{1}*bsxfun(@times,x{2}',dzdy(:).*single(0.5));
      obj.weights.dzdw{1} = obj.weights.dzdw{1}+m+m';
      
      dzdx = {zeros(size(x{1}),'like',x{1}),zeros(size(x{2}),'like',x{2})};
      dzdx{1}(:) = bsxfun(@times,(obj.weights.w{1}*x{2}),dzdy);
      dzdx{2}(:) = bsxfun(@times,(obj.weights.w{1}*x{1}),dzdy);
    end
    
  end
  
end