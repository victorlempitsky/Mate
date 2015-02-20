classdef MateL2NormalizeLayer < MateLayer
% normalizes each vector in L2 norm (flattened state only)
  properties
    N = 5;
    kappa = 0;
    alpha = 1;
    beta = 0.5;
  end
  methods
    function obj = MateL2NormalizeLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      assert(ismatrix(x));
      y = bsxfun(@times, x, 1./sqrt(sum(x.*x)+single(1e-12)));
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      len_ = 1./sqrt(sum(x.*x)+single(1e-12));    
      dzdy_ = bsxfun(@times,dzdy,len_.^3); 
      dzdx = bsxfun(@times,dzdy,len_)-bsxfun(@times,x,sum(x.*dzdy_));
%       for i=1:size(x,2)
%         dzdx(:,i) = dzdx(:,i)-x(:,i)*x(:,i)'*dzdy_(:,i);
%       end
    end
  end  
end