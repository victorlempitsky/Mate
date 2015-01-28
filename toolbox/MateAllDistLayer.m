classdef MateAllDistLayer < MateLayer
  methods
    function obj = MateAllDistLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      sz = size(x);
      x_ = reshape(x, prod(sz(1:end-1)), sz(end));
      len = sum(x_.*x_);
      y = bsxfun(@plus, len, len').*single(0.5) - x_'*x_;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      sz = size(x);
      x_ = reshape(x, prod(sz(1:end-1)), sz(end));      
      dzdx = reshape( bsxfun(@times, x_, sum(dzdy,1)) - x_*dzdy, sz );
    end
  end  
end
