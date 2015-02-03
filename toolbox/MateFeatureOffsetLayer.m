classdef MateFeatureOffsetLayer < MateLayer
% implements local normalization across feature channels
%derived from MatConvNet vl_nnnoffset
  properties
    alpha = single(1.0);
    beta = single(0.5);
  end
  methods
    function obj = MateFeatureOffsetLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      L = sum(x.^2,3) ;
      L = max(L, single(1e-8)) ;      
      y = bsxfun(@minus, x, obj.alpha*L.^obj.beta) ;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      L = sum(x.^2,3) ;
      L = max(L, single(1e-8)) ;      
      dzdx = dzdy - bsxfun(@times, (2*obj.alpha*obj.beta)* x,...
        sum(dzdy,3) .* (L.^(obj.beta-1))) ;
    end
  end  
end

