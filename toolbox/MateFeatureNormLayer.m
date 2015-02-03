classdef MateFeatureNormLayer < MateLayer
% implements local normalization across feature channels
%a wrapper around MatConvNet vl_nnnormalize
  properties
    N = 5;
    kappa = 2;
    alpha = 1e-4;
    beta = 0.75;
  end
  methods
    function obj = MateFeatureNormLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = vl_nnnormalize(x, [obj.N obj.kappa obj.alpha obj.beta]);
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = vl_nnnormalize(x, [N kappa alpha beta], dzdy);
    end
  end  
end
