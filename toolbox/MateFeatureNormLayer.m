classdef MateFeatureNormLayer < MateLayer
% implements local normalization across feature channels
%a wrapper around MatConvNet vl_nnnormalize
  properties
    N = 5;
    kappa = 0;
    alpha = 1;
    beta = 0.5;
  end
  methods
    function obj = MateFeatureNormLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      assert(ndims(x) > 3);
      y = vl_nnnormalize(x, [obj.N obj.kappa obj.alpha obj.beta]);
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = vl_nnnormalize(x, [obj.N obj.kappa obj.alpha obj.beta], dzdy);
    end
  end  
end
