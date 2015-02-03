classdef MatePoolLayer < MateLayer
% implements spatial pooling within CNN
%a wrapper around MatConvNet pooling
  properties
    pool = [2,2];
    pad = 0;
    stride = 1;
    method = 'max';
  end
  methods
    function obj = MatePoolLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = vl_nnpool(x, obj.pool,...
        'pad', obj.pad, 'stride', obj.stride, 'method', obj.method) ;
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = vl_nnpool(x, obj.pool, dzdy, ...
        'pad', obj.pad, 'stride', obj.stride, 'method', obj.method) ;
    end
  end  
end

