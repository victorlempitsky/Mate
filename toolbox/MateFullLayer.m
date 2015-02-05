classdef MateFullLayer < MateLayer
  properties
  end
  methods
    function obj = MateFullLayer(matrix,bias,varargin)
      obj@MateLayer(varargin);
      if isempty(obj.shareWith)
        obj.weights.w{1} = matrix;
        obj.weights.w{2} = bias;
      end
      %if isempty(obj.learningRate) obj.learningRate = single([1.0 1.0]); end
      %if isempty(obj.weightDecay) obj.weightDecay = single([0.0 0.0]); end
    end
    
    function [y,obj] = forward(obj,x)
      if ~isempty(obj.weights.w{2})
        y = bsxfun(@plus,obj.weights.w{1}*x,obj.weights.w{2});
      else
        y = obj.weights.w{1}*x;
      end      
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = obj.weights.w{1}'*dzdy;
      obj.weights.dzdw{1} = obj.weights.dzdw{1}+dzdy*x';
      if ~isempty(obj.weights.w{2})
        obj.weights.dzdw{2} = obj.weights.dzdw{2}+sum(dzdy,2);
      end
    end
  end  
end