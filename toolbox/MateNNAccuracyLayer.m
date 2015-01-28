classdef MateNNAccuracyLayer < MateLayer
%takes the matrix of pairwise distances/similarities
%and the matrix of same/diff class labels
%computes the mean accuracy of the 1-NN classifier
  
  properties
    invert = false;
   end

  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MateNNAccuracyLayer(varargin)
      obj@MateLayer(varargin);
      obj.skipBackward = true;
    end
    
    function [y,obj] = forward(obj,x)
      if obj.invert
        [~,lab] = max(x{1}-eye(size(x{1}),'like',x{1}).*single(1e10));
      else
        [~,lab] = min(x{1}+eye(size(x{1}),'like',x{1}).*single(1e10));       
      end
      %todo: make more efficient in  GPU mode
      y = mean((x{2}(sub2ind(size(x{2}),[1:numel(lab)],lab))));
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      error('Undefined');%will be skipped
    end
  end  
end