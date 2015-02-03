classdef MatePreprocessImageLayer < MateLayer
  properties
    averageImage = [];
    imageSize = [224 224 3];
    interpMethod = 'bicubic';
    keepAspect = 1;
  end
  methods
    function obj = MatePreprocessImageLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      assert(ndims(x) == 3 && size(x,3) == 3);
      
      if obj.keepAspect
        factor = max([obj.imageSize(1)/size(x,1) obj.imageSize(2)/size(x,2)]);
        y = imresize(x, factor, 'method', obj.interpMethod);
        y = y(floor(end/2-obj.imageSize(1)/2)+1:floor(end/2+obj.imageSize(1)/2),...
          floor(end/2-obj.imageSize(2)/2)+1:floor(end/2+obj.imageSize(2)/2),:);
      else
        y = imresize(x, obj.imageSize(1:2), 'method', obj.interpMethod);
      end
      y = reshape(single(y)-obj.averageImage, [size(y) 1]);
    end 
  end  
end