classdef MateArgmaxLayer < MateLayer
% Produces the positions of the maximal elements and their values
  properties
    nTop = 1;
  end

  methods (Static)
    function t = canProduce(nOut)
      t = nOut == 2;
    end 
  end
  
  methods
    function obj = MateArgmaxLayer(varargin)
      obj@MateLayer(varargin);
      obj.skipBackward = true;
    end
    
    function [y,obj] = forward(obj,x)
      x = squeeze(x);
      if obj.nTop == 1
        [y{2},y{1}] = max(x,[],ndims(x)-1);
      else
        [srt,ord] = sort(x,ndims(x)-1,'descend');
        indx = repmat({':'},1,ndims(x));
        indx{ndims(x)-1} = '1:nTop';
        y{1} = ord(indx{:});
        y{2} = srt(indx{:});
      end
    end
  end  
end