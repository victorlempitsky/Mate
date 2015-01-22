classdef MateSliceLayer < MateLayer
  properties
    nChunks = 2;
    sliceDim = 3;
  end
  
  methods (Static)
    function t = canProduce(nChunks)
      t = nChunks >= 2;
    end 
  end
  
  methods
    function obj = MateSliceLayer(varargin)
      obj@MateLayer(varargin);
    end
    
    function [y,obj] = forward(obj,x)
      y = cell(1,obj.nChunks);
      chunkSz = size(x,obj.sliceDim)/obj.nChunks;
      index = repmat({':'},1,ndims(x));                                           
      for i = 1:obj.nChunks
        index{obj.sliceDim} = (i-1)*chunkSz+1:i*chunkSz;      
        y{i} = x(index{:});
      end
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = cat(obj.sliceDim,dzdy{:});
    end
    
  end
  
end