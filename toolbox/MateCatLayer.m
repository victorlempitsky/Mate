classdef MateCatLayer < MateLayer
  properties
    nChunks = 2;
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn >= 2;
    end 
  end
  
  methods
    function obj = MateCatLayer(varargin)
      obj@MateLayer(varargin);
      obj.nChunks = numel(obj.takes);
    end
    
    function [y,obj] = forward(obj,x)
      y = cat(ndims(x{1})-1, x{:});
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)
      dzdx = cell(1,obj.nChunks);
      sliceDim = ndims(x{1})-1;
      chunkSz = size(x{1},sliceDim);
      assert(chunkSz*obj.nChunks == size(dzdy,sliceDim));
      index = repmat({':'},1,ndims(x{1}));                                           
      for i = 1:obj.nChunks
        index{sliceDim} = (i-1)*chunkSz+1:i*chunkSz;      
        dzdx{i} = dzdy(index{:});
      end          
    end
    
  end
  
end