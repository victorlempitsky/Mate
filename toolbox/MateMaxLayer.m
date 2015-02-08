classdef MateMaxLayer < MateLayer
  properties
    nChunks = 2;
    argmax = [];
  end
  
  methods (Static)
    function t = canTake(nIn)
      t = nIn >= 2;
    end 
  end
  
  methods
    function obj = MateMaxLayer(varargin)
      obj@MateLayer(varargin);
      obj.nChunks = numel(obj.takes);
    end
    
    function [y,obj] = forward(obj,x)
      [y, obj.argmax] = max(cat(ndims(x{1})+1,x{:}),[],ndims(x{1})+1);
    end
    
    function [dzdx,obj] = backward(obj, x, dzdy, y)                                      
      for i = 1:obj.nChunks
        dzdx{i} = dzdy .* (obj.argmax == i);
      end          
    end
    
  end
  
end