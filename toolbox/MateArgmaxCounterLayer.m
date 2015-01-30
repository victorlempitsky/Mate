classdef MateArgmaxCounterLayer < MateLayer
  properties
    count
  end

%   methods (Static)
%     function t = canProduce(nOut)
%       t = nOut == 0;
%     end 
%   end
  
  methods
    function obj = MateArgmaxCounterLayer(varargin)
      obj@MateLayer(varargin);
      obj.skipBackward = true;
      obj.count = [];
    end
    
    function [y,obj] = forward(obj,x)
      if isempty(obj.count)
        obj.count = zeros([size(x,1) 1], 'single');
      end
      
      [~,argmax] = max(x);
      obj.count = vl_binsum(obj.count, ones(size(argmax),'single'), gather(argmax));
      
      y = [];
    end
        
    function obj = displayWeights(obj)
      bar(obj.count);
    end
  end  
end