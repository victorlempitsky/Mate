classdef MateRankErrorLayer < MateLayer
  properties
    invert = false;
    buf1 = [];
    buf2 = [];
    bufneq = [];
  end

  methods (Static)
    function t = canTake(nIn)
      t = nIn == 2;
    end 
  end
  
  methods
    function obj = MateRankErrorLayer(varargin)
      obj@MateLayer(varargin);
      obj.skipBackward = true;
    end
    
    function [y,obj] = forward(obj,x)
      obj.buf1 = bsxfun(@lt,x{1}(:),x{1}(:)');
      if obj.invert
        obj.buf2 = bsxfun(@gt,x{2}(:),x{2}(:)');
      else
        obj.buf2 = bsxfun(@lt,x{2}(:),x{2}(:)');
      end
      obj.bufneq = bsxfun(@ne,x{2}(:),x{2}(:)'); %in case the second argument is discrete
                                          %  omits same-valued samples from
                                          %  the comparison
      n = sum(obj.bufneq(:)+single(1e-6));
      y = sum(abs(obj.buf1(:)-obj.buf2(:)).*obj.bufneq(:)./n);
    end
    
  end  
end