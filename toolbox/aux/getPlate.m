function x = getPlate(array, maxSz)

x = squeeze(array);
assert(ndims(x) > 2,'getPlate should be called for arrays of dimensionality 3 or 4.');

if nargin == 1
  maxSz = 1024;
end

if isscalar(maxSz)
  maxSz = [maxSz maxSz];
end

if ndims(x) == 3
   t = ceil(sqrt(size(x,3)));
   x(:,:,end+1:ceil(size(x,3)/t)*t) = 0;
   x = reshape(x, size(x,1), size(x,2), t, []);
end

minx = min(x(:));

x(end+1,:,:,:) = minx;
x(:,end+1,:,:) = minx;

verSz = size(x,1)*size(x,3);
if verSz > maxSz(1)
  factor = ceil(verSz/maxSz(1));
  x = x(:,:,1:factor:end,:);
end

horSz = size(x,2)*size(x,4);
if horSz > maxSz(2)
  factor = ceil(horSz/maxSz(2));
  x = x(:,:,:,1:factor:end);
end

x = reshape(permute(x,[1 3 2 4]), size(x,1)*size(x,3), size(x,2)*size(x,4));
x(:,end) = [];
x(end,:) = [];





