function x = getPlate(array, maxSz)
%turns a 3D or 4D array into a squarish 2D array displayable by imagesc via
%reshaping, subsampling, padding.
%Outputs a 2D unnormalized array if size(array,3)~=3 and a normalized
%3-channeled array otherwise.

x = squeeze(array);
assert(ndims(x) == 2 || ndims(x) == 3 || ndims(x) == 4,...
  'getPlate should be called for arrays of dimensionality 3 or 4.');

if ndims(x) == 2
  return
end

if size(array,3) == 3 
  if ndims(array) == 3
    x = cat(3, getPlate(array(:,:,1)),...
      getPlate(array(:,:,2)),...
      getPlate(array(:,:,3)));
  else
    x = cat(3, getPlate(array(:,:,1,:)),...
      getPlate(array(:,:,2,:)),...
      getPlate(array(:,:,3,:)));
  end
  x = x-min(x(:));
  x = x/max(x(:));
  return
end

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





