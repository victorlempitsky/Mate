function net = move( net, destination)
%MOVE Summary of this function goes here
%   Detailed explanation goes here
net = net.clearBlobs;
if ~strcmp(net.mode,destination)
  switch destination
    case 'gpu', moveop = @(x) gpuArray(x) ;
    case 'cpu', moveop = @(x) gather(x) ;
    otherwise, error('Unknown desitation ''%s''.', destination) ;
  end
  for l=1:numel(net.layers)
    net.layers{l} = net.layers{l}.move(destination);
  end

%   for n = 1:net.nBlobs
%     net.x{n} = moveop(net.x{n});
%     net.dzdx{n} = moveop(net.dzdx{n});
%   end

  net.mode = destination;
end

