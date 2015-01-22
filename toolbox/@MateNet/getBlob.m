function [x, dzdx] = getBlob(net, blobName)

assert(isKey(net.blobsId, blobName));
x = net.x{net.blobsId(blobName)};
if nargout >= 2
  dzdx = net.dzdx{net.blobsId(blobName)};
end