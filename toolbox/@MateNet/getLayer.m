function l = getLayer(net, layerName);

assert(isKey(net.layersId, layerName));
l = net.layers{net.layersId(layerName)};