function dispLayer(net,layerName)

l = getLayer(net,layerName);

figure(2000+net.layersId(layerName));
set(gcf,'Name', [layerName ' (' class(l) ')']);

displayWeights(l);