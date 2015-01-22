function net = dispTimes(net)

forw = zeros(numel(net.layers),1);
backw = zeros(numel(net.layers),1);
captions = cell(1,numel(net.layers));

for i = 1:numel(net.layers)
  forw(i) = net.layers{i}.forwardTime;
  backw(i) = net.layers{i}.backwardTime;
  captions{i} = net.layers{i}.name;
end

figure(100);
set(gcf,'Name','Layer times');
subplot(1,2,1);
barh(forw,'stacked');
set(gca, 'YTick', 1:numel(net.layers));
set(gca,'YTickLabel', captions);
title('Forward times');
subplot(1,2,2);
barh(backw,'stacked');
set(gca, 'YTick', 1:numel(net.layers));
set(gca,'YTickLabel', captions);
title('Backward times');


disp(['Total forward time ' num2str(sum(forw)) ' s']);
disp(['Total backward time ' num2str(sum(backw)) ' s']);
  