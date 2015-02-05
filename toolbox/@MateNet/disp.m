function disp(net)

fprintf('Network with %d layers:\n', numel(net.layers));
disp('------------------------');
for i=1:numel(net.layers)
  disp(net.layers{i});
  disp('------------------------');
end