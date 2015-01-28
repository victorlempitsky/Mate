function net = makePass(net, xin, dobackward, varargin)


opts.sync = false ;
%opts.disableDropout = false ;
%opts.freezeDropout = false ;
opts = vl_argparse(opts, varargin);

%n = numel(net.schedule) ;

if (nargin <= 3) 
  dobackward = false ;
end


if net.expectIn == 0
  assert(isempty(xin));
  %gpuMode = isa(xin, 'gpuArray'); %x == either [] or gpuArray()
elseif net.expectIn == 1
  if(iscell(xin))
    assert(numel(xin) == 1);
    net.x{1} = xin{1}; 
  else
    net.x{1} = xin; 
  end
  assert( (isa(net.x{1}, 'gpuArray')) == strcmp(net.mode, 'gpu') );
  %gpuMode = isa(net.x{1}, 'gpuArray');
else
  assert(iscell(xin) && numel(xin) >= net.expectIn,...
    'The input to the net is not a cell array with a sufficient number of blobs');
%   if numel(xin) ~= net.expectIn
%     warning('The network has received more data blobs than it needs');
%   end
  for j = 1:numel(xin)
    net.x{j} = xin{j};
  end
  
  assert( (isa(net.x{1}, 'gpuArray')) == strcmp(net.mode, 'gpu') );
  %gpuMode = isa(xin{1}, 'gpuArray') ;    
end


for i=1:numel(net.forwardSchedule)
  in = net.forwardSchedule(i).in;
  out = net.forwardSchedule(i).out;
  l = net.forwardSchedule(i).layer;
  net.layers{l}.forwardTime = tic;
  
  if isscalar(in) && isscalar(out)
    [net.x{out},net.layers{l}] = forward(net.layers{l},net.x{in});
  elseif isscalar(in) && ~isscalar(out)
    [net.x(out),net.layers{l}] = forward(net.layers{l},net.x{in});
  elseif ~isscalar(in) && isscalar(out)
    [net.x{out},net.layers{l}] = forward(net.layers{l},net.x(in));
  else
    [net.x(out),net.layers{l}] = forward(net.layers{l},net.x(in));
  end
  
  if strcmp(net.mode, 'gpu') & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  
  net.layers{l}.forwardTime = toc(net.layers{l}.forwardTime);  
end


if dobackward
  for i=1:numel(net.backwardSchedule)
    in = net.backwardSchedule(i).in;
    out = net.backwardSchedule(i).out;
    l = net.backwardSchedule(i).layer;
    net.layers{l}.backwardTime = tic;

    if isscalar(in) && isscalar(out)
      [net.dzdx{in},net.layers{l}] = backward(net.layers{l},net.x{in},net.dzdx{out},net.x{out});
    elseif ~isscalar(in) && isscalar(out)
      [net.dzdx(in),net.layers{l}] = backward(net.layers{l},net.x(in),net.dzdx{out},net.x{out});
    elseif isscalar(in) && ~isscalar(out)
      [net.dzdx{in},net.layers{l}] = backward(net.layers{l},net.x{in},net.dzdx(out),net.x(out));
    else
      [net.dzdx(in),net.layers{l}] = backward(net.layers{l},net.x(in),net.dzdx(out),net.x(out));
    end

    if strcmp(net.mode, 'gpu') & opts.sync
      % This should make things slower, but on MATLAB 2014a it is necessary
      % for any decent performance.
      wait(gpuDevice) ;
    end

    net.layers{l}.backwardTime = toc(net.layers{l}.backwardTime);  
  end
end