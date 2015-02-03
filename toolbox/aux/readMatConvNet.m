function [layers,classes] = readMatConvNet(fname)

f = load(fname);
layers = cell(numel(f.layers)+1,1);
if nargout > 1
  classes = f.classes;
end

layers{1} = MatePreprocessImageLayer('name','preprocessor',...
  'keepAspect', f.normalization.keepAspect,...
  'averageImage', f.normalization.averageImage,...
  'imageSize', f.normalization.imageSize,...
  'interpMethod', f.normalization.interpolation);

for i=1:numel(f.layers)
  switch f.layers{i}.type
    case 'conv'
      layers{i+1} = MateConvLayer(f.layers{i}.filters,f.layers{i}.biases,...
        'stride',f.layers{i}.stride, 'pad', f.layers{i}.pad,...
        'name',  f.layers{i}.name);
       
    case 'pool'
      layers{i+1} = MatePoolLayer('pool',f.layers{i}.pool,...
        'stride',f.layers{i}.stride, 'pad', f.layers{i}.pad,...
        'method', f.layers{i}.method, 'name', f.layers{i}.name);      
    
    case 'normalize'
      layers{i+1} = MateFeatureNormLayer('N',f.layers{i}.param(1),...
        'kappa', f.layers{i}.param(2), 'alpha',f.layers{i}.param(3),...
        'beta', f.layers{i}.param(4), 'name',  f.layers{i}.name);
      
    case 'softmax'
      layers{i+1} = MateSoftmaxLayer('name',  f.layers{i}.name);
      
    case 'loss'
      layers{i+1} = MateLogLossLayer('takes',{layers{i}.name,'input:2'},...
          'name',  f.layers{i}.name);
        
    case 'softmaxloss'
      layers{i+1} = MateSoftmaxLossLayer('takes',{layers{i}.name,'input:2'},...
          'name',  f.layers{i}.name);
        
    case 'relu'
      layers{i+1} = MateReluLayer('name', f.layers{i}.name);
      
    case 'nnoffset'
      layers{i+1} = MateFeatureOffsetLayer('alpha', f.layers{i}.param(1), ...
          'beta', f.layers{i}.param(2), 'name', f.layers{i}.name);
        
    case 'dropout'
      layers{i+1} = MateDropoutLayer('rate', f.layers{i}.rate,...
           'name',  f.layers{i}.name);
         
    otherwise
      error('Unsupported MatConvNet layer.');
  end
end

