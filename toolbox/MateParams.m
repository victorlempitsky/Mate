classdef MateParams < handle
% a container class for trainable parameters
% supports traning by SGD with momentum
% all members are cell arrays designed to contain all trainable parameters
% for a certain layer  
  properties
    w = {};
    dzdw = {};
    momentum = {};
  end
end