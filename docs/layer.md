## Defining a new layer

To define a new layer, derive a new class from `MateLayer` (which is a base class (superclass) for all layers in Maté).
After this derivation, your layer will have such properties as `name`,`takes`,`shareWith` that are all discussed in
the [basics section](network.md).

Then, implement the following steps:

* If your layer has some meta-parameters (not trainable by SGD) define them as the properties of new class, e.g.:
  ```matlab
  classdef MateFooLayer < MateLayer
    properties
      bar = 100;
    end
    methods
  ....
    end  
  end
  ```

* If your layer has some learnable parameters (that will be trained by SGD), do NOT define them as properties. Instead, they
should be handled using the `weights` variable defined in `MateLayer`. Thus, `weights.w{1}` should refer to the first learnable
parameter, `weights.w{2}` to the second, etc. Likewise, `weights.dzdw{i}` refers to the partial derivatives w.r.t. these parameters.

* Define a constructor, make sure to call the base class constructor:
  ```matlab
  classdef MateFooLayer < MateLayer
  .....
    methods
      function obj = MateFooLayer(varargin)
        obj@MateLayer(varargin);
      end
  ....
    end  
  end
  ```
  Keep, the call to the base constructor as it is shown above. This will allow
  to set properties using `'propertyName',propertyValue` format. This will work
  both for base properties and the new properties. E.g. with such call it is possible to create the 
  new layer using `foo = MateFooLayer('name','foooo','bar',101);`.

* Define a **forward** member function that converts input `x` into the output `y` during forward propagation,e.g.:
  ```matlab
  function [y,obj] = forward(obj,x)
    y = x.*obj.weights.w{1}*obj.bar;
  end
  ```

* If your layer does not participate in backprop (e.g. it measures some sort of error), 
add `obj.skipBackward = true;` to the constructor.

* Otherwise, redefine the **backward** member function that propagates the partial derivative `dzdy` further backwards into `dzdx`, 
and also computes (and adds to the current state of) the `dzdw` partial derivative:
  ```matlab
  function [dzdx,obj] = backward(obj, x, dzdy, y)
    dzdx = obj.bar*dzdy.*obj.weights.w{1} ;
    obj.weights.dzdw{1} = obj.weights.dzdw{1}+obj.bar*x.*dzdy;
  end
  ```
  Note, that `dzdw` should be updated by addition (not by assignment) in order to allow parameter sharing.

* If you are defining a **loss layer** than the `backward` operator should **ignore** the `dzdy` variable, and
simply define the derivative of the loss `dzdx`.

* **Multiple inputs and outputs.** By default, it is assumed that each layer takes one input and one output. 
  If it is not the case for your layer, then redefine the static functions `canTake` and/or `canProduce` that should
  check whether the number of input or output parameters is acceptable by returning an appropriate boolean variable, e.g.:
  ```matlab
  methods (Static)
    function t = canProduce(nOut)
      t = nOut >= 2;
    end 
  end
  ```
  This checks that the layer's output is used by at least two layers when the network is constructed.

* If your layer has more than one input and/or output, then the corresponding arrays among `x`,`y`,`dzdx`,`dzdy` should
be treated as cell arrays, e.g. for a layer that sums the two inputs:
  ```matlab
  function [y,obj] = forward(obj,x)
    y = x{1}+x{2};
  end

  function [dzdx,obj] = backward(obj, x, dzdy, y)
    dzdx{1} = dzdy;
    dzdx{2} = dzdy;
  end
  ```

