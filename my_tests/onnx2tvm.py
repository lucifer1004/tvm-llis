import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.driver import tvmc
import sys

def change_input_dim(model, dim):
    # Use some symbolic name not used for any other dimension
    #sym_batch_dim = "N"
    # or an actal value
    #actual_batch_dim = 4 

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        # dim1.dim_param = dim
        # or update it to be an actual value:
        dim1.dim_value = dim

model_path = sys.argv[1] # path to the onnx model
output_path = sys.argv[2] # path to output the tvm-compiled model
target = sys.argv[3] # cuda or cuda_llis
#input_dims = tuple(int(x) for x in sys.argv[4:]) # e.g., `1 3 224 224` for mobilenet

onnx_model = onnx.load(model_path)
change_input_dim(onnx_model, 1)

#input_names_all = [node.name for node in onnx_model.graph.input]
#input_initializer =  [node.name for node in onnx_model.graph.initializer]
#input_names = list(set(input_names_all)  - set(input_initializer))
#
#input_name = input_names[0]

#shape_dict = {input_name: input_dims}
#mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
mod, params = relay.frontend.from_onnx(onnx_model)
#

#model = tvmc.load(model_path, shape_dict=shape_dict)
#model = tvmc.load(model_path)

#mod = model.mod
#params = model.params

opt_level = 3
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

lib.export_library(output_path)

