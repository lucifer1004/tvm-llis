import onnx
import tvm
from tvm import te
import tvm.relay as relay
import sys

def change_batch_size(model, dim):
    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = dim

model_path = sys.argv[1] # path to the onnx model
output_path = sys.argv[2] # path to output the tvm-compiled model
target = sys.argv[3] # cuda or cuda_llis
batch_size = int(sys.argv[4]) # batch size

onnx_model = onnx.load(model_path)
change_batch_size(onnx_model, batch_size)

mod, params = relay.frontend.from_onnx(onnx_model)

opt_level = 3
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

lib.export_library(output_path)

