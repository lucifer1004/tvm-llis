#!/bin/bash

ONNX_DIR=$1
PRECONVERT_DIR=$2
POSTCONVERT_DIR=$3
CONVERTER_PATH=$4

#bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} mnist-8 1 1 28 28
#bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} arcfaceresnet100-8 1 3 112 112
#bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} ultraface320 1 3 240 320
#bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} densenet121 1 3 224 224
#bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} inception_v3 1 3 224 224
#bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} gpt2 1 3 224 224

bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} mnist-8 1 1 28 28
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} densenet-9 1 3 224 224
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} googlenet-9 1 3 224 224
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} mobilenetv2-7 1 3 224 224
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} resnet18-v2-7 1 3 224 224
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} resnet34-v2-7 1 3 224 224
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} resnet50-v2-7 1 3 224 224
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} squeezenet1.1-7 1 3 224 224
bash ./onnx2tvm_clockwork.sh ${ONNX_DIR} ${PRECONVERT_DIR} ${POSTCONVERT_DIR} ${CONVERTER_PATH} inception_v3 1 3 224 224
