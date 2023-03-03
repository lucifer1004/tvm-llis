#!/bin/bash

ONNX_DIR=$1
DEST_DIR=$2

#python3 onnx2tvm.py ${ONNX_DIR}/arcfaceresnet100-8.onnx ${DEST_DIR}/arcfaceresnet100-8_cuda_llis-pack.so cuda_llis 1 3 112 112
#python3 onnx2tvm.py ${ONNX_DIR}/ultraface320.onnx ${DEST_DIR}/ultraface320_cuda_llis-pack.so cuda_llis 1 3 240 320
#python3 onnx2tvm.py ${ONNX_DIR}/densenet121.onnx ${DEST_DIR}/densenet121_cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/inception_v3.onnx ${DEST_DIR}/inception_v3_cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/mnist-8.onnx ${DEST_DIR}/mnist-8_cuda_llis-pack.so cuda_llis 1 1 28 28
#python3 onnx2tvm.py ${ONNX_DIR}/gpt2.onnx ${DEST_DIR}/gpt2_cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/resnet18.onnx ${DEST_DIR}/resnet18-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/resnet34.onnx ${DEST_DIR}/resnet34-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/squeezenet1_1.onnx ${DEST_DIR}/squeezenet1_1-cuda_llis-pack.so cuda_llis 1 3 224 224

python3 onnx2tvm.py ${ONNX_DIR}/mnist-8.onnx ${DEST_DIR}/mnist-8-cuda_llis-pack.so cuda_llis 1 1 28 28
#python3 onnx2tvm.py ${ONNX_DIR}/densenet-9.onnx ${DEST_DIR}/densenet-9-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/googlenet-9.onnx ${DEST_DIR}/googlenet-9-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/mobilenetv2-7.onnx ${DEST_DIR}/mobilenetv2-7-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/resnet18-v2-7.onnx ${DEST_DIR}/resnet18-v2-7-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/resnet34-v2-7.onnx ${DEST_DIR}/resnet34-v2-7-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/resnet50-v2-7.onnx ${DEST_DIR}/resnet50-v2-7-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/squeezenet1.1-7.onnx ${DEST_DIR}/squeezenet1.1-7-cuda_llis-pack.so cuda_llis 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/inception_v3.onnx ${DEST_DIR}/inception_v3-cuda_llis-pack.so cuda_llis 1 3 224 224

#python3 onnx2tvm.py ${ONNX_DIR}/mnist-8.onnx ${DEST_DIR}/mnist-8-cuda-pack.so cuda 1 1 28 28
#python3 onnx2tvm.py ${ONNX_DIR}/densenet-9.onnx ${DEST_DIR}/densenet-9-cuda-pack.so cuda 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/googlenet-9.onnx ${DEST_DIR}/googlenet-9-cuda-pack.so cuda 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/mobilenetv2-7.onnx ${DEST_DIR}/mobilenetv2-7-cuda-pack.so cuda 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/resnet18-v2-7.onnx ${DEST_DIR}/resnet18-v2-7-cuda-pack.so cuda 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/resnet34-v2-7.onnx ${DEST_DIR}/resnet34-v2-7-cuda-pack.so cuda 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/resnet50-v2-7.onnx ${DEST_DIR}/resnet50-v2-7-cuda-pack.so cuda 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/squeezenet1.1-7.onnx ${DEST_DIR}/squeezenet1.1-7-cuda-pack.so cuda 1 3 224 224
#python3 onnx2tvm.py ${ONNX_DIR}/inception_v3.onnx ${DEST_DIR}/inception_v3-cuda-pack.so cuda 1 3 224 224

