# TVM with LLIS Support

## Build

Follow the instruction of the original TVM. It should be built with CUDA and LLVM support. To do that, modify config.cmake to change `set(USE_CUDA OFF)` to `set(USE_CUDA ON)` and `set(USE_LLVM OFF)` to `set(USE_LLVM ON)`.

Also, add the following to config.cmake to enable LLIS support:
```
set(USE_LLIS /path/to/llis)
```
Inside `/path/to/llis`, it should include a directory called `include` which contains the headers of LLIS, and a directory called `lib`, which contains the libraries of LLIS.

After installing, manually copy some header files that CMake fails to install:
```
cp -r 3rdparty/dmlc-core/include/dmlc "${INSTALL_PREFIX}/include"
cp -r 3rdparty/dlpack/include/dlpack "${INSTALL_PREFIX}/include"
```

When executing a TVM script that targets `cuda-llis`, the environment variable `LLIS_PATH` should be set to `/path/to/llis`.



