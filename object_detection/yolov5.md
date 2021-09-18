# yolov5
## yolov5 tensorrt

### tensorrt中plugin


### opencv编译

**将opencv_contrib解压后文件夹放到opencv文件目录下**

mv opencv_contrib-4.4.0 opencv-4.4.0/

**新建安装目录**

cd opencv-4.4.0; 
mkdir -p build/installed; 
cd build;

**Cmake配置**

cmake -BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=/usr/local/opencv \
	-DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-3.4.14/modules \
	-DOPENCV_DNN_CUDA=True \
	-DWITH_CUDA=True \
	-DCUDA_ARCH_BIN=6.1 \
	-DBUILD_TESTS=False \
	-DOPENCV_GENERATE_PKGCONFIG=ON ..;

**编译**

make -j4; 
sudo make install;

// class
    // data_preprocess  inference, post-process
    // pybind


**gcc**

背景：
直接通过yum install gcc安装的版本4.8.5太老了，很多新的库的用不起，没办法，只有升级了。
手动编译安装太过于麻烦，于是乎网上找到了这个方法。

方法:
yum install centos-release-scl
yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version


**dcn**
DBNet中的dcn模块默认`pythons setup.py build_ext --inplace`编译后，换一块不同架构的gpu，结果就大不相同了。

原因是默认的nvcc编译过程中只针对本机环境的显卡架构进行编译，未对其他架构/计算能力的显卡做兼容处理。需要在编译过程中加入以下参数：
```python
setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ], extra_compile_args={
            "nvcc": ['-gencode=arch=compute_61,code=sm_61', '-gencode=arch=compute_75,code=sm_75',
                     '-gencode=arch=compute_72,code=sm_72', '-gencode=arch=compute_70,code=sm_70'],
            'cxx': []}
                      ),
        CUDAExtension('deform_pool_cuda', [
            'src/deform_pool_cuda.cpp', 'src/deform_pool_cuda_kernel.cu'
        ], extra_compile_args={
            "nvcc": ['-gencode=arch=compute_61,code=sm_61', '-gencode=arch=compute_75,code=sm_75',
                     '-gencode=arch=compute_72,code=sm_72', '-gencode=arch=compute_70,code=sm_70'],
            'cxx': []}),
    ],
    cmdclass={'build_ext': BuildExtension})
```