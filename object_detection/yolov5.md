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
