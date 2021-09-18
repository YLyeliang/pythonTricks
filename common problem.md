### paddle
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found

查看：strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX

更新libstdc++.so.6，下载地址：http://www.vuln.cn/wp-content/uploads/2019/08/libstdc.so_.6.0.26.zip

替换/usr/lib64中的libstdc++.so.6软连接

### pytorch
编译DBnet中的deform conv时，报AT_CHECK错误，原因是torch1.6以上更换了语法，改为TORCH_CHECK

### 随机数
在numpy中，使用多进程时，其每个进程的随机数种子默认是一致的，均fork于主进程。因此，要在每个进程使用不同的随机数，尽量使用
python内置的random函数，或是每个进程传入独立的随机状态np.random.RandomState()