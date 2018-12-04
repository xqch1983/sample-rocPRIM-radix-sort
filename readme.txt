I wrote one sample, which can deliver to baidu, testing perf of hip radi_sort named test benchmark_hip_device_radix_sort as attached, which will gain 1.17ms per 1M pairs [key==float, value==int], 1.13ms per 1M pairs [key==int, value==float], on MI8 (IP:10.67.10.126, centos7.4 + rocm 1.9 + rocmPrim + gcc 7.3).

usage:
1.It bases on our Hcc, So install Hcc firstly
---- https://github.com/RadeonOpenCompute/hcc
2.untar sample_radix_qcxie
--- tar xvf sample_radix_qcxie.tgz
3.build 
-- make sure the gcc version(gcc version 5.4.0 in Ubuntu 16.04; gcc version 7.3.in centos 7.4, or type: scl enable devtoolset-7 bash)
--- sh build.sh
4.run
--- ./benchmark_hip_device_radix_sort --size 1000000 --trials 5  or  --help for usage
