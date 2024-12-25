<!--
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2024-12-25 19:04:19
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2024-12-25 19:04:26
-->




Firstly you need to obtain libopenrox.so
module load cmake/3.7.2
cmake -DOPENROX_CREATE_MANUAL_PROG=OFF -DOPENROX_BUILD_EXAMPLES=ON ..
make -j128  # 128 threads

1- lib库加入 新的动态链接库, 在openrox/build目录下 (libary is added to new .so lib, under openrox/build)
gcc -fPIC -shared  libopenrox.so  ../examples/rox_odometry_module.c  -o rox_odometry_module.so  -I  /home/ziliu/openrox/sources 
2- 
cp /home/ziliu/openrox/build/libopenrox.so ~/mmvo/
复制  /home/ziliu/openrox/build/libopenrox.so 到 python项目root目录mmvo/。
为了运行时能找到 libopenrox.so （运行时，python代码会在python命令的目录下去搜索使用 libopenrox.so）


cmake -DOPENROX_CREATE_MANUAL_PROG=OFF -DOPENROX_BUILD_EXAMPLES=ON .. && make -j 128 &&   gcc -fPIC -shared  libopenrox.so  ../examples/rox_odometry_module.c  -o rox_odometry_module.so  -I  /home/ziliu/openrox/sources \
&& cp /home/ziliu/openrox/build/libopenrox.so ~/mmvo/  