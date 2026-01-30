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


# possible reliable packages

```
sudo apt-get install -y doxygen

sudo apt-get install -y texlive-latex-base texlive-latex-extra
```

```
原始openrox不支持arm架构

cd /home/nvidia/ziming/openrox


cat > cmake/platform/platform_linux.cmake << 'EOF'
MESSAGE(STATUS "\n_____ rox_open/cmake/platform/platform_linux.cmake ____________________________________________\n" )

# Force disable SSE/AVX on ARM architecture
set(OPENROX_HAS_NEON_SUPPORT false)
set(OPENROX_HAS_SSE_SUPPORT false)
set(OPENROX_HAS_AVX_SUPPORT false)

MESSAGE("OPENROX_HAS_NEON_SUPPORT : ${OPENROX_HAS_NEON_SUPPORT}")
MESSAGE("OPENROX_HAS_SSE_SUPPORT : ${OPENROX_HAS_SSE_SUPPORT}")
MESSAGE("OPENROX_HAS_AVX_SUPPORT : ${OPENROX_HAS_AVX_SUPPORT}")
EOF

# 修改install.cmake，添加ARM Linux支持
sed -i '39s/message(FATAL_ERROR/message(WARNING/' cmake/install/install.cmake

# 或者更简单：注释掉整个安装配置
sed -i '39s/^/# /' cmake/install/install.cmake

# 重新配置
cd build
rm -rf *
cmake -DOPENROX_CREATE_MANUAL_PROG=OFF -DOPENROX_BUILD_EXAMPLES=ON ..
make -j 8
```


cd /home/nvidia/ziming/hdvo/openrox/build

# 确认 libopenrox.so 存在
ls -lh libopenrox.so

# 手动编译 rox_odometry_module.so
gcc -fPIC -shared \
    ../examples/rox_odometry_module.c \
    -o rox_odometry_module.so \
    -I ../sources \
    -I . \
    -L. -lopenrox \
    -lm \
    -Wl,-rpath,'$ORIGIN'

# 复制文件到项目目录
cp rox_odometry_module.so /home/nvidia/ziming/hdvo/
cp libopenrox.so /home/nvidia/ziming/hdvo/
mkdir -p /home/nvidia/ziming/hdvo/openrox/cmake
cp libopenrox.so /home/nvidia/ziming/hdvo/openrox/cmake/
cp rox_odometry_module.so /home/nvidia/ziming/hdvo/openrox/cmake/

# 验证文件
ls -lh /home/nvidia/ziming/hdvo/*.so
ldd /home/nvidia/ziming/hdvo/rox_odometry_module.so

# 运行测试
cd /home/nvidia/ziming/hdvo
export LD_LIBRARY_PATH=/home/nvidia/ziming/hdvo:/home/nvidia/ziming/hdvo/openrox/cmake:$LD_LIBRARY_PATH

OMP_NUM_THREADS=12 \
python tools/test.py \
  configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
  work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
  --launcher none \
  --eval EPE 3PE D1 \
  --test_seq_id 09 \
  --no_gt