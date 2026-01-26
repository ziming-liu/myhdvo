# OpenROX ARM64架构编译指南

本文档详细说明如何在ARM64 (aarch64)架构的NVIDIA Orion设备上编译OpenROX库及rox_odometry_module模块。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细编译步骤](#详细编译步骤)
- [问题排查](#问题排查)
- [验证编译结果](#验证编译结果)
- [技术细节](#技术细节)

---

## 环境要求

### 硬件要求
- **架构**: ARM64 (aarch64)
- **设备**: NVIDIA Orion或其他ARM64 Linux设备

### 软件依赖
```bash
# 基础编译工具
sudo apt-get update
sudo apt-get install -y build-essential cmake gcc g++

# 数学和线程库（通常已安装）
sudo apt-get install -y libgomp1 libpthread-stubs0-dev

# 可选：文档生成工具
sudo apt-get install -y doxygen graphviz texlive-latex-base
```

### 系统信息查看
```bash
# 查看系统架构
uname -m
# 输出应为: aarch64

# 查看GCC版本
gcc --version
# 推荐: GCC 9.0 或更高
```

---

## 快速开始

### 一键编译脚本

如果您只想快速编译，直接运行自动化脚本：

```bash
cd /home/nvidia/ziming/hdvo/openrox
./build_for_arm64.sh
```

脚本会自动完成：
1. 清理旧的编译文件
2. CMake配置
3. 编译OpenROX主库
4. 编译rox_odometry_module
5. 复制库文件到HDVO目录

---

## 详细编译步骤

### 步骤1: 清理并配置CMake

```bash
# 进入OpenROX构建目录
cd /home/nvidia/ziming/hdvo/openrox/build

# 清理旧的构建文件
rm -rf *

# 运行CMake配置
cmake ..
```

**预期输出**：
```
-- Detected ARM64 architecture
OPENROX_HAS_NEON_SUPPORT : true
OPENROX_HAS_SSE_SUPPORT : false
OPENROX_HAS_AVX_SUPPORT : false
Architecture: arm64
-- Configuring for ARM64 with NEON support
Adding openmp in OPENROX
-- Configuring done
-- Generating done
```

### 步骤2: 编译OpenROX主库

```bash
# 在build目录中执行
make -j$(nproc)
```

**编译时间**：约2-5分钟（取决于CPU核心数）

**预期输出**：
```
[100%] Built target openrox
```

**验证生成的库**：
```bash
ls -lh libopenrox.so
# 输出应显示: -rwxrwxr-x ... 2.1M ... libopenrox.so

file libopenrox.so
# 输出应显示: ELF 64-bit LSB shared object, ARM aarch64
```

### 步骤3: 编译rox_odometry_module

```bash
# 进入examples目录
cd /home/nvidia/ziming/hdvo/openrox/examples

# 编译目标文件
gcc -c -fPIC rox_odometry_module.c \
    -I/home/nvidia/ziming/hdvo/openrox/sources \
    -I/home/nvidia/ziming/hdvo/openrox/build \
    -o rox_odometry_module.o

# 创建共享库（关键：设置RUNPATH为$ORIGIN）
gcc -shared -o librox_odometry_module.so rox_odometry_module.o \
    -L/home/nvidia/ziming/hdvo/openrox/build \
    -lopenrox \
    -Wl,-rpath,'$ORIGIN' \
    -lm -lpthread
```

**关键参数说明**：
- `-fPIC`: 生成位置无关代码，用于共享库
- `-I...`: 指定头文件搜索路径
- `-L...`: 指定库文件搜索路径
- `-Wl,-rpath,'$ORIGIN'`: **重要！** 设置运行时库搜索路径为同目录
- `-lm -lpthread`: 链接数学和线程库

**验证生成的模块**：
```bash
ls -lh librox_odometry_module.so
# 输出应显示: -rwxrwxr-x ... 14K ... librox_odometry_module.so

# 检查RUNPATH设置
readelf -d librox_odometry_module.so | grep RUNPATH
# 输出应显示: Library runpath: [$ORIGIN]
```

### 步骤4: 部署到HDVO目录

```bash
# 复制主库
cp /home/nvidia/ziming/hdvo/openrox/build/libopenrox.so \
   /home/nvidia/ziming/hdvo/

# 复制odometry模块
cp /home/nvidia/ziming/hdvo/openrox/examples/librox_odometry_module.so \
   /home/nvidia/ziming/hdvo/rox_odometry_module.so

# 验证复制结果
ls -lh /home/nvidia/ziming/hdvo/*.so
```

**输出应显示**：
```
-rw-rw-r-- 1 nvidia nvidia 2.1M ... libopenrox.so
-rwxrwxr-x 1 nvidia nvidia  14K ... rox_odometry_module.so
```

---

## 问题排查

### 问题1: undefined symbol: rox_matse3_new

**错误现象**：
```
OSError: undefined symbol: rox_matse3_new
```

**原因**：Python运行时找不到libopenrox.so依赖库

**解决方案**：
1. 确认两个库文件在同一目录：
   ```bash
   ls -lh /home/nvidia/ziming/hdvo/*.so
   ```

2. 检查RUNPATH设置：
   ```bash
   cd /home/nvidia/ziming/hdvo
   readelf -d rox_odometry_module.so | grep RUNPATH
   ```
   必须显示：`Library runpath: [$ORIGIN]`

3. 如果RUNPATH不正确，重新编译：
   ```bash
   cd /home/nvidia/ziming/hdvo/openrox
   ./build_for_arm64.sh
   ```

### 问题2: CMake检测到x86架构

**错误现象**：
```
OPENROX_HAS_SSE_SUPPORT : true
OPENROX_HAS_NEON_SUPPORT : false
```

**解决方案**：
1. 检查系统架构：
   ```bash
   uname -m
   # 应输出: aarch64
   ```

2. 清理CMake缓存重新配置：
   ```bash
   cd /home/nvidia/ziming/hdvo/openrox/build
   rm -rf *
   cmake ..
   ```

### 问题3: 编译时类型不匹配错误

**错误现象**：
```
error: incompatible types when initializing type 'int16x8_t'
```

**解决方案**：
这些错误已在源代码中修复。如果遇到，请确保使用最新的修改版本：
- `sources/system/vectorisation/neon.c`
- `sources/baseproc/image/remap/remap_bilinear_nomask_uchar_to_uchar/remap_bilinear_nomask_uchar_to_uchar_neon.c`

### 问题4: libopenrox.so文件大小不正确

**现象**：libopenrox.so只有几KB而不是2.1M

**解决方案**：
```bash
# 重新编译主库
cd /home/nvidia/ziming/hdvo/openrox/build
make clean
make -j$(nproc)

# 检查文件大小
ls -lh libopenrox.so
```

---

## 验证编译结果

### 验证1: 检查库架构

```bash
cd /home/nvidia/ziming/hdvo
file *.so
```

**预期输出**：
```
libopenrox.so:            ELF 64-bit LSB shared object, ARM aarch64, ...
rox_odometry_module.so:   ELF 64-bit LSB shared object, ARM aarch64, ...
```

### 验证2: 检查动态链接依赖

```bash
cd /home/nvidia/ziming/hdvo
ldd rox_odometry_module.so
```

**预期输出**：
```
linux-vdso.so.1 (...)
libopenrox.so (0x...) [应成功解析]
libc.so.6 => /lib/aarch64-linux-gnu/libc.so.6 (...)
libm.so.6 => /lib/aarch64-linux-gnu/libm.so.6 (...)
libpthread.so.0 => /lib/aarch64-linux-gnu/libpthread.so.0 (...)
libgomp.so.1 => /lib/aarch64-linux-gnu/libgomp.so.1 (...)
```

### 验证3: Python加载测试

创建测试脚本：
```python
# test_openrox.py
from ctypes import cdll

try:
    lib = cdll.LoadLibrary('/home/nvidia/ziming/hdvo/rox_odometry_module.so')
    print('✓ Library loaded successfully!')
    print('✓ ddo function found:', lib.ddo)
except Exception as e:
    print('✗ Error:', e)
```

运行测试：
```bash
cd /home/nvidia/ziming/hdvo
python test_openrox.py
```

**预期输出**：
```
✓ Library loaded successfully!
✓ ddo function found: <_FuncPtr object at 0x...>
```

### 验证4: 运行HDVO测试

```bash
cd /home/nvidia/ziming/hdvo
OMP_NUM_THREADS=12 python tools/test.py \
    configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
    --launcher none --eval EPE 3PE D1 --test_seq_id 09 --no_gt
```

如果之前出现`undefined symbol`错误，现在应该可以正常运行。

---

## 技术细节

### 架构适配说明

OpenROX已针对ARM64架构进行以下适配：

1. **SIMD指令集**：
   - ARM64使用NEON指令集
   - x86使用SSE4.2和AVX指令集
   - 代码已包含NEON实现，CMake自动选择

2. **编译器标志**：
   - ARM64: NEON内置于架构，无需额外标志
   - ARM32: 需要`-marm -mfpu=neon`标志

3. **平台检测**：
   ```cmake
   # 自动检测CPU架构
   CMAKE_SYSTEM_PROCESSOR:
   - aarch64/arm64/ARM64 → ARM64 + NEON
   - arm/ARM → ARM32 + NEON
   - 其他 → x86 + SSE/AVX
   ```

### 关键修改文件

编译成功依赖于以下文件的修改：

1. **cmake/platform/platform_linux.cmake**
   - 添加ARM架构自动检测
   - 设置NEON支持标志

2. **cmake/compiler/compiler_gnucc.cmake**
   - 区分ARM32和ARM64编译选项
   - 正确设置NEON编译标志

3. **sources/system/vectorisation/neon.c**
   - 修复NEON intrinsics类型转换

4. **sources/baseproc/image/remap/.../remap_bilinear_nomask_uchar_to_uchar_neon.c**
   - 修复uint16/int16类型不匹配

### RUNPATH vs RPATH

**为什么使用$ORIGIN**：
- `$ORIGIN`是一个特殊token，表示可执行文件或库所在的目录
- 设置`RUNPATH=$ORIGIN`使库在运行时从同目录查找依赖
- 避免需要设置系统级`LD_LIBRARY_PATH`

**验证方法**：
```bash
readelf -d rox_odometry_module.so | grep -E "RPATH|RUNPATH"
```

### OpenMP支持

OpenROX启用了OpenMP多线程支持：
- 编译标志：`-fopenmp`
- 运行时可通过`OMP_NUM_THREADS`环境变量控制线程数
- 推荐设置：`OMP_NUM_THREADS=$(nproc)` 或根据性能测试调整

---

## 附录

### 完整的编译脚本

完整的自动化脚本位于：
```
/home/nvidia/ziming/hdvo/openrox/build_for_arm64.sh
```

### 相关文档

- OpenROX源码: `/home/nvidia/ziming/hdvo/openrox/`
- 移植说明: `/home/nvidia/ziming/hdvo/openrox/README_ARM_PORT.md`
- HDVO项目: `/home/nvidia/ziming/hdvo/`

### 联系信息

如遇到问题，请检查：
1. 系统架构是否为ARM64
2. 依赖库是否安装完整
3. 编译标志是否正确
4. RUNPATH是否设置为$ORIGIN

---

**文档版本**: 1.0  
**最后更新**: 2025-01-27  
**适用平台**: ARM64 (aarch64) Linux
