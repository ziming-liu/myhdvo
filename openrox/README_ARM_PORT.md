# OpenROX ARM架构移植总结

## ✓ 问题已解决

**错误**: `OSError: undefined symbol: rox_matse3_new`

**原因**: Python运行时找不到`libopenrox.so`依赖库

**解决方案**: 重新编译`rox_odometry_module.so`，设置RUNPATH为`$ORIGIN`，使其能在同目录找到`libopenrox.so`

## 快速使用

### 当前状态
✓ 库文件已正确编译并部署：
- `/home/nvidia/ziming/hdvo/libopenrox.so` (2.1M, ARM64)
- `/home/nvidia/ziming/hdvo/rox_odometry_module.so` (14K, ARM64)

✓ Python可以正确加载库和调用ddo函数

### 如需重新编译
```bash
cd /home/nvidia/ziming/hdvo/openrox
./build_for_arm64.sh
```

## 关键修改

### 1. 平台配置 - 支持ARM架构
文件: `cmake/platform/platform_linux.cmake`
- 添加ARM64/ARM32/x86架构自动检测
- 根据架构启用NEON或SSE/AVX指令集

### 2. 编译器配置 - ARM64优化
文件: `cmake/compiler/compiler_gnucc.cmake`
- ARM64: NEON内置，无需特殊标志
- ARM32: 添加`-marm -mfpu=neon`标志

### 3. NEON代码修复
文件: `sources/system/vectorisation/neon.c`
- 修复类型转换错误
- 移除不必要的vreinterpretq调用

### 4. 动态链接配置
- 设置RUNPATH为`$ORIGIN`
- 确保运行时能找到依赖库

## 技术细节

### 编译命令
```bash
# 编译rox_odometry_module.so with correct rpath
gcc -shared -o librox_odometry_module.so rox_odometry_module.o \
    -L/home/nvidia/ziming/hdvo/openrox/build \
    -lopenrox \
    -Wl,-rpath,'$ORIGIN' \
    -lm -lpthread
```

### 验证依赖
```bash
cd /home/nvidia/ziming/hdvo
readelf -d rox_odometry_module.so | grep RUNPATH
# 输出: Library runpath: [$ORIGIN]

ldd rox_odometry_module.so | grep openrox
# 应显示 libopenrox.so 被找到
```

## 重要注意事项

⚠️ **两个.so文件必须在同一目录**
- `libopenrox.so`
- `rox_odometry_module.so`

⚠️ **不要单独移动库文件**
- 如需移动，必须同时移动两个文件到同一目录

## 构建环境
- **系统**: Ubuntu on ARM64 (NVIDIA Orion)
- **架构**: aarch64
- **编译器**: GCC 9.4.0
- **SIMD**: NEON (ARM64原生支持)
- **OpenMP**: 已启用

## 日期
2025-01-27
