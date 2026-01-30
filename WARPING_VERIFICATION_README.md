# Image Warping Verification for VKITTI2 Dataset

本目录包含用于验证VKITTI2数据集标注文件中图像warping正确性的工具。

## 脚本说明

### 1. `verify_warping.py` - 单样本详细验证

验证单个样本的时序图像warping，生成详细的可视化结果。

**功能：**
- 读取标注文件中的图像、深度图和绝对位姿
- 执行3D warping（Frame 0→1, 1→2, 0→2）
- 计算覆盖率（Coverage）和光度误差（Photometric Error）
- 生成包含以下内容的详细可视化：
  - 原始图像序列
  - Warped图像
  - 目标图像
  - 差异图（Difference Maps）
  - 有效像素掩码（Valid Masks）
  - 深度图可视化
  - 统计信息汇总

**使用方法：**
```bash
# 验证单个样本（默认第0个样本）
python verify_warping.py

# 指定样本索引
python verify_warping.py --sample 10

# 指定标注文件和输出目录
python verify_warping.py \
    --annotation annotations/vkitti2/weather/clone/scene01.json \
    --sample 5 \
    --save_dir debug_warp_vis
```

**参数：**
- `--annotation`: 标注JSON文件路径（默认：`annotations/vkitti2/weather/clone/scene01.json`）
- `--sample`: 要验证的样本索引（默认：0）
- `--save_dir`: 保存可视化结果的目录（默认：`debug_warp_vis`）

### 2. `batch_verify_warping.py` - 批量验证

批量验证多个样本，生成统计报告和汇总图表。

**功能：**
- 批量处理指定数量的样本
- 统计所有样本的warping质量
- 生成以下输出：
  - Coverage和Error随样本变化的曲线图
  - Coverage和Error的分布直方图
  - 详细结果CSV文件
  - 质量分类统计（GOOD/ACCEPTABLE/POOR）

**使用方法：**
```bash
# 批量验证30个样本（均匀采样）
python batch_verify_warping.py --num_samples 30

# 使用固定步长采样
python batch_verify_warping.py --num_samples 50 --stride 10

# 完整示例
python batch_verify_warping.py \
    --annotation annotations/vkitti2/weather/clone/scene01.json \
    --num_samples 30 \
    --save_dir debug_warp_vis
```

**参数：**
- `--annotation`: 标注JSON文件路径
- `--num_samples`: 要验证的样本数量（默认：20）
- `--stride`: 采样步长，如果为None则均匀采样（默认：None）
- `--save_dir`: 保存结果的目录（默认：`debug_warp_vis`）

**输出文件：**
- `batch_verification_plots.png`: Coverage和Error曲线图
- `batch_verification_distributions.png`: 统计分布直方图
- `batch_verification_results.csv`: 详细结果数据

## 评估指标

### Coverage（覆盖率）
- 定义：Warped图像中有效像素占总像素的百分比
- 范围：0-100%
- 判断标准：
  - > 50%: 优秀
  - 40-50%: 可接受
  - < 40%: 较差

### Photometric Error（光度误差）
- 定义：Warped图像与目标图像之间的平均像素强度差异
- 范围：0-255（RGB平均）
- 判断标准：
  - < 50: 优秀（考虑遮挡、动态物体、光照变化）
  - 50-70: 可接受
  - > 70: 较差

## Warping质量判断

脚本根据以下规则自动判断warping质量：

1. **GOOD（优秀）**：
   - Coverage 0→1 > 50% AND Coverage 1→2 > 50%
   - Error 0→1 < 50 AND Error 1→2 < 50

2. **ACCEPTABLE（可接受）**：
   - Coverage 0→1 > 40% AND Coverage 1→2 > 40%
   - Error可能较高（由于遮挡或动态物体）

3. **POOR（较差）**：
   - Coverage < 40%
   - Error > 70

## 验证结果示例

根据对Scene01-clone的验证结果（30个样本）：

```
Status Distribution:
  GOOD: 25 (83.3%)
  ACCEPTABLE: 3 (10.0%)
  POOR: 2 (6.7%)

Coverage Statistics:
  0→1: 65.85% ± 11.47%
  1→2: 66.53% ± 11.46%
  0→2: 55.43% ± 15.72%

Photometric Error Statistics:
  0→1: 37.68 ± 13.64
  1→2: 37.68 ± 13.83
  0→2: 47.85 ± 17.60
```

**结论：** 标注文件中的位姿和图像warping整体质量良好，83.3%的样本达到优秀标准。

## 技术细节

### Warping实现
1. 使用深度图和相机内参将源图像像素投影到3D空间
2. 通过位姿变换将3D点从源相机坐标系转换到目标相机坐标系
3. 将3D点重新投影到目标图像平面
4. 使用双线性插值生成warped图像

### 位姿格式
标注文件中的`gt_poses`格式为：
```
"r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz"
```
表示4x4变换矩阵（世界到相机）：
```
[r11 r12 r13 tx ]
[r21 r22 r23 ty ]
[r31 r32 r33 tz ]
[0   0   0   1  ]
```

### 深度图格式
- VKITTI2深度图存储为uint16 PNG格式
- 深度值（米）= PNG像素值 / 100
- 有效范围：0.01m - 655.35m

## 故障排除

### 高光度误差的可能原因
1. **动态物体**：场景中的移动车辆或行人
2. **遮挡变化**：不同视角下的遮挡关系变化
3. **光照变化**：虚拟环境中的光照差异
4. **深度估计误差**：虽然是ground truth，但仍可能有数值精度问题

### 低覆盖率的可能原因
1. **大运动**：相机移动距离较大
2. **视场边界**：大量像素投影到图像边界外
3. **深度不连续**：深度突变导致投影失败

## 依赖项

```bash
numpy
opencv-python
matplotlib
tqdm
```

安装：
```bash
pip install numpy opencv-python matplotlib tqdm
```

## 相关文件

- `verify_warping.py`: 单样本验证脚本
- `batch_verify_warping.py`: 批量验证脚本
- `debug_warp_vis/`: 输出目录（包含所有可视化结果）
- `annotations/vkitti2/`: VKITTI2标注文件目录
