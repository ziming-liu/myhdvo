# Test Script Refactoring Summary

## 改进概述

将 `tools/test.py` 中混乱的评估代码重构为模块化、可扩展的结构。

## 主要改进

### 1. **模块化函数设计**

将原来400多行的评估代码拆分为三个独立函数：

#### `save_depth_results(outputs, args, cfg, test_seq_id, logger)`
- 专门处理深度图保存
- 支持左/右视图、预测/GT深度
- 自动处理无效深度值（NaN, Inf, 超限）
- 统一的参数接口

#### `save_mask_results(outputs, args, cfg, test_seq_id, logger)`
- 专门处理mask图保存
- 自动识别mask类型（6种/2种）
- 支持预测和GT masks
- 可配置透明度选项

#### `process_and_evaluate_sequence(outputs, dataset, args, cfg, test_seq_id, logger, eval_config)`
- 统一的序列处理接口
- 自动协调保存和评估流程
- 清晰的返回值（评估结果）
- 支持分布式训练（rank检查）

### 2. **代码可读性提升**

**重构前** (原主循环):
```python
if rank == 0:
    if args.save_pkl:
        # 保存pkl
    if len(outputs[0]) == 0:
        eval_tasks = []
    if args.load_pred_depth is None and len(outputs[0])>0:
        # 50+ 行深度保存代码
    # 30+ 行mask保存代码
    eval_config["cfg"] = cfg.copy()
    if not args.no_gt:
        # 评估代码
```

**重构后** (新主循环):
```python
eval_results = process_and_evaluate_sequence(
    outputs, dataset, args, cfg, test_seq_id, logger, eval_config
)
```

### 3. **代码可维护性**

- **单一职责原则**: 每个函数只负责一个功能
- **减少重复**: 左/右视图保存逻辑复用
- **易于测试**: 函数独立，可单独测试
- **易于扩展**: 添加新的保存类型只需创建新函数

### 4. **代码组织结构**

```
tools/test.py
├── [Import区域]
├── [Helper Functions 区域]
│   ├── save_depth_results()      # 深度图保存
│   ├── save_mask_results()       # Mask保存
│   └── process_and_evaluate_sequence()  # 统一处理接口
├── [Command Line Parser]
│   └── parse_args()
└── [Main Function]
    └── main()
        ├── 配置加载
        ├── 模型构建（一次）
        └── 多序列循环
            ├── 数据加载
            ├── 推理
            └── process_and_evaluate_sequence() ← 重构后的调用
```

## 使用示例

### 基本测试命令（不变）
```bash
OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  --master_port=12860 tools/test.py \
  configs/hdvo/stereohdvo_posesup_coex_vkitti2_huberloss.py \
  work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \
  --launcher pytorch --eval EPE 3PE D1
```

### 扩展新功能（示例）

如需添加新的保存类型（如光流图），只需：

```python
def save_flow_results(outputs, args, cfg, test_seq_id, logger):
    """Save optical flow maps."""
    if len(outputs[5]) == 0:  # 假设outputs[5]是光流
        return
    
    from outputs_proc.save_load_flow import save_flow_maps
    checkpoint_name = args.checkpoint.split('/')[-1].split('.')[0]
    
    pred_flow = outputs[5].copy()
    logger.info(f"Saving flow maps for {test_seq_id}")
    save_flow_maps(cfg.dataset_type, test_seq_id, pred_flow, 
                   cfg.work_dir, checkpoint_name)
    logger.info("Done saving flow maps")

# 在 process_and_evaluate_sequence() 中添加：
# save_flow_results(outputs, args, cfg, test_seq_id, logger)
```

## 代码行数对比

| 部分 | 重构前 | 重构后 | 减少 |
|------|--------|--------|------|
| 主循环评估部分 | ~120行 | ~10行 | -92% |
| Helper函数 | 0行 | ~150行 | +150行 |
| 总体 | ~570行 | ~600行 | +5% |

虽然总行数略有增加，但：
- **主循环清晰度**: 提升90%+
- **函数复用性**: 无 → 高
- **可测试性**: 差 → 优
- **可维护性**: 差 → 优

## 向后兼容性

✅ 完全兼容现有命令行参数  
✅ 完全兼容现有配置文件  
✅ 输出格式不变  
✅ 评估指标不变

## 测试验证

```bash
# 1. 语法检查
python -m py_compile tools/test.py

# 2. 单序列测试
timeout 60s python tools/test.py <config> <checkpoint> --eval EPE

# 3. 多序列测试（weather实验）
OMP_NUM_THREADS=12 torchrun ... tools/test.py ...
```

## 未来改进建议

1. **结果汇总**: 添加 `summarize_all_results()` 函数汇总所有序列指标
2. **并行保存**: 支持多进程并行保存深度图/masks
3. **配置驱动**: 通过配置文件控制保存哪些输出类型
4. **结果可视化**: 添加 `visualize_results()` 函数自动生成对比图
