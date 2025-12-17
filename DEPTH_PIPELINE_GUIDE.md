# 深度图生成和对齐完整流程指南

## 📋 流程概览

```
步骤1: 生成深度图 (DAV2)
  ↓
步骤2: 生成 depth_params (对齐)
  ↓
步骤3: 在训练中使用深度信息
```

---

## 🔹 步骤1: 使用 DAV2 生成深度图

### 1.1 运行深度图生成脚本

```bash
cd /home/xjw/code/CityGS/CityGaussian/utils/Depth-Anything-V2
python horizon_gs_gen_depth.py
```

### 1.2 输出文件说明

**DAV2 会生成三种格式的深度图：**

1. **`.npy` 文件（推荐）**
   - 格式：浮点数数组 `(H, W)`
   - 特点：**无损保存**，保持 DAV2 原始输出
   - 用途：**用于后续对齐脚本**（gen_depth_params.py）
   - 值域：相对深度，值域不确定（取决于场景）

2. **`.png` 文件（16位）**
   - 格式：16位整数，值域 0-65535
   - 特点：归一化保存（min-max归一化），**丢失绝对尺度**
   - 用途：可视化检查，**不建议用于对齐**
   - 注意：如果必须使用PNG，需要反归一化，但无法恢复绝对尺度

3. **`_vis.png` 文件（可视化）**
   - 格式：8位伪彩色图
   - 用途：**仅供人眼检查**深度图质量

### 1.3 DAV2 深度图特点

- ✅ **相对深度**：保持了场景内的相对深度关系
- ❌ **无绝对尺度**：不知道1.0代表多少米
- ✅ **高质量**：使用 `vitl` 模型可以获得最佳精度
- ⚠️ **值域不确定**：不同图片的深度值范围可能差异很大

---

## 🔹 步骤2: 生成 depth_params（深度对齐）

### 2.1 运行对齐脚本

```bash
cd /home/xjw/code/horizonGS/HorizonGS
python gen_depth_params.py --source_path data/fusion/train
```

### 2.2 对齐原理

**目标**：找到单目深度图（DAV2）和 COLMAP 深度之间的线性关系

**拟合公式**：
```
1/Z_colmap = scale * D_mono + offset
```

其中：
- `Z_colmap`：COLMAP 的物理深度（米）
- `D_mono`：DAV2 的单目深度（相对值）
- `scale`：缩放系数
- `offset`：偏移量

### 2.3 对齐过程

1. **读取深度图**
   - 优先读取 `.npy` 文件（推荐）
   - 如果只有 `.png`，会尝试读取（但精度较低）

2. **匹配点对**
   - 找到 COLMAP 3D 点在图像中的位置
   - 提取对应位置的单目深度值
   - 计算 COLMAP 的物理深度

3. **RANSAC 拟合**
   - 使用 RANSAC 鲁棒拟合线性关系
   - 过滤异常值（天空、遮挡等）

4. **反向拟合（如果失败）**
   - 如果检测到负 scale，尝试反向拟合
   - 假设单目深度图是逆深度格式

### 2.4 输出文件

**`sparse/0/depth_params.json`**
```json
{
    "1": {"scale": 0.001118, "offset": 0.256919},
    "2": {"scale": 0.000715, "offset": 0.268351},
    ...
}
```

**统计信息**：
- `Success`: 成功对齐的图片数
- `Negative Scale`: 负 scale 的图片数（有问题）
- `Not Enough Points`: 点数不足的图片数
- `Sky Filter Applied`: 应用天空过滤的街景图数量

---

## 🔹 步骤3: 在训练中使用深度信息

### 3.1 深度图格式转换

在 `scene/cameras.py` 中，深度图会被转换为逆深度：

```python
# COLMAP 格式
invdepthmapScaled = gt_depth * depth_params["scale"] + depth_params["offset"]
```

**转换公式**：
```
invdepth = depth * scale + offset
```

其中 `depth` 是 DAV2 的原始深度值（从 `.npy` 文件读取）。

### 3.2 训练中的深度损失

在训练脚本中，深度损失计算：

```python
invDepth = 1.0 / render_depth  # 渲染的逆深度
mono_invdepth = viewpoint_cam.invdepthmap  # 对齐后的单目逆深度
Ll1depth = torch.abs(invDepth - mono_invdepth).mean()
```

---

## ⚠️ 常见问题排查

### Q1: 为什么有很多负 scale？

**可能原因**：
1. **天空区域干扰**：街景图的天空区域没有有效的 COLMAP 点
2. **深度图格式问题**：单目深度图可能是逆深度格式
3. **深度值范围差异**：单目深度和 COLMAP 深度值域差异太大

**解决方案**：
- ✅ 已添加天空过滤（自动过滤图像上方20%区域）
- ✅ 已添加反向拟合（自动检测逆深度格式）
- ✅ 已放宽过滤阈值（减少有效点丢失）

### Q2: "Not Enough Points" 太多？

**可能原因**：
- 天空过滤太严格，过滤掉了太多有效点
- COLMAP 重建质量不好，匹配点太少

**解决方案**：
- 检查 COLMAP 重建质量
- 调整天空过滤参数（在代码中修改）

### Q3: 应该使用 .npy 还是 .png？

**强烈推荐使用 `.npy`**：
- ✅ 无损保存，精度最高
- ✅ 保持 DAV2 原始输出
- ✅ 无需反归一化

**`.png` 的问题**：
- ❌ 16位量化误差
- ❌ 丢失绝对尺度（min-max归一化）
- ❌ 需要反归一化，但无法恢复原始值域

### Q4: 深度图值域是多少？

**DAV2 输出的深度值域不确定**：
- 不同图片的值域可能差异很大
- 这是**相对深度**，不是绝对深度
- 通过 `depth_params` 的 `scale` 和 `offset` 对齐到 COLMAP 尺度

---

## 📝 代码修改建议

### 1. 优化深度图生成脚本

已添加：
- ✅ 深度统计信息输出
- ✅ 更清晰的注释说明

### 2. 优化对齐脚本

已修复：
- ✅ PNG 格式读取（正确处理16位深度图）
- ✅ 天空过滤策略（更智能的过滤）
- ✅ 反向拟合逻辑（自动检测逆深度格式）

---

## 🚀 快速开始

```bash
# 步骤1: 生成深度图
cd /home/xjw/code/CityGS/CityGaussian/utils/Depth-Anything-V2
python horizon_gs_gen_depth.py

# 步骤2: 生成对齐参数
cd /home/xjw/code/horizonGS/HorizonGS
python gen_depth_params.py --source_path data/fusion/train

# 步骤3: 检查结果
cat data/fusion/train/sparse/0/depth_params.json
```

---

## 📚 参考

- DAV2 官方文档：https://github.com/DepthAnything/Depth-Anything-V2
- COLMAP 文档：https://colmap.github.io/
- RANSAC 算法：https://en.wikipedia.org/wiki/Random_sample_consensus

