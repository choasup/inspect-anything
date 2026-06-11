# 数据目录约定

```
data/
  README.md            # 本说明（入库）
  examples/            # 各任务格式样例，可直接照抄（入库）
  distilled.jsonl      # 你的离线蒸馏数据，TrainingSample 格式（不入库）
  gold.jsonl           # 飞轮人审金标，TrainingSample 格式（不入库）
  public/              # 公开数据集原始标注，如 COCO json（不入库）
  images/              # 训练图像根目录（不入库）
  train_swift.jsonl    # mix + export 的产物（不入库，可随时重新生成）
bench/
  v0.1.jsonl           # InspectBench 评测标注（小文件，入库）
  images/              # 评测图像（不入库）
```

大文件（图像、训练 JSONL）不进 git；入库的只有说明、样例和评测标注。
图像字段写**相对于图像根目录的路径**，加载时由 `image_root` 拼接。

## 训练数据格式（TrainingSample，一行一条 JSON）

字段：`sample_id`（全局唯一）/ `image`（相对路径）/ `task` / `target` / `source`。

- `task` ∈ `ground | count | compare | vqa | judge`
- `source` ∈ `public | distilled | gold`
- `target` 按任务校验，**坐标一律为归一化 xyxy，取值 [0,1]**（像素坐标 ÷ 图像宽高）

各任务的 `target` 必填字段：

| task | target 必填 | 说明 |
|------|------------|------|
| ground | `query`, `boxes[{x1,y1,x2,y2,label,score}]` | 标注数据 score 填 1.0 |
| count | `query`, `count`, `points[{x,y}]` | points 可为空列表 |
| judge | `rule`, `passed` | 可选 `confidence`, `reason`, `evidence[ScoredBox]` |
| vqa | `question`, `answer` | 可选 `region{x1,y1,x2,y2}` |
| compare | `matches`, `diffs[]` | 另需 `reference_image`（第二张图相对路径）、`instruction` |

完整样例见 `examples/distilled.example.jsonl`（五种任务各一条）。

## 评测数据格式（InspectBench，一行一条 JSON）

字段：`id` / `subset`（`retail | security | industrial`）/ `task` / `image` /
任务输入（`query` 或 `rule` 或 `question`）/ `target`（真值，结构同上但不含 query 等输入字段）。

样例见 `examples/bench.example.jsonl`。
**评测样本的图像与 id 必须与训练集隔离**，去重在数据管线强制。

## 自助校验

转换完成后先校验再训练：

```bash
python -m inspect_one.data.validate data/distilled.jsonl            # 训练数据
python -m inspect_one.data.validate bench/v0.1.jsonl --bench        # 评测数据
```

输出每个错误行的行号与原因；全部通过才会退出码 0。
