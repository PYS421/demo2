# demo2：中文命名实体识别（NER）任务

本项目基于 **PyTorch + Hugging Face Transformers** 完成中文命名实体识别（NER）任务。

使用 `bert-base-chinese` 和 `chinese-bert-wwm` 两种中文预训练模型，在 **Weibo** 和 **MSRA** 两个中文命名实体识别数据集上进行实验。

项目自行实现数据读取、BIO 标签处理、Tokenizer 与标签对齐、动态 Padding、模型搭建、训练流程和实体级评价指标。预训练模型与 Tokenizer 使用 Hugging Face 接口加载，并使用 SwanLab 记录实验过程。

---

## 项目结构

```text
demo2/
│
├── src/
│   ├── config.py              # JSON 配置读取
│   ├── labels.py              # BIO 标签管理
│   ├── dataset.py             # 数据读取、Tokenizer、标签对齐、动态 Padding
│   ├── model.py               # BERT + Dropout + Linear 模型
│   ├── utils.py               # 随机种子、实体提取、Precision/Recall/F1
│   └── train.py               # 训练、验证、测试和 SwanLab 记录
│
├── configs/
│   ├── weibo_bert.json        # Weibo + bert-base-chinese
│   ├── weibo_wwm.json         # Weibo + chinese-bert-wwm
│   ├── msra_bert.json         # MSRA + bert-base-chinese
│   ├── msra_wwm.json          # MSRA + chinese-bert-wwm
│   └── tuning/
│       ├── learning_rate.json # 学习率调优配置
│       └── epochs.json        # Epoch 调优配置
│
├── run.py                     # 单个实验运行入口
├── tune.py                    # 参数调优入口
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 各模块说明

- `src/config.py`：读取 JSON 配置，将数据集、模型和训练参数转换为配置对象。
- `src/labels.py`：使用 `LabelManager` 管理 BIO 标签，建立 `label2id` 和 `id2label`。
- `src/dataset.py`：读取 BIO 数据，使用 Hugging Face Tokenizer 完成 Token 与 BIO 标签对齐，并实现动态 Padding。
- `src/model.py`：加载预训练 BERT，并添加 Dropout 和 Linear 分类层，完成 BIO 标签预测。
- `src/utils.py`：固定随机种子，并实现 BIO 实体提取以及实体级 Precision、Recall、F1。
- `src/train.py`：实现训练、验证、最佳模型保存、测试以及 SwanLab 实验记录。
- `configs/`：保存不同数据集、模型和训练参数的 JSON 配置。
- `run.py`：运行单个 JSON 配置对应的实验。
- `tune.py`：执行参数调优实验。

---

## 数据格式

每行保存一个 Token 和对应的实体标签，空行表示一句话结束。

例如：

```text
北 B-LOC
京 I-LOC
是 O
中 B-ORG
国 I-ORG
的 O
首 O
都 O
```

标签含义：

- `O`：不属于实体
- `B-*`：实体开始位置
- `I-*`：实体内部位置

### Weibo

Weibo 数据集包含 `PER`、`LOC`、`ORG`、`GPE` 等实体类型，并进一步区分 `.NAM` 和 `.NOM`。

共包含 **17 个 BIO 标签**。

### MSRA

MSRA 数据集包含 `LOC`、`PER`、`ORG` 三类实体。

共包含 **7 个 BIO 标签**。

---

## 数据规模

| 数据集 | 训练集 | 验证集 | 测试集 | BIO 标签数 |
|---|---:|---:|---:|---:|
| MSRA | 5000 | 1000 | 1000 | 7 |
| Weibo | 1350 | 269 | 270 | 17 |

---

## 模型结构

项目采用 **BERT + Dropout + Linear** 的序列标注结构：

```text
输入文本
   ↓
Hugging Face Tokenizer
   ↓
BERT / Chinese-BERT-WWM
   ↓
Hidden States
   ↓
Dropout
   ↓
Linear
   ↓
BIO 标签预测
```

预训练模型和 Tokenizer 使用 Hugging Face Transformers 加载。

---

## 数据处理

处理流程：

```text
原始 BIO 数据
      ↓
读取 Token 和 Label
      ↓
LabelManager
      ↓
Hugging Face Tokenizer
      ↓
Token 与 BIO 标签对齐
      ↓
DataLoader
      ↓
动态 Padding
      ↓
输入 BERT
```

特殊 Token、Padding 以及不参与标签计算的位置使用 `-100`：

```text
[CLS] → -100
[SEP] → -100
PAD   → -100
后续 subword → -100
```

`-100` 的位置在 `CrossEntropyLoss` 中被忽略，同时不参与评价指标计算。

---

## 模型与训练配置

经过参数调优后，四组正式实验统一采用：

```text
Batch Size = 16
Learning Rate = 3e-5
Epochs = 8
Max Length = 128
Dropout = 0.1
Optimizer = AdamW
Loss = CrossEntropyLoss
Random Seed = 42
```

其中：

- `Learning Rate = 3e-5` 由学习率调优得到
- `Epochs = 8` 由 Epoch 调优得到
- `Seed = 42` 用于控制实验随机性
- Test 集不参与训练和参数选择，仅用于最终评估

---

## 运行方式

进入项目根目录：

```bash
cd C:\Users\13224\Desktop\demo2
```

### 运行单个实验

#### Weibo + bert-base-chinese

```bash
python run.py --config configs/weibo_bert.json
```

#### Weibo + chinese-bert-wwm

```bash
python run.py --config configs/weibo_wwm.json
```

#### MSRA + bert-base-chinese

```bash
python run.py --config configs/msra_bert.json
```

#### MSRA + chinese-bert-wwm

```bash
python run.py --config configs/msra_wwm.json
```


---

# 参数调优

为了分析训练参数对 NER 性能的影响，选择 **Weibo + bert-base-chinese** 作为调优对象。

参数选择统一以 **Dev F1** 为依据，Test 集不参与参数选择。

## Learning Rate 调优

固定：

```text
Batch Size = 16
Epochs = 3
Max Length = 128
Dropout = 0.1
Random Seed = 42
```

测试：

```text
1e-5
2e-5
3e-5
5e-5
```

实验结果：

| Learning Rate | Best Dev F1 |
|---:|---:|
| 1e-5 | 0.6078 |
| 2e-5 | 0.6296 |
| **3e-5** | **0.6581** |
| 5e-5 | 0.6573 |

因此选择：

```text
Learning Rate = 3e-5
```

运行：

```bash
python tune.py --config configs/tuning/learning_rate.json
```

---

## Epoch 调优

固定：

```text
Dataset = Weibo
Model = bert-base-chinese
Learning Rate = 3e-5
Batch Size = 16
Max Length = 128
Dropout = 0.1
Random Seed = 42
```

测试：

```text
2
3
4
6
8
```

实验结果：

| Epoch | Best Dev F1 |
|---:|---:|
| 2 | 0.6056 |
| 3 | 0.6534 |
| 4 | 0.6722 |
| 6 | 0.6855 |
| **8** | **0.6872** |

因此选择：

```text
Epochs = 8
```

运行：

```bash
python tune.py --config configs/tuning/epochs.json
```

---

## 参数调优结论

最终选择：

```text
Learning Rate = 3e-5
Epochs = 8
```

参数调优过程中使用 **Dev F1** 选择最佳参数。

确定最佳参数后，再使用最佳参数运行四组正式实验，并使用 **Test F1** 进行最终性能比较。

---

# 评价指标

本项目采用**实体级 Precision、Recall 和 F1**。

只有一个实体的以下三个条件全部一致时，才算预测正确：

1. 实体起始位置一致
2. 实体结束位置一致
3. 实体类别一致

计算公式：

```text
Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × Precision × Recall / (Precision + Recall)
```

---

# 最终实验结果

使用最终参数：

```text
Learning Rate = 3e-5
Epochs = 8
Batch Size = 16
Max Length = 128
Dropout = 0.1
Random Seed = 42
```

四组正式实验最终以 **Test F1** 作为主要性能指标。

| 数据集 | 模型 | Best Dev F1 | **Test F1** |
|---|---|---:|---:|
| Weibo | bert-base-chinese | 0.6855 | **0.6635** |
| Weibo | chinese-bert-wwm | 0.6973 | **0.6491** |
| MSRA | bert-base-chinese | 0.9057 | **0.8947** |
| MSRA | chinese-bert-wwm | 0.9140 | **0.8910** |

---

## 结果分析

- Weibo 数据集上，`bert-base-chinese` 的 Test F1 为 **0.6635**，高于 `chinese-bert-wwm` 的 **0.6491**。
- MSRA 数据集上，`bert-base-chinese` 的 Test F1 为 **0.8947**，略高于 `chinese-bert-wwm` 的 **0.8910**。
- MSRA 数据集的整体实体识别效果高于 Weibo。
- 在本次实验设置下，`bert-base-chinese` 在两个数据集上的 Test F1 均略高于 `chinese-bert-wwm`。

---

## Train / Dev / Test 使用方式

本项目采用以下实验流程：

```text
Train
  ↓
训练模型
  ↓
Dev
  ↓
根据 Dev F1 保存最佳模型
  ↓
加载最佳模型
  ↓
Test
  ↓
最终 Test Precision / Recall / F1
```

其中：

```text
Train → 用于模型训练
Dev   → 用于模型选择和参数调优
Test  → 仅用于最终性能评估
```

Test 集不参与模型训练和参数选择。

---

## SwanLab

项目使用 SwanLab 记录训练过程、验证结果、测试结果和实验参数。

主要记录：

```text
train/loss

dev/loss
dev/precision
dev/recall
dev/f1

test/loss
test/precision
test/recall
test/f1
```

正式实验名称：

```text
weibo_bert
weibo_wwm
msra_bert
msra_wwm
```

参数调优实验也会单独记录到 SwanLab。

---

## 项目总结

本项目完成了：

- `bert-base-chinese` 与 `chinese-bert-wwm` 两种中文预训练模型实验
- Weibo 与 MSRA 两个中文 NER 数据集实验
- BIO 标签处理
- Tokenizer 与 BIO 标签对齐
- 动态 Padding
- BERT + Dropout + Linear 序列标注模型
- Train / Dev / Test 完整实验流程
- 实体级 Precision、Recall、F1
- 随机种子控制
- JSON 配置管理
- Learning Rate 参数调优
- Epoch 参数调优
- SwanLab 实验记录

项目采用模块化设计，将数据处理、标签管理、模型搭建、训练、评价和实验配置分别组织，便于进行不同模型、不同数据集以及不同训练参数的实验。