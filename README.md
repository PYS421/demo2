# demo2：中文命名实体识别（NER）任务

本项目基于 **PyTorch + Hugging Face Transformers** 完成中文命名实体识别（NER）任务。

使用 `bert-base-chinese` 和 `chinese-bert-wwm` 两种中文预训练模型，在 **Weibo** 和 **MSRA** 两个中文命名实体识别数据集上进行训练与验证。

项目自行实现数据读取、BIO 标签处理、Tokenizer 与标签对齐、动态 Padding、模型搭建、训练流程和评价指标，预训练模型与 Tokenizer 使用 Hugging Face 接口加载，并使用 SwanLab 记录实验过程。

---

## 项目结构

```text
demo2/
│
├── src/
│   ├── config.py              # 参数和数据路径配置
│   ├── labels.py              # BIO 标签管理
│   ├── dataset.py             # 数据处理、Tokenizer、标签对齐、动态 Padding
│   ├── model.py               # BERT + Linear 模型
│   ├── evaluate.py            # Precision、Recall、F1
│   └── train.py               # 训练流程和 SwanLab 记录
│
├── experiments/
│   ├── run_weibo_bert.py      # Weibo + bert-base-chinese
│   ├── run_weibo_wwm.py       # Weibo + chinese-bert-wwm
│   ├── run_msra_bert.py       # MSRA + bert-base-chinese
│   ├── run_msra_wwm.py        # MSRA + chinese-bert-wwm
│   ├── run_lr_tuning.py       # 学习率调优
│   └── run_epoch_tuning.py    # Epoch 调优
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 各模块说明

* `src/config.py`：保存数据集路径、预训练模型名称、Batch Size、Learning Rate、Epoch、Max Length 等参数。
* `src/labels.py`：使用 `LabelManager` 管理 BIO 标签，建立 `label2id` 和 `id2label`。
* `src/dataset.py`：读取 BIO 数据，使用 Hugging Face Tokenizer，完成 Token 与 BIO 标签对齐，并实现动态 Padding。
* `src/model.py`：加载预训练 BERT，并添加 Dropout 和 Linear 分类层，完成 Token 级 BIO 标签预测。
* `src/evaluate.py`：实现 Precision、Recall 和 F1。
* `src/train.py`：实现训练、验证、最佳模型保存和 SwanLab 实验记录。
* `experiments/`：保存不同模型、数据集和参数调优实验的运行入口。

---

## 数据格式

每行保存一个字符和对应的实体标签，空行表示一句话结束。

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

我 O
在 O
上 B-LOC
海 I-LOC
```

标签含义：

* `O`：不属于实体
* `B-*`：实体开始位置
* `I-*`：实体内部位置

### Weibo

Weibo 数据集包含 `PER`、`LOC`、`ORG`、`GPE` 等实体类型，并进一步区分 `.NAM` 和 `.NOM`。

共包含 **17 个 BIO 标签**。

### MSRA

MSRA 数据集包含 `LOC`、`PER`、`ORG` 三类实体。

共包含 **7 个 BIO 标签**。

---

## 数据规模

| 数据集   |  训练集 |  验证集 |  测试集 | 标签类别 |
| ----- | ---: | ---: | ---: | ---: |
| MSRA  | 5000 | 1000 | 1000 |    7 |
| Weibo | 1350 |  269 |  270 |   17 |

---

## 模型结构

项目采用 BERT + Linear 的序列标注结构：

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

特殊 Token 和 Padding 使用 `-100`：

```text
[CLS] → -100
[SEP] → -100
PAD   → -100
```

`-100` 的位置在 `CrossEntropyLoss` 中被忽略。

---

## 模型与训练配置

默认配置：

```text
Model: bert-base-chinese
Batch Size: 16
Learning Rate: 2e-5
Epochs: 3
Max Length: 128
Dropout: 0.1
Optimizer: AdamW
Loss: CrossEntropyLoss
```

---

## 运行方式

进入项目根目录：

```bash
cd C:\Users\13224\Desktop\demo2
```

### Weibo + bert-base-chinese

```bash
python -m experiments.run_weibo_bert
```

### Weibo + chinese-bert-wwm

```bash
python -m experiments.run_weibo_wwm
```

### MSRA + bert-base-chinese

```bash
python -m experiments.run_msra_bert
```

### MSRA + chinese-bert-wwm

```bash
python -m experiments.run_msra_wwm
```

---

## SwanLab

项目使用 SwanLab 记录训练过程。

主要记录：

```text
train/loss
dev/loss
dev/precision
dev/recall
dev/f1
```

四组基础实验分别为：

```text
weibo_bert
weibo_wwm
msra_bert
msra_wwm
```

---

# 运行结果

## Weibo + bert-base-chinese

最佳 Dev F1：

**0.9704**

## Weibo + chinese-bert-wwm

最佳 Dev F1：

**0.9709**

## MSRA + bert-base-chinese

最佳 Dev F1：

**0.9886**

## MSRA + chinese-bert-wwm

最佳 Dev F1：

**0.9898**

### 四组实验对比

| 数据集   | 模型                | Best Dev F1 |
| ----- | ----------------- | ----------: |
| Weibo | bert-base-chinese |      0.9704 |
| Weibo | chinese-bert-wwm  |      0.9709 |
| MSRA  | bert-base-chinese |      0.9886 |
| MSRA  | chinese-bert-wwm  |      0.9898 |

实验结果表明：

* MSRA 数据集整体取得了更高的 Dev F1。
* 在两个数据集上，`chinese-bert-wwm` 均略高于 `bert-base-chinese`。
* 两种预训练模型之间的差距较小。

---

# 参数调优

为了分析参数调优对实体识别性能的影响，选择：

**Weibo + bert-base-chinese**

作为参数调优对象。

## Learning Rate 调优

固定：

```text
Batch Size = 16
Epochs = 3
Max Length = 128
```

实验结果：

| Learning Rate | Best Dev F1 |
| ------------: | ----------: |
|          1e-5 |      0.9703 |
|          2e-5 |      0.9700 |
|          3e-5 |  **0.9734** |
|          5e-5 |      0.9712 |

在当前实验条件下，`3e-5` 的 Dev F1 最高。

---

## Epoch 调优

固定：

```text
Dataset = Weibo
Model = bert-base-chinese
Learning Rate = 3e-5
Batch Size = 16
Max Length = 128
```

实验结果：

| Epoch | Best Dev F1 |
| ----: | ----------: |
|     2 |      0.9689 |
|     4 |  **0.9728** |
|     6 |      0.9723 |
|     8 |      0.9711 |

结果表明：

* 训练轮数增加到一定程度后，Dev F1 没有继续提升。
* `Epoch=4` 的实验在第 3 个 Epoch 达到最佳 Dev F1 `0.9728`。
* 继续训练后，验证集性能出现一定波动。

---

## 参数调优结论

当前实验中较优参数：

```text
Learning Rate = 3e-5
Batch Size = 16
Epoch ≈ 3
Max Length = 128
```

实验表明，学习率并不是越大越好；训练轮数增加后，训练集 Loss 继续下降，但验证集 F1 不一定继续提升。

---

## 评价指标

项目自行实现：

```text
Precision
Recall
F1
```

评价时忽略 `-100` 对应的特殊 Token 和 Padding。

当前采用课程实验中的简化 Token-level F1 计算方式。

---

## 测试代码

### Dataset 测试

```bash
python test_dataset.py
```

用于测试原始数据读取。

### DataLoader 测试

```bash
python test_dataloader.py
```

用于测试 DataLoader 和动态 Padding。

### Tokenizer 对齐测试

```bash
python test_alignment.py
```

用于测试 Tokenizer 与 BIO 标签对齐。

### 模型测试

```bash
python test_model.py
```

用于测试模型前向传播和 Loss 计算。

---

## 环境

```text
Python 3.x
PyTorch
Transformers
SwanLab
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 注意事项

预训练模型使用 Hugging Face 接口加载，第一次运行时需要下载模型。

训练产生的：

```text
outputs/
swanlog/
```

建议不要提交到 GitHub。

数据集是否可以公开上传，需要根据课程要求和数据集许可确定。

---

## 项目总结

本项目完成了：

* `bert-base-chinese` 与 `chinese-bert-wwm` 两种预训练模型实验
* Weibo 与 MSRA 两个中文 NER 数据集实验
* BIO 标签处理
* Tokenizer 与 BIO 标签对齐
* 动态 Padding
* BERT + Linear 序列标注模型
* 自定义训练流程
* Precision、Recall、F1 评价
* SwanLab 实验记录
* Learning Rate 参数调优
* Epoch 参数调优

项目采用模块化设计，将数据处理、标签管理、模型搭建、训练和评价分别封装，便于后续进行模型、数据集和参数实验。
