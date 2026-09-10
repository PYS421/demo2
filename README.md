# 中文命名实体识别

基于 **PyTorch + Hugging Face Transformers** 实现中文命名实体识别（NER）。

使用以下模型和数据集进行实验：

* `bert-base-chinese`
* `chinese-bert-wwm`
* Weibo
* MSRA

项目自行实现数据处理、BIO 标签对齐、模型搭建、训练流程和评价指标，预训练模型与 Tokenizer 使用 Hugging Face 接口。

## 项目结构

```text
demo2/
├── data/
│   ├── weibo/
│   └── MSRA/
│
├── src/
│   ├── config.py
│   ├── labels.py
│   ├── dataset.py
│   ├── model.py
│   ├── evaluate.py
│   └── train.py
│
├── experiments/
│   ├── run_weibo_bert.py
│   ├── run_weibo_wwm.py
│   ├── run_msra_bert.py
│   ├── run_msra_wwm.py
│   ├── run_lr_tuning.py
│   └── run_epoch_tuning.py
│
├── requirements.txt
└── README.md
```

## 模型结构

```text
文本
 ↓
Tokenizer
 ↓
BERT / Chinese-BERT-WWM
 ↓
Dropout
 ↓
Linear
 ↓
BIO 标签预测
```

## 基础实验

| 数据集   | 模型                | Best Dev F1 |
| ----- | ----------------- | ----------: |
| Weibo | bert-base-chinese |      0.9704 |
| Weibo | chinese-bert-wwm  |      0.9709 |
| MSRA  | bert-base-chinese |      0.9886 |
| MSRA  | chinese-bert-wwm  |      0.9898 |

## 参数调优

以 `Weibo + bert-base-chinese` 为例进行参数调优。

### Learning Rate

| Learning Rate | Best Dev F1 |
| ------------: | ----------: |
|          1e-5 |      0.9703 |
|          2e-5 |      0.9700 |
|          3e-5 |  **0.9734** |
|          5e-5 |      0.9712 |

### Epoch

| Epoch | Best Dev F1 |
| ----: | ----------: |
|     2 |      0.9689 |
|     4 |  **0.9728** |
|     6 |      0.9723 |
|     8 |      0.9711 |

## 运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行实验：

```bash
python -m experiments.run_weibo_bert

python -m experiments.run_weibo_wwm

python -m experiments.run_msra_bert

python -m experiments.run_msra_wwm
```

参数调优：

```bash
python -m experiments.run_lr_tuning

python -m experiments.run_epoch_tuning
```

## 实验记录

使用 **SwanLab** 记录训练过程和实验指标，包括：

* Train Loss
* Dev Loss
* Precision
* Recall
* F1
