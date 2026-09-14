import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 尽可能保证 CUDA 相关操作可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_entities(labels, id2label):
    """
    从 BIO 标签序列中提取完整实体。

    返回：
        [(start, end, entity_type), ...]

    例如：
        O B-ORG I-ORG I-ORG O
    返回：
        [(1, 3, 'ORG')]

    start / end 是当前“有效 token 序列”中的位置。
    """
    entities = []

    start = None
    entity_type = None

    for i, label_id in enumerate(labels):
        label = id2label[int(label_id)]

        if label == "O":
            if start is not None:
                entities.append((start, i - 1, entity_type))
                start = None
                entity_type = None
            continue

        if label.startswith("B-"):
            if start is not None:
                entities.append((start, i - 1, entity_type))

            start = i
            entity_type = label[2:]
            continue

        if label.startswith("I-"):
            current_type = label[2:]

            if start is not None and entity_type == current_type:
                continue

            # 非法 BIO：I-XXX 前面没有同类型实体，按新实体处理
            if start is not None:
                entities.append((start, i - 1, entity_type))

            start = i
            entity_type = current_type

    if start is not None:
        entities.append((start, len(labels) - 1, entity_type))

    return entities


def calculate_metrics(predictions, labels, id2label):
    true_positive = 0
    predicted_total = 0
    gold_total = 0

    for pred_row, label_row in zip(predictions, labels):
        valid_preds = []
        valid_labels = []

        for pred, label in zip(pred_row, label_row):
            if int(label) == -100:
                continue

            valid_preds.append(int(pred))
            valid_labels.append(int(label))

        pred_entities = set(
            extract_entities(valid_preds, id2label)
        )

        gold_entities = set(
            extract_entities(valid_labels, id2label)
        )

        true_positive += len(pred_entities & gold_entities)
        predicted_total += len(pred_entities)
        gold_total += len(gold_entities)

    precision = (
        true_positive / predicted_total
        if predicted_total > 0
        else 0.0
    )

    recall = (
        true_positive / gold_total
        if gold_total > 0
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

