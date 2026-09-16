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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NERMetrics:
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.true_positive = 0
        self.predicted_total = 0
        self.gold_total = 0

    def _extract_entities(self, labels):
        entities = set()

        start = None
        entity_type = None

        for i, label_id in enumerate(labels):

            label = self.id2label[int(label_id)]

            if label == "O":

                if start is not None:
                    entities.add(
                        (
                            start,
                            i - 1,
                            entity_type
                        )
                    )

                    start = None
                    entity_type = None

                continue

            if label.startswith("B-"):

                if start is not None:
                    entities.add(
                        (
                            start,
                            i - 1,
                            entity_type
                        )
                    )

                start = i
                entity_type = label[2:]

                continue

            if label.startswith("I-"):

                current_type = label[2:]

                # 正常情况
                if (
                    start is not None
                    and entity_type == current_type
                ):
                    continue

                # 非法 BIO：
                # I-XXX 前面没有对应实体
                if start is not None:
                    entities.add(
                        (
                            start,
                            i - 1,
                            entity_type
                        )
                    )

                start = i
                entity_type = current_type


        if start is not None:
            entities.add(
                (
                    start,
                    len(labels) - 1,
                    entity_type
                )
            )

        return entities

    def update(self, predictions, labels):
        for pred_row, label_row in zip(
            predictions,
            labels
        ):

            valid_predictions = []
            valid_labels = []

            for pred, label in zip(
                pred_row,
                label_row
            ):

                # 忽略特殊 Token、Padding 和 subword
                if int(label) == -100:
                    continue

                valid_predictions.append(
                    int(pred)
                )

                valid_labels.append(
                    int(label)
                )

            # 预测实体
            pred_entities = self._extract_entities(
                valid_predictions
            )

            # 真实实体
            gold_entities = self._extract_entities(
                valid_labels
            )

            # 完整实体匹配
            self.true_positive += len(
                pred_entities & gold_entities
            )

            self.predicted_total += len(
                pred_entities
            )

            self.gold_total += len(
                gold_entities
            )

    def compute(self):
        if self.predicted_total == 0:
            precision = 0.0
        else:
            precision = (
                self.true_positive
                / self.predicted_total
            )

        if self.gold_total == 0:
            recall = 0.0
        else:
            recall = (
                self.true_positive
                / self.gold_total
            )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }