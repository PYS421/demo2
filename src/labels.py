class LabelManager:

    def __init__(self, labels):

        # 去重，同时保持原来的顺序
        self.labels = list(dict.fromkeys(labels))

        # 标签 -> ID
        self.label2id = {
            label: idx
            for idx, label in enumerate(self.labels)
        }

        # ID -> 标签
        self.id2label = {
            idx: label
            for label, idx in self.label2id.items()
        }

    def encode(self, label):

        if label not in self.label2id:
            raise ValueError(
                f"未知标签: {label}"
            )

        return self.label2id[label]

    def decode(self, label_id):

        if label_id not in self.id2label:
            raise ValueError(
                f"未知标签ID: {label_id}"
            )

        return self.id2label[label_id]

    def __len__(self):

        return len(self.labels)

    @classmethod
    def from_file(cls, file_path):
        """
        从 NER 数据文件中自动读取所有标签。

        数据格式：

        本 O
        报 O
        老 B-PER.NOM
        百 I-PER.NOM

        空行表示一句话结束。
        """

        labels = []

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                # 跳过空行
                if not line:
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                label = parts[1]

                labels.append(label)

        # 保证 O 放在第一个位置
        unique_labels = list(dict.fromkeys(labels))

        if "O" in unique_labels:
            unique_labels.remove("O")
            unique_labels.insert(0, "O")

        return cls(unique_labels)