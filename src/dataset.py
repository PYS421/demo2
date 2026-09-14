import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):

    def __init__(
        self,
        file_path,
        tokenizer,
        label_manager,
        max_length=128
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.label_manager = label_manager
        self.max_length = max_length

        self.samples = self._load_data(file_path)

    def _load_data(self, file_path):
        """
        读取 BIO 格式数据。

        每一行：
            token label

        空行表示一句话结束。
        """
        samples = []

        tokens = []
        labels = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    if tokens:
                        samples.append({
                            "tokens": tokens,
                            "labels": labels
                        })

                        tokens = []
                        labels = []

                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                token = parts[0]
                label = parts[1]

                tokens.append(token)
                labels.append(label)

        # 防止最后一句没有空行
        if tokens:
            samples.append({
                "tokens": tokens,
                "labels": labels
            })

        return samples

    def _tokenize_and_align_labels(self, tokens, labels):
        """
        将原始 token 输入 tokenizer，并把 BIO 标签对齐到 tokenizer
        产生的 token 上。

        对于一个原始 token 被拆成多个 subword 的情况：
            第一个 subword 保留原标签；
            后续 subword 使用 -100，不参与 loss 和评价。

        返回 word_ids，供评价阶段恢复有效 token 序列。
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True
        )

        word_ids = encoding.word_ids()

        aligned_labels = []
        aligned_word_ids = []

        previous_word_id = None

        for word_id in word_ids:

            # [CLS] / [SEP] 等特殊 token
            if word_id is None:
                aligned_labels.append(-100)
                aligned_word_ids.append(None)

            # 当前是一个新的原始 token
            elif word_id != previous_word_id:
                label = labels[word_id]
                label_id = self.label_manager.encode(label)

                aligned_labels.append(label_id)
                aligned_word_ids.append(word_id)

            # 当前是同一个原始 token 的后续 subword
            else:
                aligned_labels.append(-100)
                aligned_word_ids.append(word_id)

            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": aligned_labels,
            "word_ids": aligned_word_ids
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        encoded = self._tokenize_and_align_labels(
            sample["tokens"],
            sample["labels"]
        )

        return {
            "input_ids": torch.tensor(
                encoded["input_ids"],
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                encoded["attention_mask"],
                dtype=torch.long
            ),

            "labels": torch.tensor(
                encoded["labels"],
                dtype=torch.long
            )
        }


def ner_collate_fn(batch, pad_token_id):

    # 当前 batch 中最长的序列长度
    max_length = max(
        len(item["input_ids"])
        for item in batch
    )

    input_ids = []
    attention_masks = []
    labels = []

    for item in batch:

        input_id = item["input_ids"]
        attention_mask = item["attention_mask"]
        label = item["labels"]

        padding_length = max_length - len(input_id)

        padded_input_ids = torch.cat([
            input_id,
            torch.full(
                (padding_length,),
                pad_token_id,
                dtype=torch.long
            )
        ])

        padded_attention_mask = torch.cat([
            attention_mask,
            torch.zeros(
                padding_length,
                dtype=torch.long
            )
        ])

        padded_labels = torch.cat([
            label,
            torch.full(
                (padding_length,),
                -100,
                dtype=torch.long
            )
        ])

        input_ids.append(padded_input_ids)
        attention_masks.append(padded_attention_mask)
        labels.append(padded_labels)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels)
    }
