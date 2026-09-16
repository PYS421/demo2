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
        samples = []

        tokens = []
        labels = []

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:
                line = line.strip()

                # 空行：一句话结束
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

        previous_word_id = None

        for word_id in word_ids:

            # [CLS] / [SEP]
            if word_id is None:
                aligned_labels.append(-100)

            # 一个新的原始 Token
            elif word_id != previous_word_id:
                label = labels[word_id]

                label_id = self.label_manager.encode(
                    label
                )

                aligned_labels.append(label_id)

            # 同一个原始 Token 的后续 subword
            else:
                aligned_labels.append(-100)

            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": aligned_labels
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
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"]
        }

    def collate_fn(self, batch):
        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"]
            }
            for item in batch
        ]
        padded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        max_length = padded["input_ids"].size(1)

        padded_labels = []

        for item in batch:

            labels = item["labels"]

            padding_length = max_length - len(labels)

            labels = labels + [-100] * padding_length

            padded_labels.append(labels)

        padded["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long
        )

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": padded["labels"]
        }
