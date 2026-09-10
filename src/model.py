import torch
import torch.nn as nn
from transformers import AutoModel


class BertForNER(nn.Module):


    def __init__(
        self,
        model_name,
        num_labels,
        dropout=0.1
    ):
        super().__init__()

        # 1. 加载 Hugging Face 预训练 BERT

        self.bert = AutoModel.from_pretrained(
            model_name
        )

        # BERT 隐藏层维度
        hidden_size = self.bert.config.hidden_size

        # 2. Dropout

        self.dropout = nn.Dropout(
            dropout
        )

        # 3. NER 分类层

        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None
    ):

        # 1. BERT 前向传播

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # BERT 最后一层隐藏状态
        #
        # shape:
        # [batch_size, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state

        # 2. Dropout

        hidden_states = self.dropout(
            hidden_states
        )
        # 3. Linear 分类

        logits = self.classifier(
            hidden_states
        )

        # logits shape:
        #
        # [batch_size, seq_len, num_labels]

        loss = None

        # 4. 计算 Loss

        if labels is not None:

            loss_function = nn.CrossEntropyLoss(
                ignore_index=-100
            )


            loss = loss_function(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": logits
        }