import torch

def calculate_metrics(predictions, labels):
    """
    计算 Precision、Recall、F1。

    使用最简单的 token-level 计算方法。

    参数：
        predictions:
            模型预测的标签 ID
            shape: [batch_size, seq_len]

        labels:
            真实标签 ID
            shape: [batch_size, seq_len]

    注意：
        labels == -100 的位置会被忽略。
    """

    true_positive = 0
    false_positive = 0
    false_negative = 0

    # 遍历每个样本
    for pred_row, label_row in zip(predictions, labels):

        # 遍历每个 token
        for pred, label in zip(pred_row, label_row):

            # 忽略特殊 token 和 Padding
            if label == -100:
                continue

            # 预测正确
            if pred == label:
                true_positive += 1

            # 预测错误
            else:
                false_positive += 1
                false_negative += 1

    # Precision
    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (
            true_positive + false_positive
        )

    # Recall
    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (
            true_positive + false_negative
        )

    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (
            precision + recall
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device
):


    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for batch in data_loader:

        # 1. 数据放到设备

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 2. 前向传播

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        total_loss += loss.item()

        # 3. 获取预测结果

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        # 4. 保存结果

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

    # 5. 平均 Loss

    avg_loss = total_loss / len(data_loader)

    # 6. Precision / Recall / F1
    metrics = calculate_metrics(
        all_predictions,
        all_labels
    )

    return {
        "loss": avg_loss,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
    }