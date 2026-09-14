import os
import torch
import swanlab
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dataset import NERDataset, ner_collate_fn
from src.labels import LabelManager
from src.model import BertForNER
from src.utils import set_seed, calculate_metrics


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def create_dataloader(
    file_path,
    tokenizer,
    label_manager,
    batch_size,
    max_length,
    shuffle
):
    dataset = NERDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        label_manager=label_manager,
        max_length=max_length
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: ner_collate_fn(
            batch,
            tokenizer.pad_token_id
        )
    )

    return data_loader


@torch.no_grad()
def evaluate_loader(
    model,
    data_loader,
    device,
    label_manager
):
    """
    在 Dev/Test 上执行推理。

    评价指标由 utils.py 的 calculate_metrics() 完成。
    """
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

    metrics = calculate_metrics(
        predictions=all_predictions,
        labels=all_labels,
        id2label=label_manager.id2label
    )

    metrics["loss"] = total_loss / len(data_loader)

    return metrics


def train_model(config):
    """
    完整实验流程：

        train -> 训练
        dev   -> 选择最佳模型
        test  -> 使用最佳模型进行最终测试

    测试集不会参与训练和模型选择。
    """

    # =========================================================
    # 1. 固定随机种子
    # =========================================================
    set_seed(config.seed)

    print("随机种子：", config.seed)

    # =========================================================
    # 2. 设备
    # =========================================================
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("使用设备：", device)

    # =========================================================
    # 3. LabelManager
    # =========================================================
    label_manager = LabelManager.from_file(
        config.train_file
    )

    print("标签数量：", len(label_manager))
    print("label2id：")
    print(label_manager.label2id)

    # =========================================================
    # 4. Tokenizer
    # =========================================================
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    # =========================================================
    # 5. DataLoader
    # =========================================================
    train_loader = create_dataloader(
        file_path=config.train_file,
        tokenizer=tokenizer,
        label_manager=label_manager,
        batch_size=config.batch_size,
        max_length=config.max_length,
        shuffle=True
    )

    dev_loader = create_dataloader(
        file_path=config.dev_file,
        tokenizer=tokenizer,
        label_manager=label_manager,
        batch_size=config.batch_size,
        max_length=config.max_length,
        shuffle=False
    )

    test_loader = create_dataloader(
        file_path=config.test_file,
        tokenizer=tokenizer,
        label_manager=label_manager,
        batch_size=config.batch_size,
        max_length=config.max_length,
        shuffle=False
    )

    print("训练集数量：", len(train_loader.dataset))
    print("验证集数量：", len(dev_loader.dataset))
    print("测试集数量：", len(test_loader.dataset))

    # =========================================================
    # 6. SwanLab
    # =========================================================
    swanlab.init(
        project="bert-ner-project",
        experiment_name=config.experiment_name,
        config={
            "dataset": config.dataset_name,
            "model": config.model_name,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "max_length": config.max_length,
            "dropout": getattr(config, "dropout", 0.1),
            "num_labels": len(label_manager),
            "seed": config.seed
        }
    )

    # =========================================================
    # 7. 创建模型
    # =========================================================
    model = BertForNER(
        model_name=config.model_name,
        num_labels=len(label_manager),
        dropout=getattr(config, "dropout", 0.1)
    )

    model.to(device)

    # =========================================================
    # 8. 优化器
    # =========================================================
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    # =========================================================
    # 9. 训练 + Dev
    # =========================================================
    best_dev_f1 = -1.0

    for epoch in range(config.epochs):

        print()
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print("=" * 60)

        # -------------------------
        # Train
        # -------------------------
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device
        )

        # -------------------------
        # Dev
        # -------------------------
        dev_metrics = evaluate_loader(
            model=model,
            data_loader=dev_loader,
            device=device,
            label_manager=label_manager
        )

        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Dev Loss:       {dev_metrics['loss']:.4f}")
        print(f"Dev Precision:  {dev_metrics['precision']:.4f}")
        print(f"Dev Recall:     {dev_metrics['recall']:.4f}")
        print(f"Dev F1:         {dev_metrics['f1']:.4f}")

        swanlab.log({
            "train/loss": train_loss,
            "dev/loss": dev_metrics["loss"],
            "dev/precision": dev_metrics["precision"],
            "dev/recall": dev_metrics["recall"],
            "dev/f1": dev_metrics["f1"]
        })

        # -------------------------
        # 根据 Dev F1 保存最佳模型
        # -------------------------
        if dev_metrics["f1"] > best_dev_f1:

            best_dev_f1 = dev_metrics["f1"]

            save_dir = os.path.dirname(config.save_path)

            if save_dir:
                os.makedirs(
                    save_dir,
                    exist_ok=True
                )

            torch.save(
                model.state_dict(),
                config.save_path
            )

            print("保存最佳模型")
            print(
                f"Best Dev F1 = {best_dev_f1:.4f}"
            )
            print(
                f"模型保存到：{config.save_path}"
            )

    # =========================================================
    # 10. 记录最佳 Dev
    # =========================================================
    swanlab.log({
        "best/dev_f1": best_dev_f1
    })

    # =========================================================
    # 11. 加载最佳模型
    # =========================================================
    print()
    print("=" * 60)
    print("训练完成，加载最佳模型进行测试")
    print("=" * 60)

    model.load_state_dict(
        torch.load(
            config.save_path,
            map_location=device
        )
    )

    # =========================================================
    # 12. Test
    # =========================================================
    test_metrics = evaluate_loader(
        model=model,
        data_loader=test_loader,
        device=device,
        label_manager=label_manager
    )

    print()
    print("=" * 60)
    print("Test Result")
    print("=" * 60)

    print(f"Test Loss:       {test_metrics['loss']:.4f}")
    print(f"Test Precision:  {test_metrics['precision']:.4f}")
    print(f"Test Recall:     {test_metrics['recall']:.4f}")
    print(f"Test F1:         {test_metrics['f1']:.4f}")

    # =========================================================
    # 13. SwanLab 记录 Test
    # =========================================================
    swanlab.log({
        "test/loss": test_metrics["loss"],
        "test/precision": test_metrics["precision"],
        "test/recall": test_metrics["recall"],
        "test/f1": test_metrics["f1"]
    })

    # =========================================================
    # 14. 结束
    # =========================================================
    print()
    print("=" * 60)
    print("全部实验完成")
    print("=" * 60)

    print(
        f"最佳 Dev F1: {best_dev_f1:.4f}"
    )

    print(
        f"最终 Test F1: {test_metrics['f1']:.4f}"
    )

    print(
        f"最佳模型：{config.save_path}"
    )

    swanlab.finish()

    return {
    "best_dev_f1": best_dev_f1,
    "test_metrics": test_metrics
}


if __name__ == "__main__":
    from src.config import Config

    train_model(Config)

