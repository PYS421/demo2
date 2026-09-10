import os
import torch
import swanlab
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dataset import NERDataset, ner_collate_fn
from src.labels import LabelManager
from src.model import BertForNER
from src.evaluate import evaluate


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    for batch in data_loader:

        # 1. 数据放到设备

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # 2. 清空梯度
        optimizer.zero_grad()

        # 3. 前向传播

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs["loss"]
        # 4. 反向传播

        loss.backward()

        # 5. 更新参数

        optimizer.step()

        total_loss += loss.item()

    # 平均 Loss
    avg_loss = total_loss / len(data_loader)

    return avg_loss


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


def train_model(config):
    # 1. 设备

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("使用设备：", device)

    # 2. LabelManager

    label_manager = LabelManager.from_file(
        config.train_file
    )

    print("标签数量：", len(label_manager))

    print("label2id：")
    print(label_manager.label2id)

    # 3. Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    # 4. DataLoader

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

    print("训练集数量：", len(train_loader.dataset))
    print("验证集数量：", len(dev_loader.dataset))

    # 5. 初始化 SwanLab

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
            "num_labels": len(label_manager)
        }
    )
    # 6. 创建模型

    model = BertForNER(
        model_name=config.model_name,
        num_labels=len(label_manager)
    )

    model.to(device)

    # 7. 创建优化器

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    # 8. 开始训练

    best_dev_f1 = -1.0

    for epoch in range(config.epochs):

        print()
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{config.epochs}")
        print("=" * 60)

        # Train

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device
        )

        # Dev

        dev_metrics = evaluate(
            model=model,
            data_loader=dev_loader,
            device=device
        )

        # 打印结果

        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Dev Loss:       {dev_metrics['loss']:.4f}")
        print(f"Dev Precision:  {dev_metrics['precision']:.4f}")
        print(f"Dev Recall:     {dev_metrics['recall']:.4f}")
        print(f"Dev F1:         {dev_metrics['f1']:.4f}")
        # SwanLab 记录

        swanlab.log({
            "train/loss": train_loss,
            "dev/loss": dev_metrics["loss"],
            "dev/precision": dev_metrics["precision"],
            "dev/recall": dev_metrics["recall"],
            "dev/f1": dev_metrics["f1"]
        })
        # 保存最佳模型

        if dev_metrics["f1"] > best_dev_f1:

            best_dev_f1 = dev_metrics["f1"]

            # 创建保存目录
            save_dir = os.path.dirname(
                config.save_path
            )

            if save_dir:
                os.makedirs(
                    save_dir,
                    exist_ok=True
                )

            # 保存模型
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
    # 9. SwanLab 记录最佳结果
    swanlab.log({
        "best/dev_f1": best_dev_f1
    })

    # 10. 结束
    print()
    print("=" * 60)
    print("训练完成")
    print("=" * 60)

    print(
        f"最佳 Dev F1: {best_dev_f1:.4f}"
    )

    print(
        f"最佳模型：{config.save_path}"
    )

    # 结束 SwanLab 实验
    swanlab.finish()


if __name__ == "__main__":
    from src.config import Config

    train_model(Config)