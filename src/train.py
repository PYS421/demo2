import os
import torch
import swanlab
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.dataset import NERDataset
from src.labels import LabelManager
from src.model import BertForNER
from src.utils import set_seed, NERMetrics


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
    dataset,
    batch_size,
    shuffle
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn
    )


@torch.no_grad()
def evaluate_loader(
    model,
    data_loader,
    device,
    label_manager
):
    model.eval()

    total_loss = 0.0

    metrics = NERMetrics(
        label_manager.id2label
    )

    for batch in data_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        total_loss += outputs["loss"].item()

        predictions = torch.argmax(
            outputs["logits"],
            dim=-1
        )

        metrics.update(
            predictions=predictions.cpu().tolist(),
            labels=labels.cpu().tolist()
        )

    result = metrics.compute()

    result["loss"] = (
        total_loss / len(data_loader)
    )

    return result


def train_model(config):
    set_seed(config.seed)

    print(
        "随机种子：",
        config.seed
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(
        "使用设备：",
        device
    )

    label_manager = LabelManager.from_file(
        config.train_file
    )

    print(
        "标签数量：",
        len(label_manager)
    )

    print("label2id：")
    print(
        label_manager.label2id
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    train_dataset = NERDataset(
        file_path=config.train_file,
        tokenizer=tokenizer,
        label_manager=label_manager,
        max_length=config.max_length
    )

    dev_dataset = NERDataset(
        file_path=config.dev_file,
        tokenizer=tokenizer,
        label_manager=label_manager,
        max_length=config.max_length
    )

    test_dataset = NERDataset(
        file_path=config.test_file,
        tokenizer=tokenizer,
        label_manager=label_manager,
        max_length=config.max_length
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    dev_loader = create_dataloader(
        dataset=dev_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    print(
        "训练集数量：",
        len(train_dataset)
    )

    print(
        "验证集数量：",
        len(dev_dataset)
    )

    print(
        "测试集数量：",
        len(test_dataset)
    )

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
            "dropout": getattr(
                config,
                "dropout",
                0.1
            ),
            "num_labels": len(label_manager),
            "seed": config.seed
        }
    )

    model = BertForNER(
        model_name=config.model_name,
        num_labels=len(label_manager),
        dropout=getattr(
            config,
            "dropout",
            0.1
        )
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    best_dev_f1 = -1.0

    for epoch in range(
        config.epochs
    ):

        print()
        print("=" * 60)
        print(
            f"Epoch {epoch + 1}/{config.epochs}"
        )
        print("=" * 60)

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device
        )

        dev_metrics = evaluate_loader(
            model=model,
            data_loader=dev_loader,
            device=device,
            label_manager=label_manager
        )
        print(
            f"Train Loss:     {train_loss:.4f}"
        )

        print(
            f"Dev Loss:       {dev_metrics['loss']:.4f}"
        )

        print(
            f"Dev Precision:  "
            f"{dev_metrics['precision']:.4f}"
        )

        print(
            f"Dev Recall:     "
            f"{dev_metrics['recall']:.4f}"
        )

        print(
            f"Dev F1:         "
            f"{dev_metrics['f1']:.4f}"
        )

        swanlab.log({
            "train/loss": train_loss,
            "dev/loss": dev_metrics["loss"],
            "dev/precision": dev_metrics["precision"],
            "dev/recall": dev_metrics["recall"],
            "dev/f1": dev_metrics["f1"]
        })

        if (
            dev_metrics["f1"]
            > best_dev_f1
        ):

            best_dev_f1 = (
                dev_metrics["f1"]
            )

            save_dir = os.path.dirname(
                config.save_path
            )

            if save_dir:
                os.makedirs(
                    save_dir,
                    exist_ok=True
                )

            torch.save(
                model.state_dict(),
                config.save_path
            )

            print(
                "保存最佳模型"
            )

            print(
                f"Best Dev F1 = "
                f"{best_dev_f1:.4f}"
            )

            print(
                f"模型保存到："
                f"{config.save_path}"
            )

    swanlab.log({
        "best/dev_f1": best_dev_f1
    })
    print()
    print("=" * 60)
    print(
        "训练完成，加载最佳模型进行测试"
    )
    print("=" * 60)

    model.load_state_dict(
        torch.load(
            config.save_path,
            map_location=device
        )
    )

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

    print(
        f"Test Loss:       "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Test Precision:  "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall:     "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1:         "
        f"{test_metrics['f1']:.4f}"
    )

    swanlab.log({
        "test/loss": test_metrics["loss"],
        "test/precision": test_metrics["precision"],
        "test/recall": test_metrics["recall"],
        "test/f1": test_metrics["f1"]
    })

    print()
    print("=" * 60)
    print("全部实验完成")
    print("=" * 60)

    print(
        f"最佳 Dev F1: "
        f"{best_dev_f1:.4f}"
    )

    print(
        f"最终 Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"最佳模型："
        f"{config.save_path}"
    )

    swanlab.finish()

    return {
        "best_dev_f1": best_dev_f1,
        "test_metrics": test_metrics
    }


if __name__ == "__main__":
    from src.config import Config

    train_model(Config)