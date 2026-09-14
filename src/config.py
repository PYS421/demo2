import json
from dataclasses import dataclass


@dataclass
class Config:
    # 数据集
    dataset_name: str
    train_file: str
    dev_file: str
    test_file: str

    # 模型
    model_name: str

    # 实验
    experiment_name: str
    save_path: str

    # 训练参数
    batch_size: int
    learning_rate: float
    epochs: int
    max_length: int
    dropout: float

    # 随机种子
    seed: int

    @classmethod
    def from_json(cls, json_path: str):
        """
        从 JSON 配置文件创建 Config 对象。
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            dataset_name=data["dataset"],
            train_file=data["train_file"],
            dev_file=data["dev_file"],
            test_file=data["test_file"],
            model_name=data["model_name"],
            experiment_name=data["experiment_name"],
            save_path=data["save_path"],
            batch_size=int(data.get("batch_size", 16)),
            learning_rate=float(data.get("learning_rate", 2e-5)),
            epochs=int(data.get("epochs", 3)),
            max_length=int(data.get("max_length", 128)),
            dropout=float(data.get("dropout", 0.1)),
            seed=int(data.get("seed", 42))
        )

    def __repr__(self):
        return (
            f"Config(\n"
            f"  dataset_name={self.dataset_name!r},\n"
            f"  model_name={self.model_name!r},\n"
            f"  train_file={self.train_file!r},\n"
            f"  dev_file={self.dev_file!r},\n"
            f"  test_file={self.test_file!r},\n"
            f"  batch_size={self.batch_size},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  epochs={self.epochs},\n"
            f"  max_length={self.max_length},\n"
            f"  dropout={self.dropout},\n"
            f"  seed={self.seed},\n"
            f"  experiment_name={self.experiment_name!r},\n"
            f"  save_path={self.save_path!r}\n"
            f")"
        )
