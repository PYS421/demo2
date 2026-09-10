from copy import deepcopy
from src.config import Config
from src.train import train_model


if __name__ == "__main__":
    # 学习率调优实验

    learning_rates = [
        1e-5,
        2e-5,
        3e-5,
        5e-5
    ]

    for lr in learning_rates:

        print()
        print("=" * 70)
        print(f"开始学习率实验：learning_rate = {lr}")
        print("=" * 70)

        # 复制基础配置
        config = deepcopy(Config)

        # 固定实验条件
        # 数据集：Weibo
        config.dataset_name = "Weibo"

        config.train_file = Config.WEIBO_TRAIN
        config.dev_file = Config.WEIBO_DEV
        config.test_file = Config.WEIBO_TEST

        # 模型：bert-base-chinese
        config.model_name = "bert-base-chinese"

        # 固定 Batch Size
        config.batch_size = 16

        # 固定 Epoch
        config.epochs = 3

        # 固定最大长度
        config.max_length = 128

        # 当前要调整的参数：learning rate

        config.learning_rate = lr

        # SwanLab 实验名称
        lr_name = str(lr).replace("-", "m").replace(".", "_")

        config.experiment_name = (
            f"weibo_bert_lr_{lr_name}"
        )

        # 当前实验模型保存路径
        config.save_path = (
            f"outputs/"
            f"weibo_bert_lr_{lr_name}_best.pt"
        )

        # 开始训练
        train_model(config)

        print()
        print("=" * 70)
        print(f"学习率 {lr} 实验完成")
        print("=" * 70)