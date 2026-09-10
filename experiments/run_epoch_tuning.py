from copy import deepcopy
from src.config import Config
from src.train import train_model


if __name__ == "__main__":

    # Epoch 调优实验

    epoch_list = [
        2,
        4,
        6,
        8
    ]

    for epochs in epoch_list:

        print()
        print("=" * 70)
        print(f"开始 Epoch 实验：epochs = {epochs}")
        print("=" * 70)

        # 复制基础配置
        config = deepcopy(Config)

        # 数据集：Weibo

        config.dataset_name = "Weibo"

        config.train_file = Config.WEIBO_TRAIN
        config.dev_file = Config.WEIBO_DEV
        config.test_file = Config.WEIBO_TEST


        # 模型：bert-base-chinese

        config.model_name = "bert-base-chinese"

        # 固定其他参数

        config.learning_rate = 3e-5
        config.batch_size = 16
        config.max_length = 128

        # 当前调整参数：Epoch

        config.epochs = epochs

        # SwanLab 实验名称

        config.experiment_name = (
            f"weibo_bert_epoch_{epochs}"
        )

        # 当前实验模型保存路径
        config.save_path = (
            f"outputs/"
            f"weibo_bert_epoch_{epochs}_best.pt"
        )

        # -----------------------------------------------------
        # 开始训练
        # -----------------------------------------------------

        train_model(config)

        print()
        print("=" * 70)
        print(f"Epoch = {epochs} 实验完成")
        print("=" * 70)