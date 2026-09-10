from copy import deepcopy
from src.config import Config
from src.train import train_model


if __name__ == "__main__":

    # 复制当前配置
    config = deepcopy(Config)

    # 当前实验：MSRA + chinese-bert-wwm
    config.dataset_name = "MSRA"

    # MSRA 数据
    config.train_file = Config.MSRA_TRAIN
    config.dev_file = Config.MSRA_DEV
    config.test_file = Config.MSRA_TEST

    # 使用 chinese-bert-wwm
    config.model_name = "hfl/chinese-bert-wwm"

    # SwanLab 实验名称
    config.experiment_name = "msra_wwm"

    # 模型保存路径
    config.save_path = "outputs/msra_wwm_best.pt"

    # 开始训练
    train_model(config)