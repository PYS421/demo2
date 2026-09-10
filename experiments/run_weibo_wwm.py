from copy import deepcopy
from src.config import Config
from src.train import train_model


if __name__ == "__main__":

    # 复制一份当前配置
    config = deepcopy(Config)

    # 当前实验：Weibo + chinese-bert-wwm
    config.dataset_name = "Weibo"

    # Weibo 数据
    config.train_file = Config.WEIBO_TRAIN
    config.dev_file = Config.WEIBO_DEV
    config.test_file = Config.WEIBO_TEST

    # 使用 chinese-bert-wwm
    config.model_name = "hfl/chinese-bert-wwm"

    # SwanLab 实验名称
    config.experiment_name = "weibo_wwm"

    # 模型保存位置
    config.save_path = "outputs/weibo_wwm_best.pt"

    # 开始训练
    train_model(config)