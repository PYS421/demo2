class Config:

    # 数据集目录
    WEIBO_DIR = r"C:\Users\13224\Desktop\demo2\data\weibo"
    MSRA_DIR = r"C:\Users\13224\Desktop\demo2\data\MSRA"

    # Weibo 数据
    WEIBO_TRAIN = rf"{WEIBO_DIR}\train.txt"
    WEIBO_DEV = rf"{WEIBO_DIR}\dev.txt"
    WEIBO_TEST = rf"{WEIBO_DIR}\test.txt"
    WEIBO_CLASS = rf"{WEIBO_DIR}\class.txt"

    # MSRA 数据

    MSRA_TRAIN = rf"{MSRA_DIR}\train_5k.txt"
    MSRA_DEV = rf"{MSRA_DIR}\dev_1k.txt"
    MSRA_TEST = rf"{MSRA_DIR}\test_1k.txt"

    # 当前实验
    dataset_name = "Weibo"

    train_file = WEIBO_TRAIN
    dev_file = WEIBO_DEV
    test_file = WEIBO_TEST

    model_name = "hfl/chinese-bert-wwm"
    experiment_name = "weibo_wwm"
    save_path = "outputs/weibo_wwm_best.pt"

    # 训练参数
    batch_size = 16
    learning_rate = 2e-5
    epochs = 3
    max_length = 128