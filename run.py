import argparse

from src.config import Config
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="运行 BERT 中文 NER 实验"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON 配置文件路径"
    )

    args = parser.parse_args()

    # 从 JSON 读取实验配置
    config = Config.from_json(args.config)

    print("=" * 60)
    print("当前实验配置")
    print("=" * 60)
    print(config)

    # 开始训练、验证和测试
    train_model(config)


if __name__ == "__main__":
    main()