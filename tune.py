import argparse
import copy
import json
import os

from src.config import Config
from src.train import train_model


def load_tuning_config(json_path):
    """
    读取调参 JSON。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="运行 NER 参数调优实验"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="调参 JSON 文件路径"
    )

    args = parser.parse_args()

    # =========================================================
    # 1. 读取调参配置
    # =========================================================
    tuning_config = load_tuning_config(args.config)

    base_config_path = tuning_config["base_config"]
    parameter_name = tuning_config["parameter"]
    parameter_values = tuning_config["values"]

    print("=" * 70)
    print("开始参数调优")
    print("=" * 70)

    print(f"基础配置：{base_config_path}")
    print(f"调优参数：{parameter_name}")
    print(f"参数取值：{parameter_values}")
    print()

    # =========================================================
    # 2. 读取基础 JSON
    # =========================================================
    base_config = Config.from_json(base_config_path)

    results = []

    # =========================================================
    # 3. 循环运行不同参数
    # =========================================================
    for value in parameter_values:

        print()
        print("=" * 70)
        print(f"当前实验：{parameter_name} = {value}")
        print("=" * 70)

        # 创建当前实验配置副本
        current_config = copy.deepcopy(base_config)

        # 修改当前参数
        setattr(
            current_config,
            parameter_name,
            value
        )

        # 修改实验名称，避免 SwanLab 实验名称重复
        current_config.experiment_name = (
            f"{base_config.experiment_name}_"
            f"{parameter_name}_{value}"
        )

        # 修改模型保存路径
        save_dir = os.path.dirname(
            base_config.save_path
        )

        save_name = os.path.basename(
            base_config.save_path
        )

        name, ext = os.path.splitext(save_name)

        current_config.save_path = os.path.join(
            save_dir,
            f"{name}_{parameter_name}_{value}{ext}"
        )

        # -----------------------------------------------------
        # 开始训练
        # -----------------------------------------------------
        train_model(current_config)

        results.append({
            "parameter": parameter_name,
            "value": value,
            "save_path": current_config.save_path
        })

    # =========================================================
    # 4. 输出实验完成信息
    # =========================================================
    print()
    print("=" * 70)
    print("参数调优实验全部完成")
    print("=" * 70)

    for result in results:
        print(
            f"{result['parameter']} = "
            f"{result['value']}  ->  "
            f"{result['save_path']}"
        )

    print()


if __name__ == "__main__":
    main()