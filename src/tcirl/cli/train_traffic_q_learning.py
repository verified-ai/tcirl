from __future__ import annotations

from tcirl.cli.training import traffic_signal_case_study, train_q_learning
from tcirl.config import TrafficSignalQLConfig, parse_config
from tcirl.path_consts import CONFIG_DIR


def main() -> None:
    config = parse_config(
        TrafficSignalQLConfig,
        CONFIG_DIR / "traffic_signal_q_learning.yaml",
        "Q-learning on traffic signal control",
    )
    train_q_learning(config, lambda: traffic_signal_case_study(config.env))


if __name__ == "__main__":
    main()
