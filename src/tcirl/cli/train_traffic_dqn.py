from __future__ import annotations

from tcirl.cli.training import traffic_signal_case_study, train_dqn
from tcirl.config import TrafficSignalDqnConfig, parse_config
from tcirl.path_consts import CONFIG_DIR


def main() -> None:
    config = parse_config(
        TrafficSignalDqnConfig,
        CONFIG_DIR / "traffic_signal_dqn.yaml",
        "DQN baseline on traffic signal control",
    )
    train_dqn(config, lambda: traffic_signal_case_study(config.env))


if __name__ == "__main__":
    main()
