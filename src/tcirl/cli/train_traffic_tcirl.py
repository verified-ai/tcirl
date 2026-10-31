from __future__ import annotations

from tcirl.cli.training import traffic_signal_case_study, train_tcirl
from tcirl.config import TrafficSignalTcirlConfig, parse_config
from tcirl.path_consts import CONFIG_DIR


def main() -> None:
    config = parse_config(
        TrafficSignalTcirlConfig,
        CONFIG_DIR / "traffic_signal_tcirl.yaml",
        "TCIRL on traffic signal control",
    )
    train_tcirl(config, lambda: traffic_signal_case_study(config.env))


if __name__ == "__main__":
    main()
