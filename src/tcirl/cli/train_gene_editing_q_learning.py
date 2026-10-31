from __future__ import annotations

from tcirl.cli.training import gene_editing_case_study, train_q_learning
from tcirl.config import GeneEditingQLConfig, parse_config
from tcirl.path_consts import CONFIG_DIR


def main() -> None:
    config = parse_config(
        GeneEditingQLConfig,
        CONFIG_DIR / "gene_editing_q_learning.yaml",
        "Q-learning on genetic-therapy gridworld",
    )
    train_q_learning(config, lambda: gene_editing_case_study(config.env))


if __name__ == "__main__":
    main()
