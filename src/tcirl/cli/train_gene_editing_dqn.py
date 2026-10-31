from __future__ import annotations

from tcirl.cli.training import gene_editing_case_study, train_dqn
from tcirl.config import GeneEditingDqnConfig, parse_config
from tcirl.path_consts import CONFIG_DIR


def main() -> None:
    config = parse_config(
        GeneEditingDqnConfig,
        CONFIG_DIR / "gene_editing_dqn.yaml",
        "DQN baseline on the gene-editing gridworld",
    )
    train_dqn(config, lambda: gene_editing_case_study(config.env))


if __name__ == "__main__":
    main()
