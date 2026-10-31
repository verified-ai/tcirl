from __future__ import annotations

from tcirl.cli.training import gene_editing_case_study, train_tcirl
from tcirl.config import GeneEditingTcirlConfig, parse_config
from tcirl.path_consts import CONFIG_DIR


def main() -> None:
    config = parse_config(
        GeneEditingTcirlConfig,
        CONFIG_DIR / "gene_editing_tcirl.yaml",
        "TCIRL on gene-editing gridworld",
    )
    train_tcirl(config, lambda: gene_editing_case_study(config.env))


if __name__ == "__main__":
    main()
