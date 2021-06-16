# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import subprocess
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.constants import ASAP_URL


@click.command()
@click.option("--folder", type=click.Path(exists=True), default=Path("./data/raw"))
def main(folder):
    """ Download ASAP dataset from github.
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading asap dataset")

    asap_path = Path(folder, "asap-dataset")
    subprocess.run(["git", "clone", ASAP_URL, str(asap_path)])


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()