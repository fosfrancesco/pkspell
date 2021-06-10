import click
from pathlib import Path
import logging
import subprocess

from utils.data.pytorch_dataset import ASAP_URL


@click.command()
@click.option("--folder", type=click.Path(exists=True), default=Path("./data/raw"))
def main(folder):
    """ Download ASAP dataset from github.
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    subprocess.run(["git", "clone", ASAP_URL, folder])


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
