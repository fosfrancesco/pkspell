# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import subprocess
import sys
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.constants import ASAP_URL


@click.command()
@click.option("--folder", type=click.Path(exists=True), default=Path("./data/raw"))
def main(folder):
    # create the raw folder in data if it does not exist
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

    # Download ASAP dataset from github.
    print("Downloading asap dataset for training")
    asap_path = Path(folder, "asap-dataset")
    subprocess.run(["git", "clone", ASAP_URL, str(asap_path)])

    # download MuseData noisy dataset from David Meredith http://www.titanmusic.com/data.php
    print(
        "Downloading noisy musedata dataset (http://www.titanmusic.com/data.php) for evaluation"
    )
    zipurl = "http://www.titanmusic.com/data/dphil/opnd-m-noisy.zip"
    with urlopen(zipurl) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(Path(folder))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
