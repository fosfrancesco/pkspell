# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import subprocess
import sys
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from rich import print
from rich.progress import Progress

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.constants import ASAP_URL


@click.command()
@click.option("--folder", type=click.Path(exists=False), default=Path("./data/raw/"))
def main(folder):
    # create the raw folder in data if it does not exist
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)

    # Download ASAP dataset from github.
    print("Downloading ASAP dataset for training")
    asap_path = Path(folder, "asap-dataset")
    with Progress() as progress:
        clone = progress.add_task(f"[yellow]Cloning ASAP from [dim]{ASAP_URL}[/dim]")
        subprocess.run(["git", "clone", "--quiet", ASAP_URL, str(asap_path)])
        progress.update(clone, advance=100)
    
    # download MuseData noisy dataset from David Meredith http://www.titanmusic.com/data.php
    print("Downloading noisy MuseData dataset [dim](http://www.titanmusic.com/data.php)[/dim] for evaluation")
    zipurl = "http://www.titanmusic.com/data/dphil/opnd-m-noisy.zip"
    with Progress() as progress:
        musedata = progress.add_task(f"[yellow]Downloading MuseData")
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(Path(folder))
        progress.update(musedata, advance=100)


if __name__ == "__main__":
    main()
