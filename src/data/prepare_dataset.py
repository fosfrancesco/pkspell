import click
import logging
from pathlib import Path
import sys
import music21 as m21
import pandas as pd
import pickle

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.constants import accepted_pitches, accepted_intervals
from src.utils.utils import (
    transp_score,
    score2durations,
    score2midi_numbers,
    score2pitches,
    score2onsets,
)


@click.command()
@click.option("--raw-folder", type=click.Path(exists=True), default=Path("./data/raw"))
@click.option(
    "--processed-folder", type=click.Path(exists=True), default=Path("./data/processed")
)
def main(raw_folder, processed_folder):
    """ Download ASAP dataset from github.
    """
    logger = logging.getLogger(__name__)
    logger.info("Preprocessing the asap dataset")

    asap_basepath = Path(raw_folder, "asap-dataset")

    # load the dataset info
    df = pd.read_csv(Path(asap_basepath, "metadata.csv"))
    df = df.drop_duplicates(subset=["title", "composer"])

    xml_score_paths = list(df["xml_score"])

    asap_dataset_dict = []

    for i, path in enumerate(xml_score_paths):
        print("About to process", path)
        score = m21.converter.parse(Path(asap_basepath, path))
        # generate the transpositions for the piece
        all_scores = transp_score(score)
        # delete the pieces with non accepted pitches (e.g. triple sharps)
        intervals = []
        scores = []
        for s, interval in zip(all_scores, accepted_intervals):
            if all(pitch in accepted_pitches for pitch in score2pitches(s)):
                scores.append(s)
                intervals.append(interval)
        # append all information to the dictionary
        asap_dataset_dict.extend(
            [
                {
                    "onset": score2onsets(s),
                    "duration": score2durations(s),
                    "pitches": score2pitches(s),
                    "transposed_of": interval,
                    "midi_number": score2midi_numbers(s),
                    "key_signatures": s.parts[0]
                    .flat.getElementsByClass(m21.key.KeySignature)[0]
                    .sharps,
                    "original_path": str(path),
                    "composer": str(path).split("/")[0],
                }
                for s, interval in zip(scores, intervals)
            ]
        )
        if i % 10 == 9:
            logger.info(str(i), "piece of", str(len(xml_score_paths)), "preprocessed")

    # save dataset
    with open(Path(processed_folder, "asap.pkl"), "wb") as fid:
        pickle.dump(asap_dataset_dict, fid)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
