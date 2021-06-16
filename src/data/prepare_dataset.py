import click
from pathlib import Path
import sys
import music21 as m21
import pandas as pd
import pickle

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.constants import accepted_pitches, accepted_intervals


def transp_score(score):
    """ For each input generates len(accepted_intervals) transposed scores.

    Args:
        score (music21.Score): the score to transpose

    Returns:
        list[music21.Score]: a list of transposed scores
    """
    return [score.transpose(interval) for interval in accepted_intervals]


def score2pc(score):
    """ Generates a sequence of pitch-classes (midi numbers modulo 12) from a score.

    Args:
        score (music21.Score): the input score

    Returns:
        list[int]: a list of pitch-classes corresponding to each note in the score
    """
    return [p.midi % 12 for n in score.flat.notes for p in n.pitches]


def score2tpc(score):
    """ Generates a sequence of tonal-pitch-classes (diatonic pitch name + accidental) from a score.

    Sharps are represented with "#" and flats with "-". 
    Double and triple sharps and flats are just repetitions of those symbols.

    Args:
        score (music21.Score): the input score

    Returns:
        list[int]: a list of tonal-pitch-classes corresponding to each note in the score
    """
    return [p.name for n in score.flat.notes for p in n.pitches]


def score2tpc_meredith(score):
    """ Generates a sequence of tonal-pitch-classes (diatonic pitch name + accidental) from a score,
    in the format used by David Meredith (http://www.titanmusic.com/data.php)

    Sharps are represented with "s" and flats with "f". 
    Double and triple sharps and flats are just repetitions of those symbols.

    Args:
        score (music21.Score): the input score

    Returns:
        list[int]: a list of tonal-pitch-classes corresponding to each note in the score
    """
    return [
        p.nameWithOctave.replace("s", "#").replace("f", "-")
        if ("#" in p.nameWithOctave) or ("-" in p.nameWithOctave)
        else p.nameWithOctave.replace("s", "#").replace("f", "-")[:-1]
        + "n"
        + p.nameWithOctave.replace("s", "#").replace("f", "-")[-1]
        for n in score.flat.notes
        for p in n.pitches
    ]


def score2onsets(score):
    """ Generates a sequence of note onsets (following music21 convention) from a score.

    Args:
        score (music21.Score): the input score

    Returns:
        list[float]: a list of onsets corresponding to each note in the score
    """
    return [n.offset for n in score.flat.notes for p in n.pitches]


def score2durations(score):
    """ Generates a sequence of note durations (in quarterLengths) from a score.

    Args:
        score (music21.Score): the input score

    Returns:
        list[float]: a list of durations corresponding to each note in the score
    """
    return [n.duration.quarterLength for n in score.flat.notes for p in n.pitches]


def score2voice(score):
    """ Generates a sequence of voice numbers from a score.

    Args:
        score (music21.Score): the input score

    Returns:
        list[int]: a list of voice numbers corresponding to each note in the score
    """
    return [
        int(str(n.getContextByClass("Voice"))[-2])
        if not n.getContextByClass("Voice") is None
        else 1
        for n in score.flat.notes
        for p in n.pitches
    ]


def score2ks(score):
    """ Generates a sequence of key signatures, one for each note in the score.

    A key signature is represented by the number of sharps (positive number),
    or number of flats (negative number).
    For example 3 correspond to A major, and -2 to Bb major.

    Args:
        score (music21.Score): the input score

    Returns:
        list[int]: a list of key signatures corresponding to each note in the score
    """
    temp_ks = None
    out = []
    for event in score.flat:
        if isinstance(event, m21.key.KeySignature):
            temp_ks = event.sharps
        elif isinstance(event, m21.note.NotRest):
            for pitch in event.pitches:
                out.append(temp_ks)
    return out


@click.command()
@click.option("--raw-folder", type=click.Path(exists=True), default=Path("./data/raw"))
@click.option(
    "--processed-folder", type=click.Path(exists=True), default=Path("./data/processed")
)
def main(raw_folder, processed_folder):
    print("Preprocessing the asap dataset")
    asap_basepath = Path(raw_folder, "asap-dataset")

    if not asap_basepath.exists():
        raise Exception(
            "There must be a folder named 'asap-dataset', inside" + str(raw_folder)
        )

    # load the dataset info
    if not Path(asap_basepath, "metadata.csv").exists():
        raise Exception("The asap-dataset folder is not correctly structured")
    df = pd.read_csv(Path(asap_basepath, "metadata.csv"))
    df = df.drop_duplicates(subset=["title", "composer"])

    xml_score_paths = list(df["xml_score"])

    asap_dataset_dict = []

    for i, path in enumerate(xml_score_paths):
        print("About to process", path)
        score = m21.converter.parse(Path(asap_basepath, path))
        # generate the transpositions for the piece. This takes a lot of time unfortunately.
        all_scores = transp_score(score)
        # delete the pieces with non accepted pitches (e.g. triple sharps)
        intervals = []
        scores = []
        for s, interval in zip(all_scores, accepted_intervals):
            if all(pitch in accepted_pitches for pitch in score2tpc(s)):
                scores.append(s)
                intervals.append(interval)
        # append all features to the dictionary.
        # This is not optimized, but it's still very fast compare to the transposition phase
        asap_dataset_dict.extend(
            [
                {
                    "onset": score2onsets(s),
                    "duration": score2durations(s),
                    "pitches": score2tpc(s),
                    "transposed_of": interval,
                    "midi_number": score2pc(s),
                    "key_signatures": score2ks(s),
                    "original_path": str(path),
                    "composer": str(path).split("/")[0],
                }
                for s, interval in zip(scores, intervals)
            ]
        )
        if i % 10 == 9:
            print(str(i), "piece of", str(len(xml_score_paths)), "preprocessed")

    # save dataset
    with open(Path(processed_folder, "asap.pkl"), "wb") as fid:
        pickle.dump(asap_dataset_dict, fid)


if __name__ == "__main__":
    main()
