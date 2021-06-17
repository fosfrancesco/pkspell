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


def parenthetic_contents(string):
    """Parse nested parentheses in a LISP-like list of lists."""
    stack = []
    for i, c in enumerate(string):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            if len(stack) == 2:  # only consider elements at depth 2
                start = stack.pop()
                yield string[start + 1 : i]


@click.command()
@click.option("--raw-folder", type=click.Path(exists=True), default=Path("./data/raw"))
@click.option(
    "--processed-folder", type=click.Path(exists=True), default=Path("./data/processed")
)
@click.option("--process-asap", default=True, type=bool)
@click.option("--process-musedata", default=True, type=bool)
def main(raw_folder, processed_folder, process_asap, process_musedata):
    # process Musedata (no augmentation)
    if process_musedata:
        print("Preprocessing the MuseData noisy dataset")
        musedata_basepath = Path(raw_folder, "opnd-m-noisy")

        musedata_list_of_dict = []
        if not musedata_basepath.exists():
            raise Exception(
                "There must be a folder named 'opnd-m-noisy', inside" + str(raw_folder)
            )
        for ifile, file in enumerate(musedata_basepath.iterdir()):
            with open(file, "r") as f:
                file_content = f.read()
            print("Processing file", ifile, str(file))
            strings_list = list(parenthetic_contents(file_content))
            quadruples_list = [s.split(" ") for s in strings_list]
            # sort by start input and pitch
            quadruples_list = sorted(quadruples_list, key=lambda tup: int(tup[0]))
            # put the information in a list of dicts
            pitches = [
                q[1].strip('"').replace("n", "").replace("s", "#").replace("f", "-")
                for q in quadruples_list
            ]
            # transform pitches in music21 notes
            m21_notes = [m21.note.Note(p) for p in pitches]
            pitches = [n.pitch.name for n in m21_notes]
            if all(
                p in accepted_pitches for p in pitches
            ):  # consider only accepted pitches (e.g. no triple accidentals)
                musedata_list_of_dict.append(
                    {
                        # 'onset': [int(q[0]) for q in quadruples_list],
                        "duration": [int(q[2]) for q in quadruples_list],
                        "pitches": pitches,
                        "midi_number": [n.pitch.midi % 12 for n in m21_notes],
                        "transposed_of": "P1",
                        "key_signatures": None,  # no ks in musedata dataset
                        "original_path": str(file),
                    }
                )
        with open(Path(processed_folder, "musedata.pkl"), "wb") as fid:
            pickle.dump(musedata_list_of_dict, fid)

    # process Asap (with augmentation)
    if process_asap:
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
        asap_list_of_dict = []
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
            asap_list_of_dict.extend(
                [
                    {
                        # "onset": score2onsets(s),
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
        with open(Path(processed_folder, "asap_augmented.pkl"), "wb") as fid:
            pickle.dump(asap_list_of_dict, fid)


if __name__ == "__main__":
    main()
