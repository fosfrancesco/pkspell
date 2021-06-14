from pathlib import Path
import music21 as m21
import numpy as np
from src.utils.constants import INTERVALS, KEY_SIGNATURES, accepted_intervals


def keep_best_transpositions(full_dict_dataset):
    paths = list(set([e["original_path"] for e in full_dict_dataset]))
    # choose only one enharmonic version for each chromatic interval for each piece
    dict_dataset = []
    for path in paths:
        for c in range(12):
            pieces_to_consider = [
                opus
                for opus in full_dict_dataset
                if (
                    opus["original_path"] == path
                    and opus["transposed_of"] in INTERVALS[c]
                )
            ]
            # if the original is in pieces_to_consider, go with the original
            originals = [
                opus for opus in pieces_to_consider if opus["transposed_of"] == "P1"
            ]
            if len(originals) == 1:
                dict_dataset.append(originals[0])
            else:  # we go with the accidental minization criteria
                n_accidentals = [
                    sum(
                        [
                            pitch.count("#") + pitch.count("-")
                            for pitch in opus["pitches"]
                        ]
                    )
                    for opus in pieces_to_consider
                ]
                if len(pieces_to_consider) > 0:
                    dict_dataset.append(pieces_to_consider[np.argmin(n_accidentals)])
                else:
                    print("No options for", path, ". Chromatic: ", c)

    # also remove unaccepted ks
    print("Before removing according to ks:", len(dict_dataset))
    dict_dataset = [
        e
        for e in dict_dataset
        if all([k in KEY_SIGNATURES for k in e["key_signatures"]])
    ]
    print("After removing according to ks:", len(dict_dataset))
    return dict_dataset


def score2midi_numbers(score):
    return [p.midi % 12 for n in score.flat.notes for p in n.pitches]


def score2pitches(score):
    return [p.name for n in score.flat.notes for p in n.pitches]


def score2pitches_meredith(score):
    # Return the David Meredith style of pitches (http://www.titanmusic.com/data.php)
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
    return [n.offset for n in score.flat.notes for p in n.pitches]


def score2durations(score):
    return [n.duration.quarterLength for n in score.flat.notes for p in n.pitches]


def score2voice(score):
    return [
        int(str(n.getContextByClass("Voice"))[-2])
        if not n.getContextByClass("Voice") is None
        else 1
        for n in score.flat.notes
        for p in n.pitches
    ]


def score2ks(score):
    """Return one ks for each pitch for each note"""
    temp_ks = None
    out = []
    for event in score.flat:
        if isinstance(event, m21.key.KeySignature):
            #             print("Found a ks")
            temp_ks = event.sharps
        elif isinstance(event, m21.note.NotRest):
            for pitch in event.pitches:
                #                 print("FOund a note")
                out.append(temp_ks)
    return out


def transp_note_list(note_list):
    """ For each input return len(accepted_intervals) transposed list of notes"""
    return [
        [n.transpose(interval) for n in note_list] for interval in accepted_intervals
    ]


def root_folder(p):
    return Path(p).parts[0]
