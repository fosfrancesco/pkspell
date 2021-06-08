from numba import njit
from pathlib import Path

import numpy as np

PAD = "<PAD>"


PITCHES = {
    0: ["C", "B#", "D--"],
    1: ["C#", "B##", "D-"],
    2: ["D", "C##", "E--"],
    3: ["D#", "E-", "F--"],
    4: ["E", "D##", "F-"],
    5: ["F", "E#", "G--"],
    6: ["F#", "E##", "G-"],
    7: ["G", "F##", "A--"],
    8: ["G#", "A-"],
    9: ["A", "G##", "B--"],
    10: ["A#", "B-", "C--"],
    11: ["B", "A##", "C-"],
}

INTERVALS = {
    0: ["P1", "d2", "A7"],
    1: ["m2", "A1"],
    2: ["M2", "d3", "AA1"],
    3: ["m3", "A2"],
    4: ["M3", "d4", "AA2"],
    5: ["P4", "A3"],
    6: ["d5", "A4"],
    7: ["P5", "d6", "AA4"],
    8: ["m6", "A5"],
    9: ["M6", "d7", "AA5"],
    10: ["m7", "A6"],
    11: ["M7", "d1", "AA6"],
}

DIATONIC_PITCHES = ["C", "D", "E", "F", "G", "A", "B"]

KEY_SIGNATURES = list(range(-7, 8))


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


@njit
def closest_multiple(n: int, x: int):
    if x > n:
        return x
    else:
        return int(x * np.ceil(n / x))


closest_multiple(6900, 2000)


def root_folder(p):
    return Path(p).parts[0]
