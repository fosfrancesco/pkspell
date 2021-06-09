from numba import njit
from pathlib import Path

import numpy as np

from src.utils.constants import INTERVALS, KEY_SIGNATURES


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
