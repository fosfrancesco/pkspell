import pytest
import torch

from src.data.pytorch_datasets import (
    PSDataset,
    transform_pc,
    transform_tpc,
    transform_key,
)
from src.utils.constants import accepted_pitches


def test_PSDataset():
    list_of_dict_dataset = [
        {
            "midi_number": [1, 5, 8, 5, 1],
            "transposed_of": "P1",
            "original_path": "path1",
            "pitches": ["C#", "E#", "G#", "E#", "C#"],
            "duration": [1, 1, 1, 1, 3],
            "key_signatures": [7, 7, 7, 7, 7],
        },
        {
            "midi_number": [2, 6, 9, 6, 2],
            "transposed_of": "P1",
            "original_path": "path2",
            "pitches": ["D", "F#", "A", "F#", "D"],
            "duration": [1, 1, 1, 1, 3],
            "key_signatures": [2, 2, 2, 2, 2],
        },
    ]
    paths = ["path1"]
    transf_c = transform_pc
    transf_d = transform_tpc
    transf_k = transform_key

    dataset = PSDataset(list_of_dict_dataset, paths, transf_c, transf_d, transf_k)

    # extract descriptors for one piece
    for e in dataset:
        chromatic_seq, diatonic_seq, ks_seq, seq_len = e
        break
    assert torch.equal(chromatic_seq, transform_pc([1, 5, 8, 5, 1], [1, 1, 1, 1, 3]))
    assert torch.equal(diatonic_seq, transform_tpc(["C#", "E#", "G#", "E#", "C#"]))
    assert torch.equal(ks_seq, transform_key([7, 7, 7, 7, 7]))
    assert seq_len == 5
