import click
import numpy as np
import pickle
import torch
import os

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.pytorch_datasets import (
    PSDataset,
    transform_pc,
    transform_tpc,
    transform_key,
    pad_collate,
)
from src.utils.constants import accepted_pitches, KEY_SIGNATURES

def single_piece_prepare_data(p_list, d_list): 
    # transform
    out= transform_pc(p_list, d_list)
    # reshape in a convenient representation
    return out.view(-1,1,17)

def single_piece_predict(p_list, d_list, model, device):
    input = single_piece_prepare_data(p_list, d_list)
    # reshape in a convenient representation
    model.eval()
    with torch.no_grad():
        input = input.to(device)
        tpc,ks = model.predict(input,[len(input)])
    return [accepted_pitches[e] for e in tpc[0]],[KEY_SIGNATURES[e] for e in ks[0]]

def evaluate(model, dataset_path, device=None):
    # load the dataset
    with open(dataset_path, "rb") as fid:
        full_mdata_dict_dataset = pickle.load(fid)

    # add dummy ks to have the same format as asap
    for e in full_mdata_dict_dataset:
        e["key_signatures"] = np.zeros(len(e["pitches"]))
    mdata_paths = list(set([e["original_path"] for e in full_mdata_dict_dataset]))
    print(len(mdata_paths), "different pieces")

    # load dataset in pytorch convenient classes
    mdata_dataset = PSDataset(
        full_mdata_dict_dataset,
        mdata_paths,
        transform_pc,
        transform_tpc,
        transform_key,
        sort=False,
        augment_dataset=False,
    )
    mdata_dataloader = DataLoader(
        mdata_dataset, batch_size=2, shuffle=False, collate_fn=pad_collate
    )

    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    print(f"Using device: {device}")
    model = model.to(device)

    all_inputs = []
    all_predicted_pitch = []
    all_predicted_ks = []
    all_pitches = []
    all_ks = []
    model.eval()  # Evaluation mode (e.g. disable dropout)
    with torch.no_grad():  # Disable gradient tracking
        for seqs, pitches, ks, lens in mdata_dataloader:
            # Move data to device
            seqs = seqs.to(device)

            # Predict the model's output on a batch.
            predicted_pitch, predicted_ks = model.predict(seqs, lens)
            # Update the evaluation statistics.
            for i, p in enumerate(predicted_pitch):
                all_inputs.append(
                    torch.argmax(seqs[0 : int(lens[i]), i, :].cpu(), 1).numpy()
                )
                all_predicted_pitch.append(p)
                all_predicted_ks.append(predicted_ks[i])
                all_pitches.append(pitches[0 : int(lens[i]), i])
                all_ks.append(ks[0 : int(lens[i]), i])

    # Divide accuracy according to author
    authors = []

    for sequence in all_inputs:
        #     print(sequence)
        author = [
            e["original_path"].split(os.sep)[-1][:3]
            for e in full_mdata_dict_dataset
            if len(e["midi_number"]) == len(sequence)
            and list(e["midi_number"]) == list(sequence)
        ]
        assert len(author) == 1
        authors.append(author[0])

    considered_authors = list(set(authors))
    print(considered_authors)

    errors_per_author_pitch = {}
    accuracy_per_author_pitch = {}
    notes_per_author = {}
    for ca in considered_authors:
        ca_predicted_pitch = np.concatenate(
            [all_predicted_pitch[i] for i, a in enumerate(authors) if a == ca]
        )
        ca_predicted_ks = np.concatenate(
            [all_predicted_ks[i] for i, a in enumerate(authors) if a == ca]
        )
        ca_pitches = np.concatenate(
            [all_pitches[i] for i, a in enumerate(authors) if a == ca]
        )
        ca_ks = np.concatenate([all_ks[i] for i, a in enumerate(authors) if a == ca])

        ca_acc_pitch = accuracy_score(ca_predicted_pitch, ca_pitches)

        accuracy_per_author_pitch[ca] = float(ca_acc_pitch)
        errors_per_author_pitch[ca] = int(
            len(ca_pitches) - sum(np.equal(ca_predicted_pitch, ca_pitches))
        )
        notes_per_author[ca] = len(ca_pitches)

    print("Pitch Statistics----------------")
    print("Errors:")
    print(errors_per_author_pitch)
    print("Total errors :", sum([e for e in errors_per_author_pitch.values()]))
    print("Accuracy:")
    print(accuracy_per_author_pitch)
    print(
        "Total accuracy:",
        accuracy_score(np.concatenate(all_predicted_pitch), np.concatenate(all_pitches))
,
    )
    print("Error rate (as a percentage):")
    print(
        {
            k: (1 - accuracy_per_author_pitch[k]) * 100
            for k in accuracy_per_author_pitch.keys()
        }
    )

    print(
        "Total error rate (as a percentage):",
        (1- accuracy_score(np.concatenate(all_predicted_pitch), np.concatenate(all_pitches))) * 100,
    )


@click.command()
@click.option("--model", help="Path to a saved PyTorch .pt model", type=click.Path(exists=True), default=Path("./models/pkspell.pt"))
@click.option("--dataset", help="Path to one of the preprocessed datasets", type=click.Path(exists=True), default=Path("./data/processed/musedata.pkl") )
@click.option(
    "--device",
    default="cpu",
    help='Device (default="cpu", use "cuda" for faster computation)',
)
def run_evaluate(model, dataset, device):
    model = torch.load(model)
    evaluate(model, dataset, device=device)


if __name__ == "__main__":
    run_evaluate()
