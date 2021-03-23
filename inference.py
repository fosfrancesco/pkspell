import click
import numpy as np
import pickle
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from datasets import PSDataset
from datasets import transform_chrom
from datasets import transform_diat
from datasets import transform_key
from datasets import pad_collate


def evaluate(model, dataset_path, device=None):
    # load the dataset
    with open(dataset_path, "rb") as fid:
        full_mdata_dict_dataset = pickle.load(fid)

    # add dummy ks to have the same format as asap
    for e in full_mdata_dict_dataset:
        e["key_signatures"] = np.zeros(len(e["pitches"]))
    mdata_paths = list(set([e["original_path"] for e in full_mdata_dict_dataset]))

    # # remove the symbphony No.100 from Haydn because of the enharmonic transposition
    # paths.remove("datasets\\opnd\\haydndoversyms-10004m.opnd-m")

    # print(paths)
    print(len(mdata_paths), "different pieces")
    print(
        "Average number of notes: ",
        np.mean([len(e["midi_number"]) for e in full_mdata_dict_dataset]),
    )

    mdata_dataset = PSDataset(
        full_mdata_dict_dataset,
        mdata_paths,
        transform_chrom,
        transform_diat,
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
            e["original_path"].split("\\")[-1][:3]
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
    print(errors_per_author_pitch)
    print(accuracy_per_author_pitch)
    print(notes_per_author)
    print("Total errors :", sum([e for e in errors_per_author_pitch.values()]))
    print("Error rate:")
    print(
        {
            k: (1 - accuracy_per_author_pitch[k]) * 100
            for k in accuracy_per_author_pitch.keys()
        }
    )

    print(
        "Total error rate:",
        sum(errors_per_author_pitch.values()) / sum(notes_per_author.values()) * 100,
    )


@click.command()
@click.option("--model", help="Path to a saved PyTorch .pt model")
@click.option("--dataset", help="Path to one of the datasets")
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
