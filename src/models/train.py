import pickle
import random
import click
import numpy as np
import sklearn
import sklearn.model_selection
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.utils import keep_best_transpositions
from src.data.pytorch_datasets import (
    N_DURATION_CLASSES,
    PSDataset,
    pad_collate,
    ks_to_ix,
    midi_to_ix,
    pitch_to_ix,
    transform_pc,
    transform_tpc,
    transform_key,
)

# For reproducibility
# See https://pytorch.org/docs/stable/notes/randomness.html
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False


def training_loop(
    model,
    optimizer,
    train_dataloader,
    epochs=50,
    val_dataloader=None,
    device=None,
    scheduler=None,
):
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Training on device: {device}")

    # Move model to GPU if needed
    model = model.to(device)

    history = defaultdict(list)
    for i_epoch in range(1, epochs + 1):
        loss_sum = 0
        accuracy_pitch_sum = 0
        accuracy_ks_sum = 0
        model.train()
        for idx, (seqs, pitches, keysignatures, lens,) in enumerate(
            train_dataloader
        ):  # seqs, pitches, keysignatures, lens are batches
            seqs, pitches, keysignatures = (
                seqs.to(device),
                pitches.to(device),
                keysignatures.to(device),
            )
            optimizer.zero_grad()
            loss = model(seqs, pitches, keysignatures, lens)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            with torch.no_grad():
                predicted_pitch, predicted_ks = model.predict(seqs, lens)
                for i, (p, k) in enumerate(zip(predicted_pitch, predicted_ks)):
                    # compute the accuracy without considering the padding
                    acc_pitch = accuracy_score(p, pitches[:, i][: len(p)].cpu())
                    acc_ks = accuracy_score(k, keysignatures[:, i][: len(k)].cpu())
                    # normalize according to the number of sequences in the batch
                    accuracy_pitch_sum += acc_pitch / len(lens)
                    accuracy_ks_sum += acc_ks / len(lens)

        train_loss = loss_sum / len(train_dataloader)
        # normalize according to the number of batches
        train_accuracy_pitch = accuracy_pitch_sum / len(train_dataloader)
        train_accuracy_ks = accuracy_ks_sum / len(train_dataloader)
        history["train_loss"].append(train_loss)
        history["train_accuracy_pitch"].append(train_accuracy_pitch)
        history["train_accuracy_ks"].append(train_accuracy_ks)
        print(
            "Train Loss: {}, Train Accuracy (Pitch): {}, Train Accuracy (KS): {}".format(
                train_loss, train_accuracy_pitch, train_accuracy_ks
            )
        )

        if val_dataloader is not None:
            # Evaluate on the validation set
            model.eval()
            accuracy_pitch_sum = 0
            accuracy_ks_sum = 0
            with torch.no_grad():
                for seqs, pitches, keysignatures, lens in val_dataloader:
                    # Predict the model's output on a batch
                    predicted_pitch, predicted_ks = model.predict(seqs.to(device), lens)
                    # Update the lists that will be used to compute the accuracy
                    for i, (p, k) in enumerate(zip(predicted_pitch, predicted_ks)):
                        # compute the accuracy without considering the padding
                        acc_pitch = accuracy_score(p, pitches[:, i][: len(p)].cpu())
                        acc_ks = accuracy_score(k, keysignatures[:, i][: len(k)].cpu())
                        # normalize according to the number of sequences in the batch
                        accuracy_pitch_sum += acc_pitch / len(lens)
                        accuracy_ks_sum += acc_ks / len(lens)

                # normalize according to the number of batches
                val_accuracy_pitch = accuracy_pitch_sum / len(val_dataloader)
                val_accuracy_ks = accuracy_ks_sum / len(val_dataloader)

            history["val_accuracy_pitch"].append(val_accuracy_pitch)
            history["val_accuracy_ks"].append(val_accuracy_ks)
            print(
                "Validation Accuracy (Pitch): {}, Validation Accuracy (KS): {}".format(
                    val_accuracy_pitch, val_accuracy_ks
                )
            )

        if scheduler is not None:
            scheduler.step()

        # save the model
        torch.save(model, Path("./models/temp/model_temp_epoch{}.pkl".format(i_epoch)))

    return history


def train_pkspell(
    model,
    epochs,
    lr,
    hidden_dim,
    momentum,
    hidden_dim2,
    rnn_depth,
    device,
    dropout,
    dropout2,
    rnn_cell,
    weight_decay,
    optimizer,
    bidirectional,
    mode,
    train_dataloader,
    val_dataloader=None,
):
    from models import PKSpell, PKSpell_single

    if model == "PKSpellsingle":
        model = PKSpell_single(
            len(midi_to_ix) + N_DURATION_CLASSES,
            hidden_dim,
            pitch_to_ix,
            ks_to_ix,
            rnn_depth=rnn_depth,
            dropout=dropout,
            cell_type=rnn_cell,
            bidirectional=bidirectional,
            mode=mode,
        )
    elif model == "PKSpell":
        model = PKSpell(
            len(midi_to_ix) + N_DURATION_CLASSES,
            hidden_dim,
            pitch_to_ix,
            ks_to_ix,
            rnn_depth=rnn_depth,
            dropout=dropout,
            dropout2=dropout2,
            hidden_dim2=hidden_dim2,
            cell_type=rnn_cell,
            bidirectional=bidirectional,
            mode=mode,
        )
    else:
        raise Exception("Model must be either 'PKSpellsingle' or  'PkSpell'")

    from torch import optim
    from torch.optim import lr_scheduler

    if optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, [epochs // 2], gamma=0.1, verbose=True
        )
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, [epochs // 2], gamma=0.1, verbose=True
        )
    else:
        raise Exception("Only SGD and Adam are supported optimizers.")

    history = training_loop(
        model,
        optimizer,
        train_dataloader,
        epochs=epochs,
        val_dataloader=val_dataloader,
        device=device,
        scheduler=scheduler,
    )

    return model, history


# %%
@click.command()
@click.option("--model", default="PKSpell", type=str)
@click.option("--epochs", default=40, type=int)
@click.option("--lr", default=0.01, type=float)
@click.option("--momentum", default=0.9, type=float)
@click.option("--decay", default=1e-4, type=float)
@click.option("--hidden_dim", default=300, type=int)
@click.option("--hidden_dim2", default=24, type=int)
@click.option("--bs", default=32, type=int)
@click.option("--rnn_depth", default=1, type=int)
@click.option("--device", type=str)
@click.option("--dropout", default=0.5, type=float)
@click.option("--dropout2", default=0.5, type=float)
@click.option("--cell", default="GRU", type=str)
@click.option("--optimizer", default="Adam", type=str)
@click.option("--learn_all", is_flag=True, default=False)
@click.option("--bidirectional", default=True, type=bool)
@click.option("--mode", default="both", type=str)
@click.option("--augmentation", default=True, type=bool)
def start_experiment(
    model,
    epochs,
    lr,
    hidden_dim,
    bs,
    momentum,
    hidden_dim2,
    rnn_depth,
    device,
    dropout,
    dropout2,
    cell,
    decay,
    optimizer,
    learn_all,
    bidirectional,
    mode,
    augmentation,
):
    print("Loading the augmented ASAP dataset")
    # load the asap datasets with ks
    with open(Path("./data/processed/asapks.pkl"), "rb") as fid:
        full_list_of_dict_dataset = pickle.load(fid)

    paths = list(set([e["original_path"] for e in full_list_of_dict_dataset]))
    list_of_dict_dataset = keep_best_transpositions(full_list_of_dict_dataset)

    # remove pieces from asap that are in Musedata
    paths = [p for p in paths if p != "Bach/Prelude/bwv_865/xml_score.musicxml"]
    # remove mozart Fantasie because of incoherent key signature
    paths = [p for p in paths if p != "Mozart/Fantasie_475/xml_score.musicxml"]
    paths = sorted(paths)

    from functools import partial

    trainer = partial(
        train_pkspell,
        model,
        epochs,
        lr,
        hidden_dim,
        momentum,
        hidden_dim2,
        rnn_depth,
        device,
        dropout,
        dropout2,
        cell,
        decay,
        optimizer,
        bidirectional,
        mode,
    )

    def train_dataloader(ds):
        return DataLoader(
            ds, batch_size=bs, shuffle=True, collate_fn=pad_collate, num_workers=2,
        )

    def val_dataloader(ds):
        return DataLoader(
            ds, batch_size=1, shuffle=False, collate_fn=pad_collate, num_workers=1,
        )

    if learn_all:
        print("Learning from full dataset")
        train_dataset = PSDataset(
            list_of_dict_dataset,
            paths,
            transform_pc,
            transform_tpc,
            transform_key,
            augment_dataset=augmentation,
            sort=True,
            truncate=None,
        )
        _, history = trainer(train_dataloader(train_dataset),)
    else:
        paths = np.array(paths)
        # Divide train and validation set
        path_train, path_validation = sklearn.model_selection.train_test_split(
            paths, test_size=0.15, random_state=seed,
        )
        print(path_validation)
        print("Train and validation lenghts: ", len(path_train), len(path_validation))
        train_dataset = PSDataset(
            list_of_dict_dataset,
            path_train,
            transform_pc,
            transform_tpc,
            transform_key,
            augment_dataset=augmentation,
            sort=True,
            truncate=None,
        )
        validation_dataset = PSDataset(
            list_of_dict_dataset,
            path_validation,
            transform_pc,
            transform_tpc,
            transform_key,
            augment_dataset=False,
        )
        _, history = trainer(
            train_dataloader(train_dataset), val_dataloader(validation_dataset),
        )


if __name__ == "__main__":
    start_experiment()
