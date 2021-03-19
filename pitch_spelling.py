# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/fosfrancesco/pitch-spelling/blob/main/rnncrf_pitch_spelling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# ! pip install --upgrade pytorch-crf


# %%
# from google.colab import files

import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.model_selection
import torch

# from torchcrf import CRF
from torch.utils.data import DataLoader

# %% [markdown]
# # Pitch Spelling and ks Prediction
#
# Dataset: different authors from ASAP collection
# Challenges:
# - extremely long sequences
# - small dataset
# %% [markdown]
# ## Import ASAP dataset

# %%
# !git clone https://github.com/fosfrancesco/pitch-spelling.git

basepath = "./"  # to change if running locally or on colab
# load the asap datasets with ks
with open(Path("./asapks.pkl"), "rb") as fid:
    full_dict_dataset = pickle.load(fid)

# ####### Note for Nicolas: I called it "dict_dataset", but it is a list of dictionaries
paths = list(set([e["original_path"] for e in full_dict_dataset]))
print(f"{len(paths)} different pieces")
print(
    "Average number of notes: ",
    np.mean([len(e["midi_number"]) for e in full_dict_dataset]),
)

# %% [markdown]
# ## Chose the convenient data augmentation
# For each chromatic interval, take only the diatonic transposition that produce the smallest number of accidentals (or the original if present).
#
# Then remove the pieces with ks that have more than 7 sharps or 7 flats.

# %%
from pitches import keep_best_transpositions
from utils import root_folder

dict_dataset = keep_best_transpositions(full_dict_dataset)
# #test if it worked
# for i,e in enumerate(dict_dataset):
#     print(e["original_path"], e["transposed_of"], e["key_signatures"])
#     print(e["pitches"][:10])
#     print(e["midi_number"][:10])
#     if i == 100:
#         break

composer_per_piece = [root_folder(p) for p in paths]
c = Counter(composer_per_piece)

print(f"Dataset has {len(dict_dataset)} pieces.")
print(
    f"By composer: {', '.join(map(lambda item: item[0] + ': ' + str(item[1]), c.items()))}"
)


# %%
# remove pieces from asap that are in Musedata
print(len(paths), "initial pieces")
paths = [p for p in paths if p != "Bach/Prelude/bwv_865/xml_score.musicxml"]

# remove mozart Fantasie because of incoherent key signature
paths = [p for p in paths if p != "Mozart/Fantasie_475/xml_score.musicxml"]

print(len(paths), "pieces after removing overlapping with musedata and Mozart Fantasie")


# %%
# Temporary remove composer with only one piece, because they create problems with sklearn stratify
one_piece_composers = [
    "Balakirev",
    "Prokofiev",
    "Brahms",
    "Glinka",
    "Debussy",
    "Ravel",
    "Scriabin",
    "Liszt",
]
paths = [p for p in paths if root_folder(p) not in one_piece_composers]

# Divide train and validation set
path_train, path_validation = sklearn.model_selection.train_test_split(
    paths, test_size=0.15, stratify=[root_folder(p) for p in paths]
)
print("Train and validation lenghts: ", len(path_train), len(path_validation))

# need to find a better way to visualize this
composers = list(set([root_folder(p) for p in paths]))
print(f"Remaining composers: {composers}")

train_composer = [composers.index(root_folder(p)) for p in path_train]
val_composer = [composers.index(root_folder(p)) for p in path_validation]

# _ = plt.hist([train_composer, val_composer], label=["train", "validation"])
# _ = plt.legend(loc="upper left")
# _ = plt.xticks(list(range(len(composers))), composers)

# %% [markdown]
# ## Transform the input into a convenient format for the Model

# %%
# Helper functions to feed the correct input into the NN

from datasets import Pitch2Int, Ks2Int
from datasets import ToTensorLong, ToTensorFloat
from datasets import MultInputCompose
from datasets import DurationOneHotEncoder, MultInputCompose
from datasets import ks_to_ix, midi_to_ix, pitch_to_ix
from datasets import N_DURATION_CLASSES
from utils import PAD


# %%
# Create the dataset
from datasets import PSDataset
from datasets import transform_chrom
from datasets import transform_diat
from datasets import transform_key

train_dataset = PSDataset(
    dict_dataset,
    path_train,
    transform_chrom,
    transform_diat,
    transform_key,
    True,
    sort=True,
)
validation_dataset = PSDataset(
    dict_dataset, path_validation, transform_chrom, transform_diat, transform_key, False
)

print(f"Train pieces: {len(train_dataset)}, test pieces: {len(validation_dataset)}")


# test if it works
# for chrom, diat, ks, seq_len in train_dataset:
#     print(chrom.shape)
#     print(ks.shape)
#     print("Division", diat.shape[0] / 65)
#     #     print(torch.argmax(chrom[0:30],1))
#     #     # print([diatonic_pitches[p.item()] for p in diat[0:30]])
#     #     print([accepted_pitches[p.item()] for p in diat[0:30]])
#     print([p.item() for p in ks[-20:]])
#     print([p for p in chrom[-10:, :]])
#     print(seq_len)


# %%
from datasets import pad_collate

data_loader = DataLoader(
    dataset=validation_dataset,
    num_workers=1,
    batch_size=4,
    shuffle=True,
    collate_fn=pad_collate,
)

# test if it work
# for batch in data_loader:
#    print(batch[0].shape, batch[1].shape, batch[2].shape)
#    print(batch[0])
#    break

# %% [markdown]
# ## Model Definition

# %%
# TODO: search over the best hyperparameters


# %%
n_epochs = 30
HIDDEN_DIM = 96  # as it is implemented now, this is double the hidden_dim
LEARNING_RATE = 0.05
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
MOMENTUM = 0.9
RNN_LAYERS = 1

# ks rnn hyperparameter
HIDDEN_DIM2 = 48

# attention hyperparameter
NUM_HEAD = 2
NUM_LANDMARKS = 64  # should we make this depending on the seq length for each batch?


train_dataset = PSDataset(
    dict_dataset,
    path_train,
    transform_chrom,
    transform_diat,
    transform_key,
    True,
    sort=True,
    truncate=None,
)
validation_dataset = PSDataset(
    dict_dataset, path_validation, transform_chrom, transform_diat, transform_key, False
)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate
)
val_dataloader = DataLoader(
    validation_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate
)

# model = torch.load("./models/temp/model_temp_epoch30-to_restart.pkl")

from models import (
    RNNMultiTagger,
    RNNMultNystromAttentionTagger,
    RNNNystromAttentionTagger,
    RNNTagger,
)

# model = RNNTagger(len(midi_to_ix)+N_DURATION_CLASSES,HIDDEN_DIM,pitch_to_ix,ks_to_ix, n_layers =RNN_LAYERS)
model = RNNMultiTagger(
    len(midi_to_ix) + N_DURATION_CLASSES,
    HIDDEN_DIM,
    pitch_to_ix,
    ks_to_ix,
    n_layers=RNN_LAYERS,
)
# model = RNNNystromAttentionTagger(len(midi_to_ix)+N_DURATION_CLASSES,HIDDEN_DIM,pitch_to_ix,ks_to_ix, n_layers =RNN_LAYERS,num_head=NUM_HEAD,num_landmarks=NUM_LANDMARKS)
# model = RNNMultNystromAttentionTagger(
#     len(midi_to_ix) + N_DURATION_CLASSES,
#     HIDDEN_DIM,
#     pitch_to_ix,
#     ks_to_ix,
#     n_layers=RNN_LAYERS,
#     hidden_dim2=HIDDEN_DIM2,
#     num_head=NUM_HEAD,
#     num_landmarks=NUM_LANDMARKS,
# )

optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
from train import training_loop

history = training_loop(
    model, optimizer, train_dataloader, epochs=5, val_dataloader=val_dataloader
)

# model = torch.load("./models/model_RNNks.pkl")
# from inference import evaluate

# model = torch.load("./models/temp/model_temp_epoch6.pkl")
# musedata_noisy_path = Path(basepath, "./datasets/musedata_noisy.pkl")
# evaluate(model, musedata_noisy_path)


# %%


# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
# ### BEst accuracy with ks
# n_epochs = 30
# HIDDEN_DIM = 96
# LEARNING_WEIGHT = 0.05
# WEIGHT_DECAY = 1e-4
# BATCH_SIZE = 8
# MOMENTUM = 0.9
# RNN_LAYERS = 1
#
# model = RNNMultiTagger(len(midi_to_ix)+N_DURATION_CLASSES,HIDDEN_DIM,pitch_to_ix,ks_to_ix, n_layers =RNN_LAYERS)
#
# Model available in: ""./models/model_RNNks.pkl""
# accuracy on validation set 0.9424
# Trained on all asap dataset
#
# {'cor': 4, 'viv': 29, 'moz': 70, 'bac': 13, 'han': 15, 'bee': 99, 'hay': 273, 'tel': 8}
# {'cor': 0.9998366880333156, 'viv': 0.9988161815732539, 'moz': 0.9971421572630031, 'bac': 0.9994694960212201, 'han': 0.9993877551020408, 'bee': 0.9959580288245621, 'hay': 0.9888525928950592, 'tel': 0.9996734693877551}
# {'cor': 24493, 'viv': 24497, 'moz': 24494, 'bac': 24505, 'han': 24500, 'bee': 24493, 'hay': 24490, 'tel': 24500}
# Total errors : 511
#
#
# Epoch 22: train loss = 0.4723, train_accuracy: 0.9533,val_accuracy_pitch: 0.9424,val_accuracy_ks: 0.7938, time = 106.7093
#
#

# %%
