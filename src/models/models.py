import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PAD
from utils import closest_multiple


class RNNTagger(nn.Module):
    """Vanilla RNN Model, slide 12 of powerpoint presentation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        pitch_to_ix,
        ks_to_ix,
        n_layers=1,
        cell_type="GRU",
        dropout=None,
        bidirectional=True,
        mode="both",
    ):
        super(RNNTagger, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)
        self.hidden_dim = hidden_dim

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer. We're using a bidirectional GRU
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            num_layers=n_layers,
            # dropout=dropout if dropout is not None else 0
        )

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        # Output layer. The input will be two times
        # the RNN size since we are using a bidirectional RNN.
        self.top_layer_pitch = nn.Linear(hidden_dim, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len):
        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)

        out_pitch = self.top_layer_pitch(rnn_out)
        out_ks = self.top_layer_ks(rnn_out)

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)
        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        return loss

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )


class RNNMultiTagger(nn.Module):
    """RNN decoupling key from pitch spelling by adding a second RNN,
    slide 13 of powerpoint presentation
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        pitch_to_ix,
        ks_to_ix,
        hidden_dim2=24,
        n_layers=1,
        dropout=None,
        dropout2=None,
        cell_type="GRU",
        bidirectional=True,
        mode="both",
    ):
        super(RNNMultiTagger, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        if cell_type == "GRU":
            rnn_cell = nn.GRU
        elif cell_type == "LSTM":
            rnn_cell = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN cell type: {cell_type}")

        # RNN layer. We're using a bidirectional GRU
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            bidirectional=bidirectional,
            # dropout=dropout,
            num_layers=n_layers,
        )
        self.rnn2 = rnn_cell(
            input_size=hidden_dim,
            hidden_size=hidden_dim2 // 2 if bidirectional else hidden_dim2,
            bidirectional=bidirectional,
            # dropout=dropout,
            num_layers=n_layers,
        )

        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if dropout2 is not None and dropout2 > 0:
            self.dropout2 = nn.Dropout(p=dropout2)
        else:
            self.dropout2 = None

        # Output layer. The input will be two times
        # the RNN size since we are using a bidirectional RNN.
        self.top_layer_pitch = nn.Linear(hidden_dim, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim2, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])
        self.mode = mode

    def compute_outputs(self, sentences, sentences_len):
        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)
        out_pitch = self.top_layer_pitch(rnn_out)

        # pass the ks information into the second rnn
        rnn_out = nn.utils.rnn.pack_padded_sequence(rnn_out, sentences_len)
        rnn_out, _ = self.rnn2(rnn_out)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        if self.dropout2 is not None:
            rnn_out = self.dropout2(rnn_out)
        out_ks = self.top_layer_ks(rnn_out)

        return out_pitch, out_ks

    def forward(self, sentences, pitches, keysignatures, sentences_len):
        # First computes the predictions, and then the loss function.

        # Compute the outputs. The shape is (max_len, n_sentences, n_labels).
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores_pitch = scores_pitch.view(-1, self.n_out_pitch)
        scores_ks = scores_ks.view(-1, self.n_out_ks)
        pitches = pitches.view(-1)
        keysignatures = keysignatures.view(-1)
        if self.mode == "both":
            loss = self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
                scores_ks, keysignatures
            )
        elif self.mode == "ks":
            loss = self.loss_ks(scores_ks, keysignatures)
        elif self.mode == "ps":
            loss = self.loss_pitch(scores_pitch, pitches)
        else:
            raise Exception("self.mode can only be in ['both','ks','ps']")
        return loss

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return (
            [
                predicted_pitch[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
            [
                predicted_ks[: int(l), i].cpu().numpy()
                for i, l in enumerate(sentences_len)
            ],
        )

