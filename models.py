import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PAD
from utils import closest_multiple


class RNNTagger(nn.Module):
    """Vanilla RNN Model, slide 12 of powerpoint presentation"""

    def __init__(self, input_dim, hidden_dim, pitch_to_ix, ks_to_ix, n_layers=1):
        super(RNNTagger, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)
        self.hidden_dim = hidden_dim

        # RNN layer. We're using a bidirectional GRU
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            num_layers=n_layers,
        )

        # Output layer. The input will be two times
        # the RNN size since we are using a bidirectional RNN.
        self.top_layer_pitch = nn.Linear(hidden_dim, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])

    def compute_outputs(self, sentences, sentences_len):
        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

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
        return self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
            scores_ks, keysignatures
        )

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return [
            predicted_pitch[: int(l), i].cpu().numpy()
            for i, l in enumerate(sentences_len)
        ], [predicted_ks[: int(l), i].cpu().numpy() for i, l in enumerate(sentences_len)]


class RNNMultiTagger(nn.Module):
    """RNN decoupling key from pitch spelling by adding a second RNN,
    slide 13 of powerpoint presentation
    """

    def __init__(
        self, input_dim, hidden_dim, pitch_to_ix, ks_to_ix, hidden_dim2=24, n_layers=1,dropout=None
    ):
        super(RNNMultiTagger, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        # RNN layer. We're using a bidirectional GRU
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            num_layers=n_layers,
        )
        self.rnn2 = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim2 // 2,
            bidirectional=True,
            num_layers=n_layers,
        )

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout

        # Output layer. The input will be two times
        # the RNN size since we are using a bidirectional RNN.
        self.top_layer_pitch = nn.Linear(hidden_dim, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim2, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])

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

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)
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
        return self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
            scores_ks, keysignatures
        )

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return [
            predicted_pitch[: int(l), i].cpu().numpy()
            for i, l in enumerate(sentences_len)
        ], [predicted_ks[: int(l), i].cpu().numpy() for i, l in enumerate(sentences_len)]


class NystromAttention(nn.Module):
    def __init__(self, head_dim, num_head, num_landmarks):
        super().__init__()

        self.head_dim = head_dim
        self.num_head = num_head

        self.num_landmarks = num_landmarks

    def forward(self, Q, K, V, mask):
        seq_len = mask.shape[1]

        Q = Q * mask[:, None, :, None] / np.sqrt(np.sqrt(self.head_dim))
        K = K * mask[:, None, :, None] / np.sqrt(np.sqrt(self.head_dim))

        if self.num_landmarks == seq_len:
            attn = nn.functional.softmax(
                torch.matmul(Q, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]),
                dim=-1,
            )
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(
                -1,
                self.num_head,
                self.num_landmarks,
                seq_len // self.num_landmarks,
                self.head_dim,
            ).mean(dim=-2)
            K_landmarks = K.reshape(
                -1,
                self.num_head,
                self.num_landmarks,
                seq_len // self.num_landmarks,
                self.head_dim,
            ).mean(dim=-2)

            kernel_1 = F.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim=-1)
            kernel_2 = F.softmax(
                torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim=-1
            )
            kernel_3 = F.softmax(
                torch.matmul(Q_landmarks, K.transpose(-1, -2))
                - 1e9 * (1 - mask[:, None, None, :]),
                dim=-1,
            )
            X = torch.matmul(
                torch.matmul(kernel_1, self.iterative_inv(kernel_2)),
                torch.matmul(kernel_3, V),
            )

        return X

    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat
        V = (
            1
            / (
                torch.max(torch.sum(torch.abs(K), dim=-2))
                * torch.max(torch.sum(torch.abs(K), dim=-1))
            )
            * K.transpose(-1, -2)
        )
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(
                0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV))
            )
        return V

    def extra_repr(self):
        # return f"num_landmarks={self.num_landmarks}, seq_len={seq_len}"
        return f"num_landmarks={self.num_landmarks}"


class Attention(nn.Module):
    def __init__(self, dim, num_head, num_landmarks):
        """
        dim : if used after a rnn, dim is the hidden dim of the rnn

        """
        super().__init__()

        self.dim = dim
        self.num_head = num_head
        self.num_landmarks = num_landmarks

        self.W_q = nn.Linear(self.dim, self.num_head * self.dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.dim)

        self.attn = NystromAttention(self.dim, self.num_head, self.num_landmarks)

    def forward(self, X, sentences_len):
        # transpose matrix to shape [Batch,seq_len,features]
        X = torch.transpose(X, 0, 1)
        # pad to multiple of self.num_landmarks (pad with value "0", should we change it?)
        padding_length = closest_multiple(X.shape[1], self.num_landmarks) - X.shape[1]
        X = F.pad(X, (0, 0, 0, padding_length, 0, 0), "constant", 0)
        # compute padding mask (1 for elements to consider, ignore 0)
        pad_mask = torch.arange(X.shape[1])[None, :] < sentences_len[:, None]

        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))
        with torch.cuda.amp.autocast(enabled=False):
            attn_out = self.attn(
                Q.float(),
                K.float(),
                V.float(),
                pad_mask.float().to(next(self.W_q.parameters()).device),
            )
        out = self.combine_heads(attn_out)

        # slice to the original shape
        out = out[:, : int(torch.max(sentences_len)), :]

        # transpose back to shape [seq_len,batch,features]
        out = torch.transpose(out, 0, 1)
        return out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.dim)
        X = X.transpose(1, 2)
        return X


class RNNNystromAttentionTagger(nn.Module):
    """Vanilla RNN + Nystrom Attention"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        pitch_to_ix,
        ks_to_ix,
        n_layers=1,
        num_head=1,
        num_landmarks=64,
    ):
        super(RNNNystromAttentionTagger, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)
        self.hidden_dim = hidden_dim
        self.num_landmarks = num_landmarks

        # RNN layer. We're using a bidirectional GRU
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            num_layers=n_layers,
        )

        # Output layer. The input will be two times
        # the RNN size since we are using a bidirectional RNN.
        self.top_layer_pitch = nn.Linear(hidden_dim * num_head, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim * num_head, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])

        # attention function
        self.attention = Attention(self.hidden_dim, num_head, num_landmarks)

    def compute_outputs(self, sentences, sentences_len):
        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        # use attention
        attn_applied = self.attention(rnn_out, sentences_len)

        out_pitch = self.top_layer_pitch(attn_applied)
        out_ks = self.top_layer_ks(attn_applied)

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
        return self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
            scores_ks, keysignatures
        )

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return [
            predicted_pitch[: int(l), i].cpu().numpy()
            for i, l in enumerate(sentences_len)
        ], [predicted_ks[: int(l), i].cpu().numpy() for i, l in enumerate(sentences_len)]


class RNNMultNystromAttentionTagger(nn.Module):
    """Pitch-key decoupled model + 2 Nystrom Attention (one att for pich + one att for key)"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        pitch_to_ix,
        ks_to_ix,
        n_layers=1,
        hidden_dim2=24,
        num_head=1,
        num_landmarks=64,
    ):
        super(RNNMultNystromAttentionTagger, self).__init__()

        self.n_out_pitch = len(pitch_to_ix)
        self.n_out_ks = len(ks_to_ix)
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.num_landmarks = num_landmarks

        # RNN layer. Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            num_layers=n_layers,
        )
        self.rnn2 = nn.GRU(
            input_size=hidden_dim * num_head,
            hidden_size=hidden_dim2 // 2,
            bidirectional=True,
            num_layers=n_layers,
        )

        # Output layer. The input will be two times
        # the RNN size since we are using a bidirectional RNN.
        self.top_layer_pitch = nn.Linear(hidden_dim * num_head, self.n_out_pitch)
        self.top_layer_ks = nn.Linear(hidden_dim2 * num_head, self.n_out_ks)

        # Loss function that we will use during training.
        self.loss_pitch = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=pitch_to_ix[PAD]
        )
        self.loss_ks = nn.CrossEntropyLoss(reduction="mean", ignore_index=ks_to_ix[PAD])

        # attention function
        self.attention_pitch = Attention(self.hidden_dim, num_head, num_landmarks)
        self.attention_ks = Attention(self.hidden_dim2, num_head, num_landmarks)

    def compute_outputs(self, sentences, sentences_len):
        sentences = nn.utils.rnn.pack_padded_sequence(sentences, sentences_len)
        rnn_out, _ = self.rnn(sentences)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        # use attention pitch
        rnn_out = self.attention_pitch(rnn_out, sentences_len)

        out_pitch = self.top_layer_pitch(rnn_out)

        # pass the ks information into the second rnn
        rnn_out = nn.utils.rnn.pack_padded_sequence(rnn_out, sentences_len)
        rnn_out, _ = self.rnn2(rnn_out)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        # use attention ks
        attn_applied = self.attention_ks(rnn_out, sentences_len)

        out_ks = self.top_layer_ks(attn_applied)

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
        return self.loss_pitch(scores_pitch, pitches) + self.loss_ks(
            scores_ks, keysignatures
        )

    def predict(self, sentences, sentences_len):
        # Compute the outputs from the linear units.
        scores_pitch, scores_ks = self.compute_outputs(sentences, sentences_len)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted_pitch = scores_pitch.argmax(dim=2)
        predicted_ks = scores_ks.argmax(dim=2)
        return [
            predicted_pitch[: int(l), i].cpu().numpy()
            for i, l in enumerate(sentences_len)
        ], [predicted_ks[: int(l), i].cpu().numpy() for i, l in enumerate(sentences_len)]
