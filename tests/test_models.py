import pytest

import numpy as np
import torch
from src.models.models import PKSpell, PKSpell_single
from src.data.pytorch_datasets import pitch_to_ix, ks_to_ix
from pathlib import Path


def test_PKSpell_single():
    n_features = 12
    piece_len = [20, 9, 8, 5]
    batch_size = 4

    model = PKSpell_single(
        n_features,
        12,
        pitch_to_ix,
        ks_to_ix,
        rnn_depth=1,
        cell_type="GRU",
        dropout=None,
        bidirectional=True,
        mode="both",
    )
    dummy_input_midi = np.random.randint(
        0, 12, size=(max(piece_len), batch_size, n_features)
    )
    dummy_pitch = np.random.randint(
        0, len(pitch_to_ix), size=(max(piece_len), batch_size)
    )
    dummy_ks = np.random.randint(0, len(ks_to_ix), size=(max(piece_len), batch_size))

    # try a training
    loss = model(
        torch.Tensor(dummy_input_midi),
        torch.Tensor(dummy_pitch).long(),
        torch.Tensor(dummy_ks).long(),
        torch.Tensor(piece_len),
    )
    assert loss.shape == torch.Size([])
    # try a prediction
    prediction = model.predict(
        torch.Tensor(dummy_input_midi), torch.Tensor([20, 9, 8, 5])
    )
    assert type(prediction) == tuple
    assert len(prediction[0]) == batch_size
    assert len(prediction[1]) == batch_size
    for i, l in enumerate(piece_len):
        assert len(prediction[0][i]) == l
        assert len(prediction[1][i]) == l


def test_PKSpell():
    n_features = 12
    piece_len = [20, 9, 8, 5]
    batch_size = 4

    model = PKSpell(
        n_features,
        12,
        pitch_to_ix,
        ks_to_ix,
        rnn_depth=1,
        cell_type="GRU",
        dropout=None,
        bidirectional=True,
        mode="both",
    )
    dummy_input_midi = np.random.randint(
        0, 12, size=(max(piece_len), batch_size, n_features)
    )
    dummy_pitch = np.random.randint(
        0, len(pitch_to_ix), size=(max(piece_len), batch_size)
    )
    dummy_ks = np.random.randint(0, len(ks_to_ix), size=(max(piece_len), batch_size))

    # try a training
    loss = model(
        torch.Tensor(dummy_input_midi),
        torch.Tensor(dummy_pitch).long(),
        torch.Tensor(dummy_ks).long(),
        torch.Tensor(piece_len),
    )
    assert loss.shape == torch.Size([])
    # try a prediction
    prediction = model.predict(
        torch.Tensor(dummy_input_midi), torch.Tensor([20, 9, 8, 5])
    )
    assert type(prediction) == tuple
    assert len(prediction[0]) == batch_size
    assert len(prediction[1]) == batch_size
    for i, l in enumerate(piece_len):
        assert len(prediction[0][i]) == l
        assert len(prediction[1][i]) == l


def test_PKSpell_odd_hidden_dim():
    hidden_dim = 11
    hidden_dim2 = 7

    with pytest.raises(ValueError):
        model = PKSpell(
            5,
            hidden_dim,
            pitch_to_ix,
            ks_to_ix,
            rnn_depth=1,
            cell_type="GRU",
            dropout=None,
            bidirectional=True,
            mode="both",
        )

    with pytest.raises(ValueError):
        model = PKSpell(
            5,
            10,
            pitch_to_ix,
            ks_to_ix,
            rnn_depth=1,
            cell_type="GRU",
            dropout=None,
            bidirectional=True,
            mode="both",
            hidden_dim2=hidden_dim2,
        )


def test_import_model():
    # import pkspell
    model = torch.load(Path("models/pkspell.pt"))

    # import pkspell_single
    model = torch.load(Path("models/pkspell_single.pt"))
    assert True


def test_import_pretrained_state_dict():
    # import pkspell
    model = PKSpell(17, 300, pitch_to_ix, ks_to_ix, hidden_dim2=24)
    model.load_state_dict(torch.load(Path("models/pkspell_statedict.pt")))

    # import pkspell_single
    model = PKSpell_single(17, 300, pitch_to_ix, ks_to_ix)
    model.load_state_dict(torch.load(Path("models/pkspell_single_statedict.pt")))
    assert True


from src.models.inference import single_piece_predict


def test_single_predict():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = PKSpell(17, 300, pitch_to_ix, ks_to_ix, hidden_dim2=24)
    model.load_state_dict(torch.load(Path("models/pkspell_statedict.pt")))
    p_list = [3, 5, 7, 8, 0, 3, 8, 1, 5, 8, 5, 3, 1, 0, 10, 8]
    d_list = [1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3]

    single_piece_predict(p_list, d_list, model, device)
    assert True
