import music21 as m21
from pathlib import Path
import torch
import click
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.prepare_dataset import score2pc, score2durations
from src.models.models import PKSpell
from src.models.inference import single_piece_predict



def process_score(input_path, output_path, model, device):
    print("Parsing the musical score")
    score = m21.converter.parse(str(Path(input_path)))
    # extract note list
    notes = [p for n in score.flat.notes for p in n.pitches]
    # extract features for each note
    pitch_classes = score2pc(score)
    durations = score2durations(score)
    # run the DL system
    print("Running PKSpell")
    tpcs, kss = single_piece_predict(pitch_classes, durations, model, device)
    # update the pitch spelling in the score
    for i, n in enumerate(notes):
        # correct the octave jump problem for C and B#
        if n.name == "C" and tpcs[i] == "B#":
            n.octave = n.octave - 1
        if n.name == "B#" and tpcs[i] == "C":
            n.octave == n.octave + 1
        # set the correct pitch spelling class
        n.name = tpcs[i]

    # update the ks in the score
    # WARNING : only consider a single ks
    for event in score.flat:
        if isinstance(event, m21.key.KeySignature):
            event.sharps = max(set(kss), key=kss.count)

    score.write("musicxml", fp=Path(output_path))
    print("Score saved in ",output_path)

@click.command()
@click.option(
    "--input-path",
    help="Path to a musicxml file",
    type=click.Path(exists=True),
    default=Path("./tests/test_scores/bach_bwv867P_wrong.xml"),
)
@click.option(
    "--output-path",
    help="Path to save the new processed musicxml file",
    type=click.Path(exists=False),
    default=Path("./tests/test_scores/pkspelled_score.xml"),
)
@click.option(
    "--device",
    default="cpu",
    help='Device (default="cpu", use "cuda" for faster computation)',
)
def pkspell_score(input_path, output_path, device):
    model = PKSpell()
    model.load_state_dict(torch.load(Path("./models/pkspell_statedict.pt")))
    process_score(input_path, output_path, model,device)


if __name__ == "__main__":
    process_score()
