# PKSpell

A deep learning system for pitch spelling and key signature estimation. It can work directly on musical scores (see picture below), but it only require few input information that can be easily extracted from any kind of symbolic music representation (e.g. MIDI).

![Score Example](/tests/test_scores/example.jpg)

## Setup

Dependencies can be installed using `pip` (it is recommended to use a virtual environment with Python>=3.7) with the command `pip install -r requirements.txt`. 

We suggest to manually install PyTorch following the instructions in the [website](https://pytorch.org/get-started/locally/) to select the cpu or CUDA version.
Moreover, git must be installed and accessible to correctly download the asap dataset.

## Basic Usage
The system takes as inputs two lists of equal lengths: 
- a list of pitch-classes (obtainable from midi-numbers modulo 12);
- a list of durations in any format (e.g., milliseconds, seconds, quarterLengths, beats).

The output consists of two lists of the same lengths of the input lists:
- a list of tonal-pitch-classes (e.g. A#, Bb, D, D##);
- a list of key signatures, represented by the number of accidentals (sharps if the number is positive or flats if the number is negative). For example, Ab maj is represented with "-4", and D maj with "2".

If you already have a formatted musical score (e.g. a musicxml), it is possible to change the tonal-pitch-classes and key signature using pkspell with the command `python ./src/models/process_score.py --input-path [path]`. Note that we use music21 to load and save musicxml files, so files that cannot be correctly imported by music21 may not give a valid musical score.

A code example is contained in [notebooks/usage_example.ipynb](notebooks/usage_example.ipynb).

## Reproducibility
*(On Windows systems, substitute all "/" with "\\" for all path specifications.)* 

To retrain the model of the paper from the ASAP dataset:
1. Run `python src/data/download_dataset.py`.
1. Run `python src/data/prepare_dataset.py` (this can take some time, especially on slower hardware).
1. Run `python src/models/train.py`. Use the ``--device cuda`` flag for faster training (requires an NVIDIA GPU).
1. The model is saved in `models/temp`.


To evaluate the model on the MuseData dataset for the pitch-spelling task:
1. Run `python src/data/download_dataset.py`  
1. Run `python src/data/prepare_dataset.py`
1. Run `python src/models/inference.py`. Use the ``--device cuda`` flag for faster inference (requires an NVIDIA GPU).

The evaluation for the key-signature estimation task is not available, as the dataset is not public. Contact Francesco Foscarin for further information.


## Citing
If you use this approach in any research, please cite the relevant [paper](https://hal.archives-ouvertes.fr/hal-03300102):

```
@inproceedings{pkspell,
  title={{PKSpell}: Data-Driven Pitch Spelling and Key Signature Estimation},
  author={Foscarin, Francesco and Audebert, Nicolas  and Fournier S'niehotta, Raphaël},
  booktitle={International Society for Music Information Retrieval Conference {(ISMIR)}},
  year={2021},
}
```

## License
Licensed under the [MIT License](LICENSE).


