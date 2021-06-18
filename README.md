# PKSpell

A deep learning system for pitch spelling and key signature estimation.

The dependencies are listed in the file [environment.yml](environment.yml).
If you use conda, you can install the dependencies with: `conda env create -f environment.yml` .

## Basic Usage
The systems take as input two lists of equal lengths: 
- a list of pitch-classes (obtainable from midi-numbers modulo 12);
- a list of durations in any format (e.g., milliseconds, seconds, quarterLengths, beats).

The output consists of two lists of the same lenghts of the input lists:
- a list of tonal-pitch-classes (e.g. A#, Bb, D, D##);
- a list of key signatures, represented by the number of accidentals (sharps if the number is positive or flats if the number is negative). For example Ab maj is represented with "-4", and D maj with "2".

A complete code example is contained in [notebooks/usage_example.ipynb](notebooks/usage_example.ipynb).

## Reproducibility
*(all paths must be modify for windows systems, by substituting "/" with "\\".)* 

To retrain the model of the paper from the asap dataset:
1. Run `python src/data/download_dataset.py` .
1. Run `python src/data/prepare_dataset.py` .
1. Run `python src/models/train.py`, eventually with ``--device cuda`` for faster training.
1. The model is saved in `models/temp` .


To evaluate the model on the MuseData dataset for the pitch-spelling task:
1. Open a terminal in the project root folder
1. Run `python src/data/download_dataset.py`  
1. Run `python src/data/prepare_dataset.py`
1. Run `python src/models/inference.py`, eventually with ``--device cuda`` for faster computation.

The evaluation for the key-signature estimation task is not available, as the dataset is not public. Contact me for further information.


## Citing
If you use this dataset in any research, please cite the relevant paper:

```
@inproceedings{pkspell,
  title={{PkSpell}: Data-Driven Pitch Spelling and Key Signature Estimation},
  author={Foscarin, Francesco and Audebert, Nicolas  and Fournier S'niehotta, Raphaël},
  booktitle={International Society for Music Information Retrieval Conference {(ISMIR)}},
  year={2021},
  pages={xxx--xxx}
}
```

## License
Licensed under the [MIT License](LICENSE).


