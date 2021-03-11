# Pitch-spelling and Key Signature estimation
Predict the correct pitch spelling and the key signature given a list of MIDI notes.

A sync seq2seq model is used to produce the two information for each note in the input.
Cross-validated on a dataset, this method produces half of the errors of current state of the art approaches.
