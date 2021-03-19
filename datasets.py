from torch.utils.data import Dataset


class PSDataset(Dataset):
    def __init__(
        self,
        dict_dataset,
        paths,
        transf_c,
        transf_d,
        transf_k,
        augment_dataset=False,
        sort=False,
        truncate=None,
    ):
        """
        truncate: either None or an integer.
        pad_to_multiple: either None or an integer. If not None, pad the sequence until reaching a multiple of the specified value
        """
        if sort:  # sort (in reverse order) to reduce padding
            dict_dataset = sorted(
                dict_dataset, key=lambda e: (len(e["midi_number"])), reverse=True
            )
        if not augment_dataset:  # remove the transposed pieces
            dict_dataset = [e for e in dict_dataset if e["transposed_of"] == "P1"]
        # consider only pieces in paths
        dict_dataset = [e for e in dict_dataset if e["original_path"] in paths]

        # extract the useful data from dataset
        self.chromatic_sequences = [e["midi_number"] for e in dict_dataset]
        self.diatonic_sequences = [e["pitches"] for e in dict_dataset]
        self.durations = [e["duration"] for e in dict_dataset]
        self.ks = [e["key_signatures"] for e in dict_dataset]

        # the transformations to apply to data
        self.transf_c = transf_c
        self.transf_d = transf_d
        self.transf_ks = transf_k

        # truncate and pad
        self.truncate = truncate

    def __len__(self):
        return len(self.chromatic_sequences)

    def __getitem__(self, idx):
        chromatic_seq = self.chromatic_sequences[idx]
        diatonic_seq = self.diatonic_sequences[idx]
        duration_seq = self.durations[idx]
        ks_seq = self.ks[idx]

        # transform
        chromatic_seq = self.transf_c(chromatic_seq, duration_seq)
        diatonic_seq = self.transf_d(diatonic_seq)
        ks_seq = self.transf_ks(ks_seq)

        if not self.truncate is None:
            if len(diatonic_seq) > self.truncate:
                chromatic_seq = chromatic_seq[0 : self.truncate]
                diatonic_seq = diatonic_seq[0 : self.truncate]
                ks_seq = ks_seq[0 : self.truncate]

        # sanity check
        assert len(chromatic_seq) == len(diatonic_seq) == len(ks_seq)
        seq_len = len(diatonic_seq)

        return chromatic_seq, diatonic_seq, ks_seq, seq_len