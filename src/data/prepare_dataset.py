def score2midi_numbers(score):
    return [p.midi % 12 for n in score.flat.notes for p in n.pitches]


def score2pitches(score):
    return [p.name for n in score.flat.notes for p in n.pitches]


def score2pitches_meredith(score):
    # Return the David Meredith style of pitches (http://www.titanmusic.com/data.php)
    return [
        p.nameWithOctave.replace("s", "#").replace("f", "-")
        if ("#" in p.nameWithOctave) or ("-" in p.nameWithOctave)
        else p.nameWithOctave.replace("s", "#").replace("f", "-")[:-1]
        + "n"
        + p.nameWithOctave.replace("s", "#").replace("f", "-")[-1]
        for n in score.flat.notes
        for p in n.pitches
    ]


def score2onsets(score):
    return [n.offset for n in score.flat.notes for p in n.pitches]


def score2durations(score):
    return [n.duration.quarterLength for n in score.flat.notes for p in n.pitches]


def score2voice(score):
    return [
        int(str(n.getContextByClass("Voice"))[-2])
        if not n.getContextByClass("Voice") is None
        else 1
        for n in score.flat.notes
        for p in n.pitches
    ]


def score2ks(score):
    """Return one ks for each pitch for each note"""
    temp_ks = None
    out = []
    for event in score.flat:
        if isinstance(event, m21.key.KeySignature):
            #             print("Found a ks")
            temp_ks = event.sharps
        elif isinstance(event, m21.note.NotRest):
            for pitch in event.pitches:
                #                 print("FOund a note")
                out.append(temp_ks)
    return out


accepted_intervals = [ii for i in INTERVALS.values() for ii in i]
print([e for e in enumerate(accepted_intervals)])


def transp_score(score):
    """ For each input return len(accepted_intervals) transposed scores"""
    return [score.transpose(interval) for interval in accepted_intervals]


def transp_note_list(note_list):
    """ For each input return len(accepted_intervals) transposed list of notes"""
    return [
        [n.transpose(interval) for n in note_list] for interval in accepted_intervals
    ]


# def acc_simple_enough(score, accepted_ratio=0.2):
#     pitches = score2pitches(score)
#     double_acc = sum(el in double_acc_pitches for el in pitches)
#     if double_acc / len(pitches) < accepted_ratio:
#         return True
#     else:
#         return False
