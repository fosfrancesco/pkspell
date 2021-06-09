PAD = "<PAD>"

PITCHES = {
    0: ["C", "B#", "D--"],
    1: ["C#", "B##", "D-"],
    2: ["D", "C##", "E--"],
    3: ["D#", "E-", "F--"],
    4: ["E", "D##", "F-"],
    5: ["F", "E#", "G--"],
    6: ["F#", "E##", "G-"],
    7: ["G", "F##", "A--"],
    8: ["G#", "A-"],
    9: ["A", "G##", "B--"],
    10: ["A#", "B-", "C--"],
    11: ["B", "A##", "C-"],
}

INTERVALS = {
    0: ["P1", "d2", "A7"],
    1: ["m2", "A1"],
    2: ["M2", "d3", "AA1"],
    3: ["m3", "A2"],
    4: ["M3", "d4", "AA2"],
    5: ["P4", "A3"],
    6: ["d5", "A4"],
    7: ["P5", "d6", "AA4"],
    8: ["m6", "A5"],
    9: ["M6", "d7", "AA5"],
    10: ["m7", "A6"],
    11: ["M7", "d1", "AA6"],
}

DIATONIC_PITCHES = ["C", "D", "E", "F", "G", "A", "B"]

KEY_SIGNATURES = list(range(-7, 8))

accepted_pitches = [ii for i in PITCHES.values() for ii in i]

double_acc_pitches = [
    ii for i in PITCHES.values() for ii in i if ii.endswith("##") or ii.endswith("--")
]

ASAP_URL = "https://github.com/fosfrancesco/asap-dataset.git"
