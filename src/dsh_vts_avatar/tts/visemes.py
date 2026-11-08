"""Phoneme -> coarse viseme mapping.

Piper voices use eSpeak IPA (Chinese voices may expose pinyin-like phones).
The mapping is deliberately conservative: it only needs a dominant mouth
shape, base openness and a small form offset. Exact tuning belongs in a model
profile, not in this table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisemeSpec:
    viseme: str  # A/I/U/E/O or "" for silence
    kind: str  # vowel | consonant | silence
    open: float
    form: float
    vowel_gain: float = 0.0


SILENCE = VisemeSpec("", "silence", 0.0, 0.0, 0.0)

VOWELS: dict[str, VisemeSpec] = {
    # A
    "a": VisemeSpec("A", "vowel", 0.98, 0.00, 0.90),
    "ɑ": VisemeSpec("A", "vowel", 0.98, 0.00, 0.90),
    "æ": VisemeSpec("A", "vowel", 0.90, 0.05, 0.85),
    "ʌ": VisemeSpec("A", "vowel", 0.88, 0.00, 0.80),
    "ɐ": VisemeSpec("A", "vowel", 0.88, 0.00, 0.82),
    # I
    "i": VisemeSpec("I", "vowel", 0.55, 0.38, 0.82),
    "ɪ": VisemeSpec("I", "vowel", 0.52, 0.35, 0.78),
    "y": VisemeSpec("I", "vowel", 0.52, 0.30, 0.80),
    "ʏ": VisemeSpec("I", "vowel", 0.52, 0.30, 0.78),
    # U
    "u": VisemeSpec("U", "vowel", 0.58, -0.48, 0.82),
    "ʊ": VisemeSpec("U", "vowel", 0.54, -0.42, 0.78),
    "ʉ": VisemeSpec("U", "vowel", 0.55, -0.40, 0.80),
    # E
    "e": VisemeSpec("E", "vowel", 0.78, 0.16, 0.82),
    "ɛ": VisemeSpec("E", "vowel", 0.80, 0.12, 0.80),
    "ə": VisemeSpec("E", "vowel", 0.62, 0.05, 0.75),
    "ɤ": VisemeSpec("E", "vowel", 0.78, 0.05, 0.82),
    "ɜ": VisemeSpec("E", "vowel", 0.72, 0.05, 0.78),
    # O
    "o": VisemeSpec("O", "vowel", 0.92, -0.32, 0.86),
    "ɔ": VisemeSpec("O", "vowel", 0.94, -0.35, 0.86),
    "ɒ": VisemeSpec("O", "vowel", 0.96, -0.25, 0.86),
}

# Coarse consonant classes. `open` is the jaw/lip opening while the phone is
# active; bilabials are intentionally near-closed because RMS is meaningless
# during their silence.
CONSONANTS: dict[str, VisemeSpec] = {}
for _p in ("p", "b", "m"):
    CONSONANTS[_p] = VisemeSpec("", "consonant", 0.035, 0.00, 0.0)
for _p in ("f", "v"):
    CONSONANTS[_p] = VisemeSpec("", "consonant", 0.10, 0.05, 0.0)
for _p in ("s", "z", "ʃ", "ʒ", "θ", "ð", "ɕ", "ʑ", "ʂ", "ʐ", "χ", "x", "h", "ɸ", "β"):
    CONSONANTS[_p] = VisemeSpec("", "consonant", 0.13 if _p not in ("x", "χ", "h") else 0.20, 0.18 if _p in ("s", "z", "ʃ", "ʒ", "ɕ", "ʑ", "ʂ", "ʐ") else 0.0, 0.0)
for _p in ("t", "d", "n", "l", "r", "ɾ", "k", "g", "h", "x", "ŋ", "ɲ", "ɳ", "ɹ", "ɫ", "ʎ", "ʔ"):
    CONSONANTS.setdefault(_p, VisemeSpec("", "consonant", 0.28, 0.00, 0.0))
for _p in ("ts", "dz", "tʃ", "dʒ", "ʈʂ", "ɖʐ", "c", "ɟ", "q"):
    CONSONANTS[_p] = VisemeSpec("", "consonant", 0.16, 0.10, 0.0)
CONSONANTS["w"] = VisemeSpec("U", "consonant", 0.24, -0.45, 0.35)
CONSONANTS["j"] = VisemeSpec("I", "consonant", 0.20, 0.38, 0.35)

# Chinese pinyin finals that eSpeak may leave as Latin text.
PINYIN_VOWELS: dict[str, VisemeSpec] = {
    "a": VOWELS["a"],
    "o": VOWELS["o"],
    "e": VOWELS["e"],
    "i": VOWELS["i"],
    "u": VOWELS["u"],
    "v": VOWELS["y"],
    "ü": VOWELS["y"],
    "ai": VisemeSpec("A", "vowel", 0.72, 0.15, 0.82),
    "ei": VisemeSpec("E", "vowel", 0.46, 0.25, 0.80),
    "ao": VisemeSpec("A", "vowel", 0.72, -0.10, 0.82),
    "ou": VisemeSpec("O", "vowel", 0.50, -0.20, 0.80),
    "an": VOWELS["a"],
    "en": VOWELS["ə"],
    "ang": VOWELS["ɑ"],
    "eng": VOWELS["ə"],
    "ong": VOWELS["o"],
}

_DROP = set("ˈˌˑː‿|‖!,.?;:，。！？；：、 \t\n\r()[]{}'\"")
_SILENCE_TOKENS = {"", "_", "sil", "pau", "sp", "space", "0", "pad"}


def normalize_phone(phone: str) -> str:
    text = phone.strip().lower()
    for char in _DROP:
        text = text.replace(char, "")
    return text


def map_phoneme(phone: str, language: str = "zh") -> VisemeSpec:
    raw = normalize_phone(phone)
    if raw in _SILENCE_TOKENS:
        return SILENCE
    if raw in VOWELS:
        return VOWELS[raw]
    if raw in PINYIN_VOWELS:
        return PINYIN_VOWELS[raw]
    if raw in CONSONANTS:
        return CONSONANTS[raw]

    # A phone can be a sequence (affricate/diphthong). Pick the most open
    # vowel-like component, or the first known consonant.
    for char in raw:
        if char in VOWELS:
            base = VOWELS[char]
            return VisemeSpec(base.viseme, base.kind, base.open, base.form, base.vowel_gain)
    for char in raw:
        if char in CONSONANTS:
            return CONSONANTS[char]
    if len(raw) > 1:
        # Unknown but non-empty: keep the mouth slightly active rather than
        # snapping shut for the whole phone.
        return VisemeSpec("", "consonant", 0.18, 0.0, 0.0)
    return SILENCE


def vowel_name_for(phone: str, language: str = "zh") -> str:
    return map_phoneme(phone, language).viseme
