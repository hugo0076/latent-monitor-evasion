"""The 60 concepts: 50 for training and 10 held out.

The adapter trains on episodes of `TRAIN_CONCEPTS` only. No loss term uses the monitors of
`TEST_CONCEPTS`, so these test transfer to monitored directions never seen in training.
"""

from __future__ import annotations

TRAIN_CONCEPTS: list[str] = [
    "all_caps",
    "biology",
    "chemistry",
    "comforting",
    "german",
    "history",
    "html",
    "jokey",
    "mathematics",
    "movie_review",
    "side_effects",
    "french",
    "spanish",
    "mandarin_chinese",
    "japanese",
    "arabic",
    "russian",
    "latin",
    "philosophy",
    "literature_focused",
    "linguistics",
    "music_theory",
    "religion_scripture",
    "mythology_folklore",
    "art_history",
    "cooking_recipes",
    "sports_commentary",
    "finance_economics",
    "gardening",
    "automotive_mechanical",
    "travel_writing",
    "fashion",
    "gaming",
    "geology",
    "computer_programming",
    "statistics",
    "json_format",
    "markdown_headers",
    "bullet_lists",
    "emoji_heavy",
    "all_lowercase",
    "formal_academic",
    "aggressive_hostile",
    "melancholy_sad",
    "enthusiastic_excited",
    "nostalgic",
    "condescending_pedantic",
    "military_jargon",
    "corporate_speak",
    "journalistic_headline",
]

TEST_CONCEPTS: list[str] = [
    "deception",
    "physics",
    "astronomy",
    "portuguese",
    "archaeology",
    "sarcastic",
    "sycophancy",
    "evasiveness_noncommittal",
    "imperatives_commands",
    "urgency_panic",
]

ALL_CONCEPTS: list[str] = TRAIN_CONCEPTS + TEST_CONCEPTS


def split_of(concept: str) -> str:
    """Return 'train' or 'test'."""
    if concept in TRAIN_CONCEPTS:
        return "train"
    if concept in TEST_CONCEPTS:
        return "test"
    raise KeyError(f"unknown concept: {concept!r}")


def held_out_concepts() -> list[str]:
    """The 10 concepts the trainer never builds episodes from."""
    return list(TEST_CONCEPTS)


def same_family(a: str, b: str) -> bool:
    """Whether two concepts are the same concept. Roster concepts are distinct, so this is
    equality."""
    if a not in ALL_CONCEPTS or b not in ALL_CONCEPTS:
        raise KeyError(f"unknown concept: {a!r} or {b!r}")
    return a == b


def _check() -> None:
    train, test = set(TRAIN_CONCEPTS), set(TEST_CONCEPTS)
    assert train.isdisjoint(test)
    assert len(ALL_CONCEPTS) == len(set(ALL_CONCEPTS)) == 60
    assert len(TRAIN_CONCEPTS) == 50
    assert len(TEST_CONCEPTS) == 10


_check()
