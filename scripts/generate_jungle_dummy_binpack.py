import argparse
import random
import struct
from pathlib import Path

BOARD_FILES = 7
BOARD_RANKS = 9
NUM_SQUARES = BOARD_FILES * BOARD_RANKS
NUM_PIECE_TYPES = 8
CAPTURED_SQUARE = 63

ELEPHANT = 0
LION = 1
TIGER = 2
PANTHER = 3
WOLF = 4
DOG = 5
CAT = 6
RAT = 7

RECORD_STRUCT = struct.Struct("<16sBhHbB")


def sq(file: int, rank: int) -> int:
    return rank * BOARD_FILES + file


# A conventional-looking jungle setup in this project's square numbering.
STARTING_SQUARES = [
    sq(0, 2),  # white elephant
    sq(6, 0),  # white lion
    sq(0, 0),  # white tiger
    sq(4, 2),  # white panther
    sq(2, 2),  # white wolf
    sq(5, 1),  # white dog
    sq(1, 1),  # white cat
    sq(6, 2),  # white rat
    sq(6, 6),  # black elephant
    sq(0, 8),  # black lion
    sq(6, 8),  # black tiger
    sq(2, 6),  # black panther
    sq(4, 6),  # black wolf
    sq(1, 7),  # black dog
    sq(5, 7),  # black cat
    sq(0, 6),  # black rat
]


def make_record(index: int, rng: random.Random) -> bytes:
    piece_squares = STARTING_SQUARES.copy()
    occupied = set(piece_squares)

    # Vary material so psqt/layer-stack buckets are exercised.
    capture_count = index % 7
    capture_candidates = list(range(16))
    rng.shuffle(capture_candidates)
    for piece_idx in capture_candidates[:capture_count]:
        occupied.discard(piece_squares[piece_idx])
        piece_squares[piece_idx] = CAPTURED_SQUARE

    # Move several surviving pieces to free squares. These are pseudo positions,
    # not game-legal move sequences; they are intended to validate the loader.
    move_count = 1 + (index % 5)
    for piece_idx in capture_candidates[capture_count : capture_count + move_count]:
        if piece_squares[piece_idx] == CAPTURED_SQUARE:
            continue
        occupied.discard(piece_squares[piece_idx])
        free = [sq for sq in range(NUM_SQUARES) if sq not in occupied]
        new_sq = rng.choice(free)
        piece_squares[piece_idx] = new_sq
        occupied.add(new_sq)

    side_to_move = index % 2
    score = ((index % 41) - 20) * 12
    ply = index
    result = (-1, 0, 1)[index % 3]
    flags = 0

    return RECORD_STRUCT.pack(
        bytes(piece_squares),
        side_to_move,
        score,
        ply,
        result,
        flags,
    )


def write_binpack(path: Path, record_count: int, seed: int) -> None:
    rng = random.Random(seed)
    records = [make_record(i, rng) for i in range(record_count)]
    payload = b"".join(records)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"BINP")
        f.write(struct.pack("<I", len(payload)))
        f.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy jungle NNUE binpack data.")
    parser.add_argument(
        "output",
        nargs="?",
        default=".pgo/jungle_dummy.binpack",
        help="Output .binpack path.",
    )
    parser.add_argument("--records", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    write_binpack(Path(args.output), args.records, args.seed)
    print(
        f"Wrote {args.records} jungle records to {args.output} "
        f"({args.records * RECORD_STRUCT.size} payload bytes)."
    )


if __name__ == "__main__":
    main()
