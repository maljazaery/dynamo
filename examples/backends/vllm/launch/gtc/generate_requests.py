"""Generate a .jsonl benchmark file for aiperf with single-turn requests.

Each request contains user text (~4,300 tokens by default) and N images (512x512 PNG).
A configurable fraction of image slots reuse a previously seen image (see README.md).

Images are saved as PNG files in an output directory; the JSONL references
them by path. System prompt length is NOT in the JSONL — pass it via
aiperf's --shared-system-prompt-length flag.

Usage:
    python generate_requests.py
    python generate_requests.py -n 200 --cache-hit-rate 0.3
    python generate_requests.py -n 100 --images-per-request 20 --cache-hit-rate 0.27
    python generate_requests.py --user-text-tokens 4000
    python generate_requests.py -o out.jsonl --image-dir /tmp/bench_images

Example aiperf invocation:
    aiperf profile \\
      --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \\
      --input-file examples/backends/vllm/launch/gtc/requests.jsonl \\
      --custom-dataset-type single_turn \\
      --shared-system-prompt-length 8600 \\
      --extra-inputs "max_tokens:500" \\
      --extra-inputs "min_tokens:500" \\
      --extra-inputs "ignore_eos:true"

  The file contains the actual request content (user text + image paths), not token
  counts. Input length is computed from that content plus --shared-system-prompt-length.
  Do not pass --isl: it applies only to synthetic data generation, not to
  --input-file single_turn.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_IMAGES_PER_REQUEST = 3
IMAGE_SIZE = (512, 512)
SEED = 42
USER_TEXT_TOKENS = 4_300

# Common English words that each tokenize to a single BPE token on most LLMs.
_ENGLISH_VOCAB = [
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "I",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "great",
    "world",
    "still",
    "own",
    "find",
    "here",
    "thing",
    "many",
    "long",
    "hand",
    "high",
    "keep",
    "place",
    "start",
    "might",
    "old",
    "home",
    "big",
    "end",
    "while",
    "last",
    "turn",
    "ask",
    "need",
    "too",
    "feel",
    "seem",
    "call",
    "head",
    "put",
    "lot",
    "run",
    "every",
    "play",
    "small",
    "set",
    "live",
    "try",
    "tell",
    "few",
    "part",
    "change",
    "help",
    "show",
    "house",
    "both",
    "side",
    "point",
    "such",
    "name",
    "each",
    "right",
    "move",
    "must",
    "real",
    "left",
    "same",
    "much",
    "open",
    "near",
    "line",
    "build",
    "power",
    "water",
    "city",
    "tree",
    "earth",
    "plan",
    "food",
    "dark",
    "cold",
    "sure",
    "car",
    "face",
    "nice",
    "state",
    "fact",
    "night",
    "hard",
    "read",
    "idea",
    "stand",
    "class",
    "body",
    "book",
    "word",
    "best",
    "done",
    "case",
    "four",
    "fire",
    "front",
    "rest",
    "game",
    "war",
    "air",
    "eye",
    "true",
    "top",
    "area",
    "boy",
    "girl",
    "color",
    "oil",
    "song",
    "note",
    "low",
    "bed",
]


def _generate_filler(rng: random.Random, num_tokens: int) -> str:
    """Return ~num_tokens worth of space-separated common English words."""
    return " ".join(rng.choice(_ENGLISH_VOCAB) for _ in range(num_tokens))


def generate_image_pool(
    np_rng: np.random.Generator,
    py_rng: random.Random,
    total_slots: int,
    duplicate_prob: float,
    image_dir: Path,
) -> list[str]:
    """Generate unique PNG files and return *total_slots* file paths with ~duplicate_prob reuse.

    For each slot: with probability *duplicate_prob* pick a previously
    generated image (if any), otherwise create a fresh random 512x512 PNG.
    Returns a list of absolute file paths (strings).
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    unique_paths: list[str] = []
    slot_paths: list[str] = []

    for _ in range(total_slots):
        if unique_paths and py_rng.random() < duplicate_prob:
            slot_paths.append(py_rng.choice(unique_paths))
        else:
            idx = len(unique_paths)
            path = image_dir / f"img_{idx:04d}.png"
            pixels = np_rng.integers(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            abs_path = str(path.resolve())
            unique_paths.append(abs_path)
            slot_paths.append(abs_path)

    num_unique = len(set(slot_paths))
    print(
        f"Generated {total_slots} image slots: "
        f"{num_unique} unique images saved to {image_dir}, "
        f"{total_slots - num_unique} duplicate references "
        f"({(total_slots - num_unique) / total_slots:.1%} reuse)"
    )
    return slot_paths


def build_request(user_text: str, image_paths: list[str]) -> dict:
    """Build an aiperf SingleTurn JSONL entry."""
    return {"text": user_text, "images": image_paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=500,
        help="Number of requests to generate (default: 100)",
    )
    parser.add_argument(
        "--cache-hit-rate",
        type=float,
        default=0.20,
        help="Probability that an image slot reuses a previous image (default: 0.20)",
    )
    parser.add_argument(
        "--images-per-request",
        type=int,
        default=DEFAULT_IMAGES_PER_REQUEST,
        help=f"Number of images per request (default: {DEFAULT_IMAGES_PER_REQUEST})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .jsonl path (default: {n}req_{img}img_{pct}pct_{word}word.jsonl in script directory, e.g. 100req_20img_27pct_4000word.jsonl)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/tmp/bench_images"),
        help="Directory to save generated PNG images (default: /tmp/bench_images)",
    )
    parser.add_argument(
        "--user-text-tokens",
        type=int,
        default=USER_TEXT_TOKENS,
        help=f"Target user text tokens per request (default: {USER_TEXT_TOKENS}). --isl is an alias.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_requests: int = args.num_requests
    images_per_request: int = args.images_per_request
    cache_hit_rate: float = args.cache_hit_rate

    output_path = args.output
    if output_path is None:
        pct = int(round(cache_hit_rate * 100))
        output_path = (
            Path(__file__).parent
            / f"{num_requests}req_{images_per_request}img_{pct}pct_{args.user_text_tokens}word.jsonl"
        )

    np_rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)
    total_slots = num_requests * images_per_request

    slot_paths = generate_image_pool(
        np_rng, py_rng, total_slots, cache_hit_rate, args.image_dir
    )

    with open(output_path, "w") as f:
        for i in range(num_requests):
            user_text = _generate_filler(py_rng, args.user_text_tokens)
            start = i * images_per_request
            images = slot_paths[start : start + images_per_request]
            line = json.dumps(build_request(user_text, images), separators=(",", ":"))
            f.write(line + "\n")

    print(f"Wrote {num_requests} requests to {output_path}")


if __name__ == "__main__":
    main()
