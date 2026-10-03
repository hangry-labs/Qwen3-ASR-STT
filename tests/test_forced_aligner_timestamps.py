import itertools
import random
import unittest

import torch

from qwen_asr.inference.qwen3_forced_aligner import _decode_timestamp_bins, _repair_timestamps


def _legacy_repair(values: list[int]) -> list[int]:
    data = values.copy()
    count = len(data)
    lengths = [1] * count
    parents = [-1] * count

    for current in range(1, count):
        for previous in range(current):
            if data[previous] <= data[current] and lengths[previous] + 1 > lengths[current]:
                lengths[current] = lengths[previous] + 1
                parents[current] = previous

    max_length = max(lengths)
    index = lengths.index(max_length)
    stable_indices = []
    while index != -1:
        stable_indices.append(index)
        index = parents[index]
    stable_indices.reverse()

    is_normal = [False] * count
    for index in stable_indices:
        is_normal[index] = True

    result = data.copy()
    block_start = 0
    while block_start < count:
        if is_normal[block_start]:
            block_start += 1
            continue

        block_end = block_start
        while block_end < count and not is_normal[block_end]:
            block_end += 1
        anomaly_count = block_end - block_start

        left_value = next(
            (result[pos] for pos in range(block_start - 1, -1, -1) if is_normal[pos]),
            None,
        )
        right_value = next(
            (result[pos] for pos in range(block_end, count) if is_normal[pos]),
            None,
        )

        if anomaly_count <= 2:
            for pos in range(block_start, block_end):
                if left_value is None:
                    result[pos] = right_value
                elif right_value is None:
                    result[pos] = left_value
                else:
                    result[pos] = (
                        left_value
                        if (pos - (block_start - 1)) <= (block_end - pos)
                        else right_value
                    )
        elif left_value is not None and right_value is not None:
            step = (right_value - left_value) / (anomaly_count + 1)
            for pos in range(block_start, block_end):
                result[pos] = left_value + step * (pos - block_start + 1)
        elif left_value is not None:
            for pos in range(block_start, block_end):
                result[pos] = left_value
        elif right_value is not None:
            for pos in range(block_start, block_end):
                result[pos] = right_value

        block_start = block_end

    return [int(value) for value in result]


def _logits(*rows: dict[int, float], bin_count: int = 5) -> torch.Tensor:
    logits = torch.full((len(rows), bin_count), -100.0)
    for position, scores in enumerate(rows):
        for timestamp_bin, score in scores.items():
            logits[position, timestamp_bin] = score
    return logits


class TimestampRepairTests(unittest.TestCase):
    def test_regression_cases_preserve_stable_anchors_and_interpolation(self):
        cases = [
            ([7], [7]),
            ([0, 100, 100, 100, 250], [0, 100, 100, 100, 250]),
            ([400, 300, 200, 100], [400, 400, 400, 400]),
            ([0, 400, 200, 800], [0, 400, 400, 800]),
            ([4, 1, 3, 2, 5], [1, 1, 3, 3, 5]),
            ([0, 800, 700, 600, 500, 1000], [0, 800, 850, 900, 950, 1000]),
            ([5, 4, 3, 0, 1, 2, 3], [0, 0, 0, 0, 1, 2, 3]),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                original = values.copy()
                self.assertEqual(_repair_timestamps(values), expected)
                self.assertEqual(values, original)

    def test_empty_input_preserves_legacy_error(self):
        with self.assertRaisesRegex(ValueError, "max\\(\\) arg is an empty sequence"):
            _repair_timestamps([])

    def test_matches_legacy_repair_exhaustively(self):
        for length in range(1, 8):
            for values in itertools.product(range(5), repeat=length):
                sample = list(values)
                self.assertEqual(_repair_timestamps(sample), _legacy_repair(sample))

    def test_matches_legacy_repair_for_fixed_seed_random_sequences(self):
        randomizer = random.Random(215)
        for _ in range(2_000):
            sample = [randomizer.randrange(5_000) for _ in range(randomizer.randrange(1, 513))]
            self.assertEqual(_repair_timestamps(sample), _legacy_repair(sample))


class TimestampDecodeTests(unittest.TestCase):
    def test_preserves_legacy_decode_when_every_word_has_duration(self):
        logits = _logits(
            {0: 10.0},
            {2: 10.0},
            {2: 10.0},
            {4: 10.0},
        )

        self.assertEqual(_decode_timestamp_bins(logits), [0, 2, 2, 4])

    def test_uses_best_constrained_path_for_zero_duration_word(self):
        logits = _logits(
            {0: 10.0},
            {1: 10.0},
            {1: 10.0},
            {1: 10.0, 2: 9.0},
            {1: 10.0, 2: 9.0},
            {3: 10.0},
        )

        self.assertEqual(_decode_timestamp_bins(logits), [0, 1, 1, 2, 2, 3])

    def test_can_move_start_earlier_when_better_supported_than_later_end(self):
        logits = _logits(
            {1: 9.0, 2: 10.0},
            {2: 10.0, 3: 1.0},
        )

        self.assertEqual(_decode_timestamp_bins(logits), [1, 2])

    def test_falls_back_when_checkpoint_has_only_one_timestamp_bin(self):
        self.assertEqual(_decode_timestamp_bins(torch.tensor([[1.0], [1.0]])), [0, 0])

    def test_rejects_unpaired_or_malformed_timestamp_logits(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            _decode_timestamp_bins(torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "paired"):
            _decode_timestamp_bins(torch.zeros((3, 4)))


if __name__ == "__main__":
    unittest.main()
