import itertools
import random
import unittest

from qwen_asr.inference.qwen3_forced_aligner import _repair_timestamps


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


if __name__ == "__main__":
    unittest.main()
