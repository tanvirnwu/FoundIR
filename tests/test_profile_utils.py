import importlib.util
import unittest
from pathlib import Path


def load_utils():
    helper_path = Path(__file__).resolve().parents[1] / "profile_utils.py"
    spec = importlib.util.spec_from_file_location("profile_utils", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeParam:
    def __init__(self, count, requires_grad):
        self._count = count
        self.requires_grad = requires_grad

    def numel(self):
        return self._count


class FakeModel:
    def parameters(self):
        return [
            FakeParam(10, True),
            FakeParam(20, False),
            FakeParam(30, True),
        ]


class ProfileUtilsTest(unittest.TestCase):
    def test_count_parameters_splits_trainable_and_non_trainable(self):
        utils = load_utils()

        counts = utils.count_parameters(FakeModel())

        self.assertEqual(counts["trainable"], 40)
        self.assertEqual(counts["non_trainable"], 20)
        self.assertEqual(counts["total"], 60)

    def test_format_count_uses_millions(self):
        utils = load_utils()

        self.assertEqual(utils.format_count(12_345_678), "12.346 M")


if __name__ == "__main__":
    unittest.main()
