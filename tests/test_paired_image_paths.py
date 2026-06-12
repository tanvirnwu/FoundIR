import tempfile
import unittest
import importlib.util
from pathlib import Path


def load_helper():
    helper_path = Path(__file__).resolve().parents[1] / "data" / "paired_image_paths.py"
    spec = importlib.util.spec_from_file_location("paired_image_paths", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.paired_paths_from_folders


class PairedImagePathsTest(unittest.TestCase):
    def test_pairs_flat_input_gt_by_relative_name(self):
        paired_paths_from_folders = load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            gt_dir = root / "gt"
            input_dir.mkdir()
            gt_dir.mkdir()
            (input_dir / "0001.png").touch()
            (gt_dir / "0001.png").touch()
            (input_dir / "0002.jpg").touch()
            (gt_dir / "0002.jpg").touch()

            pairs = paired_paths_from_folders(root, input_names=("input",), gt_names=("gt",))

            self.assertEqual(
                pairs,
                [
                    {"adap_path": str(input_dir / "0001.png"), "gt_path": str(gt_dir / "0001.png")},
                    {"adap_path": str(input_dir / "0002.jpg"), "gt_path": str(gt_dir / "0002.jpg")},
                ],
            )

    def test_discovers_nested_test_categories(self):
        paired_paths_from_folders = load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for category in ("haze", "rain"):
                input_dir = root / category / "input"
                gt_dir = root / category / "gt"
                input_dir.mkdir(parents=True)
                gt_dir.mkdir(parents=True)
                (input_dir / "sample.png").touch()
                (gt_dir / "sample.png").touch()

            pairs = paired_paths_from_folders(root, input_names=("input",), gt_names=("gt",))

            self.assertEqual(len(pairs), 2)
            self.assertTrue(pairs[0]["adap_path"].endswith(str(Path("haze") / "input" / "sample.png")))
            self.assertTrue(pairs[1]["adap_path"].endswith(str(Path("rain") / "input" / "sample.png")))

    def test_raises_when_gt_is_missing(self):
        paired_paths_from_folders = load_helper()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            gt_dir = root / "gt"
            input_dir.mkdir()
            gt_dir.mkdir()
            (input_dir / "missing.png").touch()

            with self.assertRaises(FileNotFoundError):
                paired_paths_from_folders(root, input_names=("input",), gt_names=("gt",))


if __name__ == "__main__":
    unittest.main()
