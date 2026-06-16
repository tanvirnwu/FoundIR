import importlib.util
import unittest
from pathlib import Path


def load_accumulator():
    helper_path = Path(__file__).resolve().parents[1] / "metrics" / "running_average.py"
    spec = importlib.util.spec_from_file_location("running_average", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MetricAccumulator


class MetricAccumulatorTest(unittest.TestCase):
    def test_averages_metrics(self):
        accumulator = load_accumulator()(names=("psnr", "ssim", "lpips", "brisque", "niqe"))

        accumulator.update(psnr=20.0, ssim=0.80, lpips=0.20, brisque=15.0, niqe=4.0)
        accumulator.update(psnr=30.0, ssim=0.90, lpips=0.10, brisque=25.0, niqe=6.0)

        self.assertEqual(accumulator.count, 2)
        averages = accumulator.averages()
        self.assertEqual(averages["psnr"], 25.0)
        self.assertAlmostEqual(averages["ssim"], 0.85)
        self.assertAlmostEqual(averages["lpips"], 0.15)
        self.assertAlmostEqual(averages["brisque"], 20.0)
        self.assertAlmostEqual(averages["niqe"], 5.0)

    def test_empty_average_is_none(self):
        accumulator = load_accumulator()()

        self.assertIsNone(accumulator.averages())


if __name__ == "__main__":
    unittest.main()
