import ast
import unittest
from pathlib import Path


def find_tuple_assignment(source_path, assignment_name):
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Tuple):
            raise AssertionError(f"{assignment_name} must be a tuple literal")
        return tuple(element.value for element in node.value.elts)
    raise AssertionError(f"{assignment_name} was not found")


class PyiqaMetricConfigTest(unittest.TestCase):
    def test_no_reference_metric_config_includes_piqe(self):
        source_path = Path(__file__).resolve().parents[1] / "src" / "model.py"

        metric_names = find_tuple_assignment(source_path, "PYIQA_BLIND_METRIC_NAMES")

        self.assertEqual(metric_names, ("brisque", "niqe", "piqe"))


if __name__ == "__main__":
    unittest.main()
