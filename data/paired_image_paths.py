from pathlib import Path


IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".tif",
    ".tiff",
}


def is_image_file(path):
    return Path(path).suffix.lower() in IMG_EXTENSIONS


def _limit_paths(paths, max_dataset_size):
    if max_dataset_size == float("inf"):
        return paths
    return paths[: min(int(max_dataset_size), len(paths))]


def _iter_images(root):
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.is_file() and is_image_file(path))


def _relative_key(path, root):
    return Path(path).relative_to(root).as_posix().lower()


def _relative_stem_key(path, root):
    return Path(path).relative_to(root).with_suffix("").as_posix().lower()


def _build_gt_index(gt_paths, gt_root):
    by_relative = {}
    by_relative_stem = {}
    by_name_stem = {}

    for gt_path in gt_paths:
        by_relative[_relative_key(gt_path, gt_root)] = gt_path
        by_relative_stem[_relative_stem_key(gt_path, gt_root)] = gt_path
        by_name_stem.setdefault(gt_path.stem.lower(), []).append(gt_path)

    return by_relative, by_relative_stem, by_name_stem


def _match_gt(input_path, input_root, gt_root, gt_index):
    by_relative, by_relative_stem, by_name_stem = gt_index

    relative_key = _relative_key(input_path, input_root)
    if relative_key in by_relative:
        return by_relative[relative_key]

    relative_stem_key = _relative_stem_key(input_path, input_root)
    if relative_stem_key in by_relative_stem:
        return by_relative_stem[relative_stem_key]

    candidates = by_name_stem.get(input_path.stem.lower(), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous GT match for {input_path}: found {len(candidates)} files named {input_path.stem}.*"
        )

    expected = Path(gt_root) / Path(input_path).relative_to(input_root)
    raise FileNotFoundError(f"Missing GT image for {input_path}. Expected {expected}")


def _folder_pairs_from_parent(parent, input_names, gt_names):
    pairs = []
    for input_name in input_names:
        input_root = parent / input_name
        if not input_root.is_dir():
            continue
        for gt_name in gt_names:
            gt_root = parent / gt_name
            if gt_root.is_dir():
                pairs.append((input_root, gt_root))
                break
    return pairs


def discover_paired_folders(dataroot, input_names=("input",), gt_names=("gt",)):
    dataroot = Path(dataroot)
    if not dataroot.is_dir():
        raise FileNotFoundError(f"{dataroot} is not a valid directory")

    direct_pairs = _folder_pairs_from_parent(dataroot, input_names, gt_names)
    if direct_pairs:
        return direct_pairs

    pairs = []
    seen = set()
    for parent in sorted(path for path in dataroot.rglob("*") if path.is_dir()):
        for input_root, gt_root in _folder_pairs_from_parent(parent, input_names, gt_names):
            key = (input_root.resolve(), gt_root.resolve())
            if key not in seen:
                pairs.append((input_root, gt_root))
                seen.add(key)

    if not pairs:
        expected_inputs = ", ".join(input_names)
        expected_gts = ", ".join(gt_names)
        raise FileNotFoundError(
            f"No paired folders found under {dataroot}. Expected siblings like "
            f"one of [{expected_inputs}] and one of [{expected_gts}]."
        )
    return pairs


def paired_paths_from_folders(
    dataroot,
    input_names=("input",),
    gt_names=("gt",),
    max_dataset_size=float("inf"),
    input_key="adap",
    gt_key="gt",
):
    pairs = []
    for input_root, gt_root in discover_paired_folders(dataroot, input_names, gt_names):
        input_paths = _iter_images(input_root)
        gt_paths = _iter_images(gt_root)
        gt_index = _build_gt_index(gt_paths, gt_root)

        for input_path in input_paths:
            gt_path = _match_gt(input_path, input_root, gt_root, gt_index)
            pairs.append(
                {
                    f"{input_key}_path": str(input_path),
                    f"{gt_key}_path": str(gt_path),
                }
            )

    if not pairs:
        raise RuntimeError(f"Found 0 paired images under {dataroot}")
    return _limit_paths(pairs, max_dataset_size)
