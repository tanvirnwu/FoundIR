import argparse
import time
from pathlib import Path

from profile_utils import count_parameters, format_count, format_macs, format_seconds


def optional_path(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in ("", "none", "null"):
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Profile FoundIR parameters, MACs, and inference time.")
    parser.add_argument("--checkpoint", type=optional_path, default=None, help="Path to model-*.pt checkpoint.")
    parser.add_argument("--checkpoint_folder", type=optional_path, default=None, help="Folder containing model-{milestone}.pt.")
    parser.add_argument("--checkpoint_milestone", type=int, default=None, help="Checkpoint milestone number.")
    parser.add_argument("--dataroot", type=optional_path, default=None, help="Optional dataset root to pick one input image.")
    parser.add_argument("--input_dir", type=str, default="input", help="Input folder name used with --dataroot.")
    parser.add_argument("--gt_dir", type=str, default="gt", help="GT folder name used with --dataroot.")
    parser.add_argument("--input_image", type=optional_path, default=None, help="Optional single input image for timing.")
    parser.add_argument("--height", type=int, default=256, help="Synthetic input height when no image is provided.")
    parser.add_argument("--width", type=int, default=256, help="Synthetic input width when no image is provided.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument("--sampling_timesteps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--show_progress", action="store_true", help="Show diffusion sampling progress bars.")
    return parser.parse_args()


def resolve_checkpoint(args):
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.checkpoint_folder and args.checkpoint_milestone is not None:
        return Path(args.checkpoint_folder) / f"model-{args.checkpoint_milestone}.pt"
    return None


def build_model(torch, UnetRes, ResidualDiffusion, sampling_timesteps):
    num_unet = 1
    objective = "pred_res"
    test_res_or_noise = "res"
    sum_scale = 0.01
    delta_end = 1.4e-3

    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=True,
        objective=objective,
        test_res_or_noise=test_res_or_noise,
    )

    return ResidualDiffusion(
        model,
        image_size=1024,
        timesteps=1000,
        delta_end=delta_end,
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=0.0,
        objective=objective,
        loss_type="l1",
        condition=True,
        sum_scale=sum_scale,
        test_res_or_noise=test_res_or_noise,
    )


def load_checkpoint(torch, diffusion, checkpoint_path, device):
    if checkpoint_path is None:
        print("Checkpoint: not provided; profiling randomly initialized weights.")
        return
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint: {checkpoint_path}")
    if missing:
        print(f"Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {len(unexpected)}")


def first_dataset_input(dataroot, input_dir, gt_dir):
    from data.paired_image_paths import discover_paired_folders, is_image_file

    for input_root, _ in discover_paired_folders(dataroot, input_names=(input_dir,), gt_names=(gt_dir,)):
        images = sorted(path for path in input_root.rglob("*") if path.is_file() and is_image_file(path))
        if images:
            return images[0]
    raise RuntimeError(f"No input images found under {dataroot}")


def load_input_tensor(torch, Image, transforms, args, device):
    image_path = None
    if args.input_image:
        image_path = Path(args.input_image)
    elif args.dataroot:
        image_path = first_dataset_input(Path(args.dataroot), args.input_dir, args.gt_dir)

    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        tensor = transforms.ToTensor()(image).unsqueeze(0)
        print(f"Input source: {image_path}")
    else:
        tensor = torch.rand(args.batch_size, 3, args.height, args.width)
        print(f"Input source: synthetic random tensor ({args.batch_size}, 3, {args.height}, {args.width})")

    return tensor.to(device)


def profile_macs(torch, thop_profile, unet_model, x_input, sampling_timesteps):
    unet_model.eval()
    x_current = 2 * x_input - 1
    x_in = torch.cat((x_current, x_current), dim=1)
    t = torch.full((x_in.shape[0],), 999, device=x_in.device, dtype=torch.long)
    macs, params = thop_profile(unet_model, inputs=(x_in, [t, t]), verbose=False)
    return {
        "one_step_macs": macs,
        "estimated_sampling_macs": macs * sampling_timesteps,
        "thop_params": params,
    }


def synchronize_if_needed(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference_time(torch, diffusion, x_input, repeats, warmup):
    diffusion.eval()
    device = x_input.device

    with torch.no_grad():
        for _ in range(warmup):
            diffusion.sample(x_input, batch_size=x_input.shape[0], last=True)
        synchronize_if_needed(torch, device)

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            diffusion.sample(x_input, batch_size=x_input.shape[0], last=True)
            synchronize_if_needed(torch, device)
            times.append(time.perf_counter() - start)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "repeats": len(times),
    }


def main():
    args = parse_args()

    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        from thop import profile as thop_profile
        import src.model as model_module
        from src.model import ResidualDiffusion, UnetRes, set_seed
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Run this inside the FoundIR conda environment.")

    set_seed(10)
    if not args.show_progress:
        model_module.tqdm = lambda iterable, **kwargs: iterable

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    diffusion = build_model(torch, UnetRes, ResidualDiffusion, args.sampling_timesteps).to(device)
    load_checkpoint(torch, diffusion, resolve_checkpoint(args), device)
    x_input = load_input_tensor(torch, Image, transforms, args, device)

    parameter_counts = count_parameters(diffusion)
    mac_stats = profile_macs(torch, thop_profile, diffusion.model, x_input, args.sampling_timesteps)
    timing = measure_inference_time(torch, diffusion, x_input, repeats=args.repeats, warmup=args.warmup)

    print("")
    print("FoundIR profile")
    print(f"Device: {device}")
    print(f"Input shape: {tuple(x_input.shape)}")
    print(f"Sampling timesteps: {args.sampling_timesteps}")
    print(f"Trainable params: {parameter_counts['trainable']} ({format_count(parameter_counts['trainable'])})")
    print(f"Non-trainable params: {parameter_counts['non_trainable']} ({format_count(parameter_counts['non_trainable'])})")
    print(f"Total params: {parameter_counts['total']} ({format_count(parameter_counts['total'])})")
    print(f"THOP params: {int(mac_stats['thop_params'])} ({format_count(mac_stats['thop_params'])})")
    print(f"One denoising-step MACs: {format_macs(mac_stats['one_step_macs'])}")
    print(f"Estimated full sampling MACs: {format_macs(mac_stats['estimated_sampling_macs'])}")
    print(f"Inference time mean: {format_seconds(timing['mean'])}")
    print(f"Inference time min/max: {format_seconds(timing['min'])} / {format_seconds(timing['max'])}")
    print(f"Timing repeats: {timing['repeats']} after {args.warmup} warmup")


if __name__ == "__main__":
    main()
