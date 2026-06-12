import argparse
import sys

from data.combined_dataset import CombinedDataset
from src.model import ResidualDiffusion, Trainer, UnetRes, set_seed


def _optional_path(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in ("", "none", "null"):
        return None
    return value


def _optional_crop_phase(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in ("", "none", "null"):
        return None
    return value


def _max_dataset_size(value):
    value = str(value).strip()
    if value.lower() in ("inf", "infinity"):
        return float("inf")
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./MillionIRData/Test")
    parser.add_argument("--phase", type=str, default="test")
    parser.add_argument("--max_dataset_size", type=_max_dataset_size, default=float("inf"))
    parser.add_argument("--load_size", type=int, default=256, help="scale images to this size")
    parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
    parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="none",
        help="scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]",
    )
    parser.add_argument("--no_flip", type=bool, default=True, help="if specified, do not flip the images for data augmentation")
    parser.add_argument("--meta", type=_optional_path, default=None, help="optional meta-info file for existing FoundIR datasets")
    parser.add_argument("--input_dir", type=_optional_path, default=None, help="input folder name, e.g. input")
    parser.add_argument("--gt_dir", type=_optional_path, default=None, help="ground-truth folder name, e.g. gt")
    parser.add_argument("--bsize", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--sampling_timesteps", type=int, default=4)
    parser.add_argument("--checkpoint_folder", type=str, default="premodel")
    parser.add_argument("--checkpoint_milestone", type=int, default=2000)
    parser.add_argument("--results_folder", type=str, default="./results")
    parser.add_argument("--test_crop_phase", type=_optional_crop_phase, default="im2overlap")
    parser.add_argument("--test_crop_size", type=int, default=1024)
    parser.add_argument("--test_crop_stride", type=int, default=512)
    return parser.parse_args()


def main():
    sys.stdout.flush()
    set_seed(10)

    opt = parse_args()
    dataset_task = "meta_info" if opt.meta is not None else None
    dataset = CombinedDataset(
        opt,
        opt.image_size,
        augment_flip=False,
        equalizeHist=True,
        crop_patch=False,
        generation=False,
        task=dataset_task,
    )

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

    diffusion = ResidualDiffusion(
        model,
        image_size=opt.image_size,
        timesteps=1000,
        delta_end=delta_end,
        sampling_timesteps=opt.sampling_timesteps,
        ddim_sampling_eta=0.0,
        objective=objective,
        loss_type="l1",
        condition=True,
        sum_scale=sum_scale,
        test_res_or_noise=test_res_or_noise,
    )

    trainer = Trainer(
        diffusion,
        dataset,
        opt,
        train_batch_size=1,
        num_samples=1,
        train_lr=2e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=opt.checkpoint_folder,
        condition=True,
        save_and_sample_every=1000,
        num_unet=num_unet,
    )

    if not trainer.accelerator.is_local_main_process:
        return

    trainer.load(opt.checkpoint_milestone)
    trainer.set_results_folder(opt.results_folder)
    trainer.test(
        last=True,
        crop_phase=opt.test_crop_phase,
        crop_size=opt.test_crop_size,
        crop_stride=opt.test_crop_stride,
    )


if __name__ == "__main__":
    main()
