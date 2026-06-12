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


def _max_dataset_size(value):
    value = str(value).strip()
    if value.lower() in ("inf", "infinity"):
        return float("inf")
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./MillionIRData/Train")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--max_dataset_size", type=_max_dataset_size, default=float("inf"))
    parser.add_argument("--batch_size", type=int, default=80, help="batch size of dataloader")
    parser.add_argument("--load_size", type=int, default=268, help="scale images to this size")
    parser.add_argument("--crop_size", type=int, default=256, help="then crop to this size")
    parser.add_argument("--direction", type=str, default="AtoB", help="AtoB or BtoA")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="crop",
        help="scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]",
    )
    parser.add_argument("--no_flip", action="store_true", help="if specified, do not flip the images for data augmentation")
    parser.add_argument("--meta", type=_optional_path, default=None, help="optional meta-info file for existing FoundIR datasets")
    parser.add_argument("--input_dir", type=_optional_path, default=None, help="input folder name, e.g. input")
    parser.add_argument("--gt_dir", type=_optional_path, default=None, help="ground-truth folder name, e.g. gt")
    parser.add_argument("--bsize", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--results_folder", type=str, default="./ckpt_single_multi")
    parser.add_argument("--train_num_steps", type=int, default=500000)
    parser.add_argument("--save_and_sample_every", type=int, default=1000)
    parser.add_argument("--gradient_accumulate_every", type=int, default=2)
    parser.add_argument("--train_lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=int, default=None, help="resume from model-{resume}.pt in results_folder")
    return parser.parse_args()


def main():
    sys.stdout.flush()
    set_seed(10)

    opt = parse_args()
    train_batch_size = opt.batch_size
    print(train_batch_size)

    dataset_task = "meta_info" if opt.meta is not None else None
    dataset = CombinedDataset(
        opt,
        opt.image_size,
        augment_flip=True,
        equalizeHist=True,
        crop_patch=True,
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
        train_batch_size=train_batch_size,
        num_samples=1,
        train_lr=opt.train_lr,
        train_num_steps=opt.train_num_steps,
        gradient_accumulate_every=opt.gradient_accumulate_every,
        ema_decay=0.995,
        amp=False,
        convert_image_to="RGB",
        results_folder=opt.results_folder,
        condition=True,
        save_and_sample_every=opt.save_and_sample_every,
        num_unet=num_unet,
    )

    if opt.resume is not None:
        trainer.load(opt.resume)
    trainer.train()


if __name__ == "__main__":
    main()
