import argparse
import os
import sys
import time
import torch
import yaml
import numpy as np

from pathlib import Path
from utils.general import (
    LOGGER, check_requirements, check_dataset, check_suffix,
    check_img_size, increment_path, print_args, colorstr
)
from utils.torch_utils import select_device
from utils.dataloaders import create_dataloader
from utils.callbacks import Callbacks
from models.experimental import attempt_load
import val as validate

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def run_test(opt, callbacks):
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    data_dict = check_dataset(opt.data)
    test_path = data_dict["test"]
    nc = 1 if opt.single_cls else int(data_dict["nc"])
    names = {0: data_dict["names"][0]} if opt.single_cls and len(data_dict["names"]) != 1 else data_dict["names"]

    check_suffix(opt.weights, ".pt")
    device = select_device(opt.device)
    model = attempt_load(opt.weights, device).half()
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    test_loader = create_dataloader(
        test_path,
        imgsz,
        opt.batch_size * 2,
        gs,
        opt.single_cls,
        hyp=None,
        cache=None,
        rect=False,
        rank=-1,
        workers=opt.workers,
        pad=0.5,
        prefix=colorstr("test: "),
        rgbt_input=opt.rgbt,
    )[0]

    LOGGER.info(f"\nRunning test on {opt.weights}...")
    validate.run(
        data_dict,
        batch_size=opt.batch_size * 2,
        imgsz=imgsz,
        model=model,
        task="test",
        iou_thres=0.60,
        single_cls=opt.single_cls,
        dataloader=test_loader,
        save_dir=save_dir,
        save_json=True,
        verbose=True,
        plots=False,
        callbacks=callbacks,
        epoch=0
    )
    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--cfg", type=str, default="", help="Path to model config .yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset .yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="CUDA device")
    parser.add_argument("--workers", type=int, default=16, help="Number of dataloader workers")
    parser.add_argument("--single-cls", action="store_true", help="Train as single-class dataset")
    parser.add_argument("--rgbt", action="store_true", help="Use RGB-T multispectral image pair")
    parser.add_argument("--project", default=ROOT / "runs/test", help="Save directory")
    parser.add_argument("--name", default="exp", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing run directory")
    parser.add_argument("--entity", default=None, help="WandB entity")
    return parser.parse_args()

def main():
    opt = parse_opt()
    print_args(vars(opt))
    check_requirements(ROOT / "requirements.txt")
    callbacks = Callbacks()
    run_test(opt, callbacks)

if __name__ == "__main__":
    main()

'''
python test_only.py \
  --imgsz 640 \
  --batch-size 16 \
  --data data/kaist-rgbt.yaml \
  --cfg models/yolov5n_kaist-rgbt.yaml \
  --weights runs/train/yolov5n-rgbt_loss/weights/best.pt \
  --workers 16 \
  --name test_prediction \
  --entity $WANDB_ENTITY \
  --rgbt \
  --device 1
'''