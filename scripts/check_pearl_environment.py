"""Check the PEARL runtime without downloading weights or running evaluation."""

import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-voc21", action="store_true", help="load one VOC21 validation sample")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_paths import DATA_ROOT, require_large_disk

    require_large_disk()
    print(f"data_root={DATA_ROOT}")
    for package in ("torch", "torchvision", "mmcv", "mmengine", "mmsegmentation", "setuptools"):
        print(f"{package}={metadata.version(package)}")

    import torch
    import mmcv._ext  # noqa: F401: MMSegmentation imports compiled MMCV ops.
    from mmengine.config import Config
    from mmseg.registry import DATASETS
    from mmseg.utils import register_all_modules

    assert torch.cuda.is_available(), "PyTorch cannot access the GPU"
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    assert arch in torch.cuda.get_arch_list(), f"PyTorch lacks a kernel for {arch}"
    value = torch.ones(2, device="cuda").sum().item()
    assert value == 2.0, "CUDA tensor operation failed"
    print(f"gpu={torch.cuda.get_device_name(0)}; arch={arch}; cuda={torch.version.cuda}; cuda_sum={value}")

    register_all_modules()
    import custom_datasets  # noqa: F401: registers project datasets.
    import pearl_ovss  # noqa: F401: registers the PEARL model.

    config = Config.fromfile(str(Path(__file__).resolve().parents[1] / "configs/cfg_voc21.py"))
    print(f"voc21_data_root={config.test_dataloader.dataset.data_root}")
    if args.check_voc21:
        dataset = DATASETS.build(config.test_dataloader.dataset)
        assert len(dataset) == 1449, f"VOC21 validation size is {len(dataset)}, expected 1449"
        sample = dataset[0]
        print(f"voc21_samples={len(dataset)}; first_input_shape={tuple(sample['inputs'].shape)}")
    print("PEARL environment: OK")


if __name__ == "__main__":
    main()
