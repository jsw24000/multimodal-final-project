# Dataset preparation and status

Dataset root: `${PEARL_DATA_ROOT}/datasets`, default `/data/Multimodal-final-project/datasets`. `project_paths.py` supplies every PEARL config with this common root. The project data root is writable. VOC2012 is verified for both VOC20 and VOC21. VOC2010 raw images have been installed, while the validation GT for Context is still missing. ADE20K is verified for its validation split.

| Benchmark | Required validation layout relative to `datasets/` | Expected val pairs | Raw labels and processing | Status |
| --- | --- | ---: | --- | --- |
| VOC20 / VOC21 | `VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation/val.txt}` | 1,449 | indexed PNG 0 background, 1–20 objects, 255 ignore; VOC20 `reduce_zero_label=True`, VOC21 includes background | COMPLETED: 1,449 pairs, 17,125 JPEGs and 2,913 masks in the full source directory; sampled IDs and dimensions valid |
| Context59 / Context60 | `VOCdevkit/VOC2010/{JPEGImages,SegmentationClassContext,ImageSets/SegmentationContext/val.txt}` | 5,105 | converted PNG 0 background, 1–59 classes, 255 ignore; Context59 reduces zero, Context60 retains it | PENDING: 11,321 raw VOC2010 JPEGs installed; Context JSON, masks and split missing |
| ADE20K | `ADEChallengeData2016/{images,annotations}/validation` | 2,000 | indexed PNG raw 0 ignore, 1–150 classes; `LoadAnnotations(reduce_zero_label=True)` | COMPLETED: 2,000 validation image/mask pairs; archive, counts, sampled IDs and dimensions verified |
| COCO-Object | `coco_stuff164k/{images,annotations}/val2017`; masks end `_instanceTrainIds.png` | 5,000 | uses project's `convert_dataset/convert_coco_object.py` on COCO-Stuff maps, 0 background + 80 objects | BLOCKED |
| COCO-Stuff 164k | same val images; masks end `_labelTrainIds.png` | 5,000 | requires official MMSegmentation conversion from indexed stuff+thing maps; 171 evaluated classes | BLOCKED |
| Cityscapes | `cityscapes/leftImg8bit/val`, `gtFine/val` | 500 | separate licensed source; not in initial download scope | PENDING |

VOC20 and VOC21 share the same images, masks, and `val.txt`. Context59 and Context60 similarly share VOC2010 images and a converted Context label directory. VOC2012 is not a substitute for VOC2010. COCO-Object here uses the project's conversion of COCO-Stuff maps; `instances_val2017.json` alone is not its expected segmentation mask. `convert_coco_object.py` asserts the complete train+val mask count (123,287), so validation-only conversion is not yet supported by the untouched official script. Do not fabricate masks or label them official GT.

## Official sources and conversion

- VOC2012: [PASCAL VOC trainval archive](https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), observed size 1,999,639,040 bytes. This is the final official redirect target and responded HTTP 200. The archive includes validation images, PNG masks and split files. `scripts/download_datasets.sh voc2012` can resume an archive on the large disk and validate it before use.
- ADE20K: [MIT ADEChallengeData2016 archive](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), observed size 967,382,037 bytes, HTTP 200. `scripts/download_datasets.sh ade20k` uses the complete official archive; validation images and annotations are checked before use.
- PASCAL Context: [VOC2010 trainval archive](https://thor.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar), observed size 1,345,040,896 bytes and HTTP 200; [MMSegmentation preparation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md). It specifies `trainval_merged.json` and `tools/dataset_converters/pascal_context.py` with `Detail` installed. The linked blob source for the split returned HTTP 409 in this inspection, so acquisition and conversion are pending source verification. The script does not produce an alternative GT.
- COCO-Stuff: [official COCO-Stuff download page](https://github.com/nightrome/cocostuff#downloads), [MMSegmentation preparation guide](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md). Validation needs `val2017.zip` (observed size 815,585,330 bytes by HTTP), indexed PNGs from `stuffthingmaps_trainval2017.zip` (~659 MB), then the MMSegmentation 164k converter. HTTPS to the COCO image host failed certificate verification on this host, so no workaround or download was attempted. Its documented command also downloads train images, but PEARL evaluation config points only to `val2017`; a val-only conversion needs a verified conversion path before execution. COCO-Object's project converter additionally asserts train+val annotation count. The separate `annotations_trainval2017.zip` contains COCO instance JSON and is not interchangeable with these masks.

`verify_datasets.py` checks the split length, file presence, non-empty files, image/mask shape and sample raw IDs. It checks only a sample of file decodes; a successful status is a structural validation, not proof of a published checksum or full image decode. Archive integrity tests are `tar -tf` and `unzip -tq`; no verified published archive digest was found at inspection. Download archives remain under `${PEARL_DATA_ROOT}/downloads` with `.part` files for resumption. Existing unverified dataset directories are preserved and require inspection before retry.

VOC2010 archive provenance: `downloads/VOCtrainval_03-May-2010.tar`, 1,345,040,896 bytes, SHA256 `1af4189cbe44323ab212bff7afbc7d0f55a267cc191eb3aac911037887e5c7d4`. The full tar listing succeeded. `datasets/VOCdevkit/VOC2010/JPEGImages` contains 11,321 JPEGs; this is **not** a validated Context benchmark.

VOC2012 archive provenance: `downloads/VOCtrainval_11-May-2012.tar`, 1,999,639,040 bytes, SHA256 `e14f763270cf193d0b5f74b169f44157a4b0c6efa708f4dd0ff78ee691763bcb`. `verify_datasets.py` passed independently for VOC20 and VOC21. The verification is structural and sampled, not a full decode of all source images.

ADE20K archive provenance: `downloads/ADEChallengeData2016.zip`, 967,382,037 bytes, local SHA256 `7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29`. The official ZIP passed `unzip -tq`; the extracted 44,422 files match the archive file count and contain no empty files. The source contains 20,210 training images/masks and 2,000 validation images/masks. The validation directory passed `verify_datasets.py`; the PEARL/MMSegmentation ADE20K config loaded 2,000 samples, and full PEARL produced one prediction matching its GT shape `(1, 512, 683)`. This is a local archive digest, not a published upstream checksum. The single-image smoke test is not a validation mIoU score.

On 2026-11-13, the background ADE20K download service finished the ZIP and extraction but exited with status 127 because its minimal PATH did not include `python`. The verified staging directory was moved to `datasets/` on the same data filesystem without redownloading or overwriting files. `download_datasets.sh` now uses `python3` or the absolute interpreter supplied by `PEARL_PYTHON`; both ADE20K and VOC2012 completion checks passed with a minimal systemd-like PATH.

## Repeat checks

```bash
export PEARL_DATA_ROOT=/data/Multimodal-final-project
bash scripts/check_environment.sh
bash scripts/download_datasets.sh voc2012
bash scripts/download_datasets.sh ade20k
python3 scripts/verify_datasets.py --dataset all
```

The `context` and `coco_val` script selectors record pending protocols and do not mark data complete. No large download should start if the mount guard fails.
