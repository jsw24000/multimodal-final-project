# Dataset preparation and status

Dataset root: `${PEARL_DATA_ROOT}/datasets`, default `/data/Multimodal-final-project/datasets`. `project_paths.py` supplies every PEARL config with this common root. Every dataset below is **BLOCKED** at zero downloaded files because the current user cannot create `/data/Multimodal-final-project`. No dataset is Ready.

| Benchmark | Required validation layout relative to `datasets/` | Expected val pairs | Raw labels and processing | Status |
| --- | --- | ---: | --- | --- |
| VOC20 / VOC21 | `VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation/val.txt}` | 1,449 | indexed PNG 0 background, 1–20 objects, 255 ignore; VOC20 `reduce_zero_label=True`, VOC21 includes background | BLOCKED |
| Context59 / Context60 | `VOCdevkit/VOC2010/{JPEGImages,SegmentationClassContext,ImageSets/SegmentationContext/val.txt}` | 5,105 | converted PNG 0 background, 1–59 classes, 255 ignore; Context59 reduces zero, Context60 retains it | BLOCKED |
| ADE20K | `ADEChallengeData2016/{images,annotations}/validation` | 2,000 | indexed PNG raw 0 ignore, 1–150 classes; `LoadAnnotations(reduce_zero_label=True)` | BLOCKED |
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

## Resume after storage permission

```bash
export PEARL_DATA_ROOT=/data/Multimodal-final-project
bash scripts/check_environment.sh
bash scripts/download_datasets.sh voc2012
bash scripts/download_datasets.sh ade20k
python scripts/verify_datasets.py --dataset all
```

The `context` and `coco_val` script selectors record pending protocols and do not mark data complete. No large download should start if the mount guard fails.
