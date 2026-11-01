# 00 - data preparation. Run once. Each step writes data/<name>/<split>.jsonl plus a
# *_preview.jpg with the GT boxes drawn - OPEN THE PREVIEWS and confirm the boxes are right.
$ErrorActionPreference = "Stop"

# RefCOCO family (lmms-lab uploads: columns question_id/image/question/answer(list)/bbox(xywh px)).
# --limit counts HF rows (regions); each row carries ~3 expressions.
# If the repo id 404s, try the lmms-lab-encoder/ organisation instead of lmms-lab/.
$ROWS = 1500
$SETS = @(
  @("hf:lmms-lab/RefCOCO",     "val",   "data/refcoco/val"),
  @("hf:lmms-lab/RefCOCO",     "testA", "data/refcoco/testA"),
  @("hf:lmms-lab/RefCOCO",     "testB", "data/refcoco/testB"),
  @("hf:lmms-lab/RefCOCOplus", "val",   "data/refcoco_plus/val"),
  @("hf:lmms-lab/RefCOCOplus", "testA", "data/refcoco_plus/testA"),
  @("hf:lmms-lab/RefCOCOplus", "testB", "data/refcoco_plus/testB"),
  @("hf:lmms-lab/RefCOCOg",    "val",   "data/refcocog/val"),
  @("hf:lmms-lab/RefCOCOg",    "test",  "data/refcocog/test")
)
foreach ($s in $SETS) {
  python tools/convert_hf_to_jsonl.py --dataset $s[0] --split $s[1] --expr-field answer `
    --bbox-field bbox --bbox-format xywh --limit $ROWS --out $s[2]
}

# ScreenSpot (GUI grounding, out-of-domain). bbox is normalised xyxy in this upload.
python tools/convert_hf_to_jsonl.py --dataset hf:rootsautomation/ScreenSpot --split test `
  --expr-field instruction --bbox-field bbox --bbox-format xyxy_norm --out data/screenspot/test

# DIOR-RSVG (remote sensing, out-of-domain). Download + unzip first, see README.
# python tools/convert_dior_rsvg.py --root data/DIOR_RSVG --out data/dior_rsvg

# Toy synthetic shapes (sanity check / pipeline dry run)
python tools/make_toy_dataset.py --n 60 --out data/toy
