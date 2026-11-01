import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding.data import load_jsonl, subset_indices, to_xyxy
from grounding.metrics import area_bucket, failure_category, has_spatial, iou_xyxy, length_bucket, summarize
from grounding.prompts import build_prompt, build_target


def test_iou():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xyxy([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert abs(iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_buckets():
    assert area_bucket([0, 0, 5, 5], 100, 100) == "small"
    assert area_bucket([0, 0, 20, 20], 100, 100) == "medium"
    assert area_bucket([0, 0, 50, 50], 100, 100) == "large"
    assert length_bucket("red cup") == "short(<=3w)"
    assert length_bucket("the red cup on the table") == "medium(4-7w)"
    assert has_spatial("the man on the left") and not has_spatial("a red car")


def test_summarize_and_category():
    recs = [
        {"iou": 0.9, "parsed": True, "expression": "a dog", "gt": [0, 0, 50, 50], "width": 100, "height": 100,
         "pred": [1, 1, 50, 50], "latency_s": 0.5, "n_boxes": 1},
        {"iou": 0.0, "parsed": False, "expression": "the left cat", "gt": [0, 0, 5, 5], "width": 100, "height": 100,
         "pred": None, "latency_s": 0.5, "n_boxes": 0},
        {"iou": 0.3, "parsed": True, "expression": "x", "gt": [0, 0, 5, 5], "width": 100, "height": 100,
         "pred": [0, 0, 8, 8], "latency_s": 0.5, "n_boxes": 2},
    ]
    s = summarize(recs)
    assert s["n"] == 3
    assert abs(s["acc@0.5"] - 1 / 3) < 1e-9
    assert abs(s["parse_fail_rate"] - 1 / 3) < 1e-9
    assert abs(s["multi_box_rate"] - 1 / 3) < 1e-9
    assert s["buckets"]["gt_area"]["small"]["n"] == 2
    assert failure_category(recs[0]) == "correct"
    assert failure_category(recs[1]) == "parse_failure"
    assert failure_category(recs[2]) == "loose_localisation"


def test_to_xyxy_formats():
    assert to_xyxy([10, 20, 30, 40], "xywh", 100, 100) == [10, 20, 40, 60]
    assert to_xyxy([0.1, 0.2, 0.3, 0.4], "xyxy_norm", 100, 200) == [10, 40, 30, 80]
    assert to_xyxy([50, 50, 20, 10], "cxcywh", 100, 100) == [40, 45, 60, 55]


def test_subset_nesting():
    a = set(subset_indices(1000, 200, 0))
    b = set(subset_indices(1000, 500, 0))
    assert a <= b and len(a) == 200 and len(b) == 500
    assert subset_indices(10, None, 0) == list(range(10))


def test_load_jsonl(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    Image.new("RGB", (64, 32)).save(tmp_path / "images" / "a.jpg")
    rows = [
        {"id": "a_0", "image": "images/a.jpg", "expression": "thing", "bbox": [1, 2, 10, 5], "bbox_format": "xywh"},
        {"id": "a_1", "image": "images/a.jpg", "expression": "other", "bbox": [1, 2, 10, 5]},
    ]
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    s = load_jsonl(str(tmp_path / "d.jsonl"))
    assert len(s) == 2 and s[0].width == 64 and s[0].height == 32
    assert s[0].bbox_xyxy == [1, 2, 11, 7]      # xywh converted
    assert s[1].bbox_xyxy == [1, 2, 10, 5]      # default xyxy
    assert s[0].load_image().size == (64, 32)


def test_prompt_and_target():
    assert "red cup" in build_prompt("json", "red cup")
    t = build_target("red cup", [1, 2, 3, 4], "json")
    assert t.startswith("```json") and '"bbox_2d": [1, 2, 3, 4]' in t
    from grounding.parse import first_box

    assert first_box(t) == [1.0, 2.0, 3.0, 4.0]
