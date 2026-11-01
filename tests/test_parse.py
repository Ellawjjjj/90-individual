import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grounding.parse import extract_boxes, first_box, from_original_xyxy, strip_thinking, to_original_xyxy


def test_json_fenced():
    txt = '```json\n[{"bbox_2d": [120, 45, 388, 512], "label": "dog"}]\n```'
    assert extract_boxes(txt) == [[120.0, 45.0, 388.0, 512.0]]


def test_json_raw_dict():
    txt = '{"bbox_2d": [1, 2, 3, 4]}'
    assert extract_boxes(txt) == [[1.0, 2.0, 3.0, 4.0]]


def test_json_multiple_objects_order():
    txt = '[{"bbox_2d": [1,2,3,4],"label":"a"},{"bbox_2d": [5,6,7,8],"label":"b"}]'
    assert extract_boxes(txt) == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_plain_list_with_prose():
    txt = "The cat is at [10, 20, 110, 220] in the image."
    assert first_box(txt) == [10.0, 20.0, 110.0, 220.0]


def test_pair_format():
    txt = "<|box_start|>(10,20),(110,220)<|box_end|>"
    assert first_box(txt) == [10.0, 20.0, 110.0, 220.0]


def test_thinking_stripped():
    txt = "<think>maybe [0,0,999,999]? no, [1,1,2,2]</think>\n```json\n[{\"bbox_2d\": [5,6,7,8]}]\n```"
    assert first_box(txt) == [5.0, 6.0, 7.0, 8.0]
    assert strip_thinking(txt).strip().startswith("```json")


def test_no_box():
    assert extract_boxes("I cannot find it.") == []
    assert first_box("") is None


def test_broken_json_falls_back_to_regex():
    txt = '```json\n[{"bbox_2d": [12, 34, 56, 78], "label": "x"\n```'  # missing closing brackets
    assert first_box(txt) == [12.0, 34.0, 56.0, 78.0]


def test_relative_1000():
    box = to_original_xyxy([500, 500, 1000, 1000], "relative_1000", 640, 480)
    assert box == [320.0, 240.0, 640.0, 480.0]


def test_absolute_input_scaling():
    # model saw a 448x448 resize of a 640x480 image
    box = to_original_xyxy([224, 224, 448, 448], "absolute_input", 640, 480, input_w=448, input_h=448)
    assert [round(v, 3) for v in box] == [320.0, 240.0, 640.0, 480.0]


def test_absolute_original_clip_and_sort():
    box = to_original_xyxy([700, 500, 10, -5], "absolute_original", 640, 480)
    assert box == [10.0, 0.0, 640.0, 480.0]


def test_roundtrip_relative():
    orig = [100.0, 50.0, 300.0, 200.0]
    enc = from_original_xyxy(orig, "relative_1000", 640, 480)
    dec = to_original_xyxy(enc, "relative_1000", 640, 480)
    assert all(abs(a - b) < 1.0 for a, b in zip(orig, dec))


def test_roundtrip_absolute_input():
    orig = [100.0, 50.0, 300.0, 200.0]
    enc = from_original_xyxy(orig, "absolute_input", 640, 480, input_w=448, input_h=336)
    dec = to_original_xyxy(enc, "absolute_input", 640, 480, input_w=448, input_h=336)
    assert all(abs(a - b) < 1.5 for a, b in zip(orig, dec))
