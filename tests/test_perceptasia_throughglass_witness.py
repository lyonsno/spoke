from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

from spoke import perceptasia_throughglass_witness as witness


def _write_rgb_png(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
    rows = []
    stride = width * 3
    for row in range(height):
        start = row * width
        raw = bytearray()
        for red, green, blue in pixels[start : start + width]:
            raw.extend((red, green, blue))
        rows.append(b"\x00" + bytes(raw[:stride]))
    payload = zlib.compress(b"".join(rows))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", payload)
        + chunk(b"IEND", b"")
    )


def _blank_frosted_frame(path: Path) -> None:
    width, height = 220, 150
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            color = (38, 38, 38)
            if 28 <= x <= 190 and 20 <= y <= 126:
                color = (122, 122, 122)
                if x < 35 or x > 183 or y < 28 or y > 118:
                    color = (82, 82, 82)
            pixels.append(color)
    _write_rgb_png(path, width, height, pixels)


def _perceptasia_like_frame(path: Path) -> None:
    width, height = 220, 150
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            color = (30, 32, 34)
            if 24 <= x <= 194 and 18 <= y <= 130:
                color = (18, 20, 22)
                if x < 34 or x > 184 or y < 28 or y > 120:
                    color = (74, 76, 78)
                if (x + y * 2) % 17 == 0 and 40 <= x <= 178:
                    color = (80, 155, 135)
                if (x * 3 - y) % 29 == 0 and 30 <= y <= 118:
                    color = (80, 110, 180)
                if abs((x - 40) - (y - 28) * 2) < 2:
                    color = (185, 95, 118)
                if abs((x - 178) + (y - 112) * 3) < 2:
                    color = (170, 150, 78)
                if (x - 108) ** 2 + (y - 74) ** 2 < 13**2:
                    color = (68, 128, 170) if (x + y) % 4 else (210, 222, 230)
            pixels.append(color)
    _write_rgb_png(path, width, height, pixels)


def _perceptasia_graph_frame_without_material_slab(path: Path) -> None:
    width, height = 220, 150
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            color = (16, 18, 20)
            if 24 <= x <= 194 and 18 <= y <= 130:
                color = (12, 14, 16)
                if (x + y * 2) % 19 == 0:
                    color = (55, 178, 142)
                if (x * 3 - y * 2) % 31 == 0:
                    color = (76, 132, 226)
                if abs((x - 34) - (y - 22) * 2) < 2:
                    color = (214, 96, 136)
                if abs((x - 185) + (y - 116) * 3) < 2:
                    color = (218, 178, 72)
                if (x - 110) ** 2 + (y - 76) ** 2 < 11**2:
                    color = (62, 166, 220) if (x + y) % 4 else (225, 236, 240)
            pixels.append(color)
    _write_rgb_png(path, width, height, pixels)


def _good_throughglass_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Perceptasia Throughglass: setup begin url=http://localhost:8753",
                "Perceptasia Throughglass: WKWebView request loaded",
                "Perceptasia Throughglass: content verified",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_default_output_dir_is_throughglass_named(tmp_path, monkeypatch):
    monkeypatch.setattr(witness, "_timestamp_slug", lambda: "20260531T000000Z")

    assert witness.default_output_dir(tmp_path) == tmp_path / "throughglass-autowitness-20260531T000000Z"


def test_throughglass_witness_defaults_to_passive_capture(tmp_path, monkeypatch, capsys):
    calls = []

    def run_witness_window(**kwargs):
        calls.append(kwargs)
        index = Path(kwargs["output_dir"]) / "witness-index.json"
        index.parent.mkdir(parents=True)
        index.write_text(
            '{"throughglass_contract":{"passed":true,"content_verified":true,"visual_content":{"passed":true,"classifier_version":"throughglass_pixels.v3"}}}\n',
            encoding="utf-8",
        )
        return index

    monkeypatch.setattr(witness, "_timestamp_slug", lambda: "20260531T010203Z")
    monkeypatch.setattr(witness, "run_witness_window", run_witness_window)

    assert witness.main(["--output-root", str(tmp_path), "--duration", "1.25"]) == 0

    assert calls[0]["output_dir"] == tmp_path / "throughglass-autowitness-20260531T010203Z"
    assert calls[0]["duration_seconds"] == 1.25
    assert calls[0]["capture_profile"] == "low_perturbation"
    assert calls[0]["lane"] == "perceptasia-throughglass-graft"
    assert calls[0]["diaulos"] == "Operator Memory"
    assert calls[0]["source_window"] == "Perceptasia Throughglass / Assistant Overlay"
    assert calls[0]["stimulus"] == {"mode": "passive-throughglass"}
    assert "witness-index.json" in capsys.readouterr().out


def test_throughglass_witness_launch_mode_uses_selected_target_contract(tmp_path, monkeypatch):
    calls = []

    def run_autonomous_hammer_witness(**kwargs):
        calls.append(kwargs)
        index = Path(kwargs["output_dir"]) / "witness-index.json"
        index.parent.mkdir(parents=True)
        index.write_text(
            '{"throughglass_contract":{"passed":true,"content_verified":true,"visual_content":{"passed":true,"classifier_version":"throughglass_pixels.v3"}}}\n',
            encoding="utf-8",
        )
        return index

    monkeypatch.setattr(witness, "run_autonomous_hammer_witness", run_autonomous_hammer_witness)

    assert witness.main(["--launch", "--output-dir", str(tmp_path / "out"), "--duration", "2"]) == 0

    assert calls[0]["output_dir"] == tmp_path / "out"
    assert calls[0]["capture_profile"] == "stress"
    assert calls[0]["launch_target"] == "perceptasia_throughglass_graft"
    assert calls[0]["hammer_toggles"] == 0
    assert calls[0]["source_app"] == "Spoke"


def test_throughglass_witness_watch_mode_captures_later_operator_events(tmp_path, monkeypatch):
    calls = []

    def run_trace_triggered_witness(**kwargs):
        calls.append(kwargs)
        index = Path(kwargs["output_dir"]) / "watch-index.json"
        index.parent.mkdir(parents=True)
        index.write_text('{"schema":"spoke.retina_lasso_trace_trigger_watch.v1","capture_count":1}\n', encoding="utf-8")
        return index

    monkeypatch.setattr(witness, "run_trace_triggered_witness", run_trace_triggered_witness)

    assert (
        witness.main(
            [
                "--watch-trace",
                "--trace",
                str(tmp_path / "trace.jsonl"),
                "--output-dir",
                str(tmp_path / "out"),
                "--watch-timeout",
                "7",
                "--event-capture-duration",
                "0.5",
                "--watch-max-captures",
                "3",
                "--max-trigger-lag",
                "2",
                "--fps",
                "12",
            ]
        )
        == 0
    )

    assert calls[0]["output_dir"] == tmp_path / "out"
    assert calls[0]["watch_timeout_seconds"] == 7
    assert calls[0]["event_capture_duration_seconds"] == 0.5
    assert calls[0]["max_captures"] == 3
    assert calls[0]["max_trigger_lag_seconds"] == 2
    assert calls[0]["fps"] == 12
    assert calls[0]["trigger_events"] == {
        "throughglass.publish.materialize",
        "throughglass.publish.rest",
        "throughglass.publish.dismiss",
        "throughglass.publish.hidden",
    }


def test_throughglass_witness_fails_when_capture_lacks_content_proof(tmp_path, monkeypatch):
    def run_witness_window(**kwargs):
        index = Path(kwargs["output_dir"]) / "witness-index.json"
        index.parent.mkdir(parents=True)
        index.write_text('{"frame_count":96}\n', encoding="utf-8")
        return index

    monkeypatch.setattr(witness, "_timestamp_slug", lambda: "20260531T010203Z")
    monkeypatch.setattr(witness, "run_witness_window", run_witness_window)

    assert (
        witness.main(
            [
                "--output-root",
                str(tmp_path),
                "--duration",
                "1.25",
                "--log-path",
                str(tmp_path / "missing.log"),
            ]
        )
        == 2
    )


def test_throughglass_contract_rejects_blank_frosted_pixels_even_when_logs_pass(tmp_path):
    index = tmp_path / "witness-index.json"
    frame = tmp_path / "screen-capture-000.png"
    log = tmp_path / "spoke.log"
    _blank_frosted_frame(frame)
    _good_throughglass_log(log)
    index.write_text(json.dumps({"frame_count": 1}) + "\n", encoding="utf-8")

    contract = witness.annotate_throughglass_contract(index, log_paths=[log])

    assert contract["webview_loaded"] is True
    assert contract["content_verified"] is True
    assert contract["visual_content"]["passed"] is False
    assert contract["passed"] is False
    assert contract["visual_content"]["failure_reason"] == "captured_pixels_do_not_show_throughglass_content"


def test_throughglass_contract_accepts_visible_perceptasia_like_pixels(tmp_path):
    index = tmp_path / "witness-index.json"
    frame = tmp_path / "screen-capture-000.png"
    log = tmp_path / "spoke.log"
    _perceptasia_like_frame(frame)
    _good_throughglass_log(log)
    index.write_text(json.dumps({"frame_count": 1}) + "\n", encoding="utf-8")

    contract = witness.annotate_throughglass_contract(index, log_paths=[log])

    assert contract["visual_content"]["passed"] is True
    assert contract["passed"] is True


def test_throughglass_contract_accepts_visible_graph_pixels_without_material_slab(tmp_path):
    index = tmp_path / "witness-index.json"
    frame = tmp_path / "screen-capture-000.png"
    log = tmp_path / "spoke.log"
    _perceptasia_graph_frame_without_material_slab(frame)
    _good_throughglass_log(log)
    index.write_text(json.dumps({"frame_count": 1}) + "\n", encoding="utf-8")

    contract = witness.annotate_throughglass_contract(index, log_paths=[log])

    visual = contract["visual_content"]
    assert visual["passed"] is True
    assert visual["best_metrics"]["panel_material_fraction"] < witness._VISUAL_PASS_THRESHOLDS[
        "panel_material_fraction"
    ]
    assert visual["best_metrics"]["pass_reason"] == "colored_graph_content"
    assert contract["passed"] is True


def test_throughglass_contract_rejects_early_content_when_settled_tail_is_blank(tmp_path):
    index = tmp_path / "witness-index.json"
    log = tmp_path / "spoke.log"
    _good_throughglass_log(log)
    frames = []
    for index_number in range(8):
        frame = tmp_path / f"screen-capture-{index_number:03d}.png"
        if index_number < 3:
            _perceptasia_like_frame(frame)
        else:
            _blank_frosted_frame(frame)
        frames.append(str(frame))
    index.write_text(
        json.dumps({"frame_count": len(frames), "retina_lasso_manifest": str(tmp_path / "manifest.json")}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"frames": frames}) + "\n", encoding="utf-8")

    contract = witness.annotate_throughglass_contract(index, log_paths=[log])

    assert contract["visual_content"]["passed"] is False
    assert contract["passed"] is False
    assert contract["visual_content"]["failure_reason"] == "settled_tail_lacks_throughglass_content"
