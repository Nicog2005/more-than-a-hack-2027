"""Audit delivered observations without interpolation or ground-truth labels.

Writes aggregate reports only; never exports frame-level tracking.
Run from the repo: .venv/Scripts/python starter/audit_tracking_coverage.py
    --data-dir <dataset folder> --out evaluation/reports/tracking-coverage
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import asset_paths, half_video_blocks, source_frames


def coverage(mask):
    return {"frames": int(np.count_nonzero(mask)),
            "percent": round(100 * float(np.mean(mask)), 4)}


def gaps(present):
    edges = np.diff(np.r_[False, ~present, False].astype(np.int8))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return {"runs": len(lengths), "max_seconds": float(lengths.max() / 25) if len(lengths) else 0,
            "median_seconds": float(np.median(lengths) / 25) if len(lengths) else 0,
            "runs_at_least_1s": int(np.count_nonzero(lengths >= 25)),
            "runs_at_least_5s": int(np.count_nonzero(lengths >= 125))}


def audit_source(source, data_dir):
    paths = asset_paths(data_dir=data_dir, source=source)
    root = ET.parse(paths["atd_meta"]).getroot()
    teams = {t.get("id"): t.findtext("Name") for t in root.iter("Team")}
    players = list(root.iter("Player"))
    team_ids = list(teams)
    home = np.array([p.get("teamId") == team_ids[0] for p in players])
    away = np.array([p.get("teamId") == team_ids[1] for p in players])
    assert np.all(home | away)
    n = source_frames(source)
    k = len(players)
    seen = np.zeros(n, dtype=bool)
    count_h = np.zeros(n, dtype=np.int16)
    count_a = np.zeros(n, dtype=np.int16)
    speeds = np.zeros(n, dtype=np.int16)
    ball_xy = np.zeros(n, dtype=bool)
    ball_speed = np.zeros(n, dtype=bool)
    partial_player_xy = partial_ball_xy = out_of_pitch_player = out_of_pitch_ball = 0
    present_positions = present_speeds = 0
    with open(paths["atd_raw"], encoding="utf-8") as handle:
        for line in handle:
            raw_id, raw_players, raw_ball = line.strip().split(":")
            f = int(raw_id) - 1
            assert 0 <= f < n and not seen[f], "Invalid or duplicate frame"
            seen[f] = True
            values = np.fromstring(raw_players.replace(";", ","), sep=",")
            assert values.size == k * 3
            values = values.reshape(k, 3)
            finite = np.isfinite(values)
            xy = finite[:, :2].all(axis=1)
            count_h[f] = np.count_nonzero(xy & home)
            count_a[f] = np.count_nonzero(xy & away)
            speeds[f] = np.count_nonzero(xy & finite[:, 2])
            present_positions += int(xy.sum())
            present_speeds += int(np.count_nonzero(xy & finite[:, 2]))
            partial_player_xy += int(np.count_nonzero(finite[:, 0] != finite[:, 1]))
            out_of_pitch_player += int(np.count_nonzero(xy & ((values[:, :2] < 0) | (values[:, :2] > 1)).any(axis=1)))
            ball = np.fromstring(raw_ball, sep=",")
            assert ball.size == 3
            ball_xy[f] = np.isfinite(ball[:2]).all()
            ball_speed[f] = ball_xy[f] and np.isfinite(ball[2])
            partial_ball_xy += int(np.isfinite(ball[0]) != np.isfinite(ball[1]))
            out_of_pitch_ball += int(ball_xy[f] and ((ball[:2] < 0) | (ball[:2] > 1)).any())
    assert seen.all(), "Missing tracking records"
    hom = pd.read_csv(paths["homography"])
    assert len(hom) == n and np.array_equal(hom["frame"], np.arange(n))
    hcols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
    matrices = hom[hcols].to_numpy().reshape(-1, 3, 3)
    hvalid = (hom["valid"].to_numpy() == 1) & np.isfinite(matrices).all(axis=(1, 2))
    invertible = np.zeros(n, dtype=bool)
    invertible[hvalid] = np.abs(np.linalg.det(matrices[hvalid])) > 1e-12
    tag_any = np.zeros(n, dtype=bool)
    tag_named = np.zeros(n, dtype=bool)
    tag_home = np.zeros(n, dtype=bool)
    tag_away = np.zeros(n, dtype=bool)
    tag_counts = Counter()
    tag_groups = {}
    event_count = 0
    if "smart_tagging" in paths:
        for instance in ET.parse(paths["smart_tagging"]).iter("instance"):
            event_count += 1
            tag_counts[instance.findtext("code")] += 1
            tags = {label.findtext("group"): label.findtext("text") for label in instance.findall("label")}
            for key, val in tags.items():
                tag_groups.setdefault(key, Counter())[val] += 1
            start = max(0, round(float(instance.findtext("start")) * 25))
            end = min(n, round(float(instance.findtext("end")) * 25))
            tag_any[start:end] = True
            if tags.get("Team") in teams.values():
                tag_named[start:end] = True
                (tag_home if tags["Team"] == teams[team_ids[0]] else tag_away)[start:end] = True
    count = count_h + count_a
    masks = {
        "tracking_record": seen,
        "ball_xy": ball_xy,
        "ball_xy_and_speed": ball_speed,
        "at_least_one_player_each_team": (count_h > 0) & (count_a > 0),
        "exactly_22_tracks": count == 22,
        "exactly_11_per_team": (count_h == 11) & (count_a == 11),
        "at_least_11_per_team": (count_h >= 11) & (count_a >= 11),
        "fewer_than_22_tracks": count < 22,
        "more_than_22_tracks": count > 22,
        "all_observed_players_have_speed": (speeds == count) & (count > 0),
        "homography_valid_finite": hvalid,
        "homography_invertible": hvalid & invertible,
        "ball_and_both_teams_and_homography": ball_xy & (count_h > 0) & (count_a > 0) & hvalid,
        "ball_and_11_per_team_and_homography": ball_xy & (count_h == 11) & (count_a == 11) & hvalid,
        "ball_and_11_per_team_and_all_speeds_and_homography": ball_speed & (count_h == 11) & (count_a == 11) & (speeds == 22) & hvalid,
    }
    if event_count:
        masks.update({"any_smart_tag": tag_any, "any_named_team_smart_tag": tag_named,
                      "smart_tags_for_both_teams": tag_home & tag_away})
    blocks = [("whole_video", 0, n - 1)]
    if source == "tactical":
        blocks += [(f"half_{half}", lo, hi) for half, lo, hi in half_video_blocks("BAR-ATH")]
    reports = {}
    for name, lo, hi in blocks:
        sl = slice(lo, hi + 1)
        errors = hom["reprojection_error"].to_numpy()[sl]
        good_errors = errors[hvalid[sl] & np.isfinite(errors)]
        reports[name] = {
            "first_video_frame": lo, "last_video_frame": hi, "total_frames": hi - lo + 1,
            "duration_seconds": (hi - lo + 1) / 25,
            "coverage": {key: coverage(mask[sl]) for key, mask in masks.items()},
            "tracks": {"mean_total": float(count[sl].mean()), "mean_home": float(count_h[sl].mean()),
                       "mean_away": float(count_a[sl].mean()), "min": int(count[sl].min()),
                       "max": int(count[sl].max()), "histogram": dict(sorted(Counter(map(int, count[sl])).items()))},
            "ball_missing_gaps": gaps(ball_xy[sl]),
            "homography_error": dict(zip(["median", "p95", "max"], map(float, np.quantile(good_errors, [.5, .95, 1])))),
        }
    return {"source": source, "teams": list(teams.values()), "slots": k,
            "slots_per_team": {teams[team_ids[0]]: int(home.sum()), teams[team_ids[1]]: int(away.sum())},
            "raw_channels_per_player": 3, "raw_channels_ball": 3,
            "position_types": dict(Counter(p.findtext(".//ProviderParameter[Name='position_type']/Value") for p in players)),
            "partial_player_xy": partial_player_xy, "partial_ball_xy": partial_ball_xy,
            "observed_player_positions": present_positions, "observed_player_positions_with_speed": present_speeds,
            "observations_outside_0_1": {"player": out_of_pitch_player, "ball": out_of_pitch_ball},
            "smart_tag_events": event_count, "smart_tag_codes": dict(tag_counts),
            "smart_tag_groups": {key: dict(value) for key, value in tag_groups.items()}, "periods": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--out", default="evaluation/reports/tracking-coverage")
    args = parser.parse_args()
    report = {"method": "Finite raw observations, no interpolation. Tactical half bounds from games.py; inclusive. Smart tags use [round(start*25),round(end*25)). Availability is not accuracy.",
              "sources": [audit_source(s, args.data_dir) for s in ("tactical", "broadcast")]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for source in report["sources"]:
        for period, values in source["periods"].items():
            for signal, value in values["coverage"].items():
                rows.append({"source": source["source"], "period": period, "signal": signal,
                             "total_frames": values["total_frames"], **value})
    pd.DataFrame(rows).to_csv(out / "coverage.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
