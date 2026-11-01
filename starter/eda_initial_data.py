"""Reproducible local EDA of delivered ATD, homography, tags and video headers.

No ground truth is read. No corrections, interpolation or model fitting.
Only aggregate tables/figures are persisted, under the ignored reports directory.
"""

import argparse
from collections import Counter
from datetime import date
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import asset_paths, half_video_blocks, source_frames, GAMES

FPS = 25
SCALE = np.array(GAMES["BAR-ATH"]["pitch"])
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def stats(values):
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": len(values), **dict(zip(["min", "p01", "p05", "p50", "p95", "p99", "max"],
            map(float, np.quantile(values, [0, .01, .05, .5, .95, .99, 1]))))}


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return starts, stops, stops - starts


def pct(mask):
    return float(np.mean(mask) * 100) if np.size(mask) else None


def count(mask):
    return int(np.count_nonzero(mask))


def table(rows, columns=None):
    rows = list(rows)
    if not rows:
        return "Sin observaciones.\n"
    columns = columns or list(rows[0])
    def fmt(value):
        if value is None:
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:,.3f}".replace(",", "_").replace(".", ",").replace("_", ".")
        if isinstance(value, (int, np.integer)):
            return f"{value:,}".replace(",", ".")
        return str(value).replace("|", "/")
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join("| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows) + "\n"


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / "figures" / f"{name}.png")
    plt.close(fig)


def video_info(path, expected):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"Cannot open {path}"
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    decoded = 0
    for f in np.linspace(0, n - 1, 5, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, image = cap.read()
        decoded += bool(ok and image is not None)
    info = {"frames_header": n, "fps_header": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds_header": n / fps, "sample_decodes": decoded,
            "samples_requested": 5, "matches_atd_expected": n == expected,
            "frames_atd": expected, "atd_minus_video_frames": expected - n}
    cap.set(cv2.CAP_PROP_POS_FRAMES, expected - 1)
    last_ok, last_image = cap.read()
    info["last_atd_index_decodes"] = bool(last_ok and last_image is not None)
    cap.release()
    return info


def load(source, data_dir):
    paths = asset_paths(data_dir=str(data_dir), source=source)
    root = ET.parse(paths["atd_meta"]).getroot()
    teams = list(root.iter("Team"))
    players = list(root.iter("Player"))
    home = np.array([p.get("teamId") == teams[0].get("id") for p in players])
    assert all(p.get("teamId") in {t.get("id") for t in teams} for p in players)
    n, k = source_frames(source), len(players)
    data = np.full((n, k + 1, 3), np.nan)
    seen = np.zeros(n, dtype=bool)
    for row in Path(paths["atd_raw"]).open(encoding="utf-8"):
        raw_id, raw_players, ball = row.strip().split(":")
        frame = int(raw_id) - 1
        assert 0 <= frame < n and not seen[frame]
        seen[frame] = True
        values = np.fromstring(raw_players.replace(";", ",") + "," + ball, sep=",")
        assert values.size == (k + 1) * 3
        data[frame] = values.reshape(k + 1, 3)
    assert seen.all()
    hom = pd.read_csv(paths["homography"])
    assert np.array_equal(hom.frame.to_numpy(), np.arange(n))
    return paths, root, teams, players, home, data, hom


def analyze(source, data_dir, out):
    print(f"Reading {source} ...", flush=True)
    paths, root, teams, players, home, data, hom = load(source, data_dir)
    n, k = len(data), len(players)
    info = video_info(paths["video"], n)
    assert info["fps_header"] == FPS
    positions = data[:, :-1, :2]
    player_ok = np.isfinite(positions).all(axis=2)
    ball = data[:, -1, :2]
    ball_ok = np.isfinite(ball).all(axis=1)
    player_speed_ok = np.isfinite(data[:, :-1, 2]) & player_ok
    ball_speed_ok = np.isfinite(data[:, -1, 2]) & ball_ok
    hc, ac = player_ok[:, home].sum(axis=1), player_ok[:, ~home].sum(axis=1)
    total = hc + ac
    matrices = hom[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(n, 3, 3)
    hgood = hom.valid.eq(1).to_numpy() & np.isfinite(matrices).all(axis=(1, 2))
    hcond = np.full(n, np.nan)
    hcond[hgood] = np.linalg.cond(matrices[hgood])
    _, _, invalid_lengths = runs(~hgood)
    # Consecutive observed frames only. Never bridge a missing observation.
    player_step = np.linalg.norm(np.diff(positions, axis=0) * SCALE, axis=2)
    ball_step = np.linalg.norm(np.diff(ball, axis=0) * SCALE, axis=1)
    # Same-team proximity is a diagnostic candidate, not proof of duplicates.
    close_pair = np.zeros(n, dtype=bool)
    for i in range(k):
        for j in range(i):
            if home[i] == home[j]:
                distance2 = np.sum(((positions[:, i] - positions[:, j]) * SCALE) ** 2, axis=1)
                close_pair |= distance2 < .25 ** 2
    nearest_home = np.full(n, np.nan)
    nearest_away = np.full(n, np.nan)
    # Avoid retaining a full pairwise player/ball distance tensor.
    for lo in range(0, n, 5000):
        hi = min(n, lo + 5000)
        distances = np.linalg.norm((positions[lo:hi] - ball[lo:hi, None, :]) * SCALE, axis=2)
        distances[~np.isfinite(distances)] = np.inf
        nearest_home[lo:hi] = np.min(distances[:, home], axis=1)
        nearest_away[lo:hi] = np.min(distances[:, ~home], axis=1)
    nearest_home[~np.isfinite(nearest_home)] = np.nan
    nearest_away[~np.isfinite(nearest_away)] = np.nan
    nearest = np.fmin(nearest_home, nearest_away)
    events = []
    tag_count = np.zeros(n, dtype=int)
    tag_named = np.zeros(n, dtype=bool)
    tag_home, tag_away = np.zeros(n, bool), np.zeros(n, bool)
    invalid_events = 0
    if "smart_tagging" in paths:
        for instance in ET.parse(paths["smart_tagging"]).iter("instance"):
            tags = {label.findtext("group"): label.findtext("text") for label in instance.findall("label")}
            start, end = float(instance.findtext("start")), float(instance.findtext("end"))
            a, b = round(start * FPS), round(end * FPS)
            invalid_events += not (0 <= a < b <= n)
            assert 0 <= a < b <= n
            tag_count[a:b] += 1
            if tags.get("Team") in [t.findtext("Name") for t in teams]:
                tag_named[a:b] = True
                (tag_home if tags["Team"] == teams[0].findtext("Name") else tag_away)[a:b] = True
            events.append({"code": instance.findtext("code"), "start_s": start, "end_s": end,
                           "duration_s": end - start, **tags})
    periods = [("video_completo", 0, n - 1)]
    if source == "tactical":
        periods += [("primera_parte", *half_video_blocks("BAR-ATH")[0][1:]),
                    ("segunda_parte", *half_video_blocks("BAR-ATH")[1][1:]),
                    ("antes", 0, 11184), ("descanso", 90652, 110807),
                    ("despues", 190198, n - 1)]
    periods_summary, slot_rows, gap_rows, minute_rows, unit_rows = {}, [], [], [], []
    for name, lo, hi in periods:
        sl = slice(lo, hi + 1)
        p, b = player_ok[sl], ball_ok[sl]
        start, stop, lengths = runs(~b)
        for a, z, length in zip(start, stop, lengths):
            gap_rows.append({"source": source, "period": name, "length_seconds": length / FPS})
        valid_steps = player_step[lo:hi].ravel()
        bsteps = ball_step[lo:hi]
        po = positions[sl]
        outside_players = ((po < 0) | (po > 1)).any(axis=2) & p
        outside_ball = ((ball[sl] < 0) | (ball[sl] > 1)).any(axis=1) & b
        repeated = np.all(np.diff(po, axis=0) == 0, axis=2) & p[:-1] & p[1:]
        gks = np.array([p_.findtext(".//ProviderParameter[Name='position_type']/Value") == "Goalkeeper" for p_ in players])
        _, _, inv_runs = runs(~hgood[sl])
        summary = {
            "first_frame": lo, "last_frame": hi, "frames": hi - lo + 1,
            "duration_s": (hi - lo + 1) / FPS,
            "coverage": {"ball_xy": pct(b), "ball_xy_speed": pct(ball_speed_ok[sl]),
                "both_teams": pct((hc[sl] > 0) & (ac[sl] > 0)),
                "exact_22": pct(total[sl] == 22), "exact_11_each": pct((hc[sl] == 11) & (ac[sl] == 11)),
                "ball_11_each_homography": pct(b & (hc[sl] == 11) & (ac[sl] == 11) & hgood[sl]),
                "full_with_speeds": pct(b & (hc[sl] == 11) & (ac[sl] == 11) & hgood[sl] & ball_speed_ok[sl] & (player_speed_ok[sl].sum(axis=1) == 22)),
                "both_goalkeepers": pct(p[:, gks].all(axis=1)),
                "homography": pct(hgood[sl]), "any_tag": pct(tag_count[sl] > 0),
                "named_tag": pct(tag_named[sl]), "both_teams_tags": pct(tag_home[sl] & tag_away[sl]),
                "all_observed_player_speeds": pct((player_speed_ok[sl].sum(axis=1) == total[sl]) & (total[sl] > 0))},
            "tracks": stats(total[sl]), "mean_tracks": float(total[sl].mean()),
            "mean_home": float(hc[sl].mean()), "mean_away": float(ac[sl].mean()),
            "less_22_pct": pct(total[sl] < 22), "more_22_pct": pct(total[sl] > 22),
            "ball_absence": {"runs": len(lengths), "length_s": stats(lengths / FPS),
                "frames_in_gaps_ge_5s_pct": float(lengths[lengths >= 125].sum() / len(b) * 100)},
            "player_observations": count(p), "player_speed_observations": count(player_speed_ok[sl]),
            "player_coordinates": {axis: stats(po[:, :, i]) for i, axis in enumerate(["x", "y"])},
            "ball_coordinates": {axis: stats(ball[sl, i]) for i, axis in enumerate(["x", "y"])},
            "outside_player_observations": count(outside_players), "outside_ball_frames": count(outside_ball),
            "same_team_pair_lt_025m_frames": count(close_pair[sl]),
            "same_team_pair_lt_025m_pct": pct(close_pair[sl]),
            "player_step_m": stats(valid_steps), "ball_step_m": stats(bsteps),
            "player_step_over_1m": count(valid_steps > 1), "ball_step_over_5m": count(bsteps > 5),
            "identical_player_adjacent_pairs": count(repeated),
            "observed_player_adjacent_pairs": count(p[:-1] & p[1:]),
            "player_s_raw": stats(data[sl, :-1, 2]), "ball_s_raw": stats(data[sl, -1, 2]),
            "nearest_player_m": stats(nearest[sl]),
            "ball_with_nearest_over_2m": count(nearest[sl] > 2),
            "ball_with_nearest_over_5m": count(nearest[sl] > 5),
            "reprojection_error": stats(hom.reprojection_error.to_numpy()[sl][hgood[sl]]),
            "homography_condition": stats(hcond[sl]), "homography_invalid_runs": len(inv_runs),
            "homography_invalid_max_s": float(max(inv_runs, default=0) / FPS),
        }
        periods_summary[name] = summary
        for j, player in enumerate(players):
            _, _, active = runs(p[:, j])
            _, _, missing = runs(~p[:, j])
            slot_rows.append({"source": source, "period": name, "slot": player.get("id"),
                "team": teams[0 if home[j] else 1].findtext("Name"),
                "role": player.findtext(".//ProviderParameter[Name='position_type']/Value"),
                "observed_frames": count(p[:, j]), "coverage_pct": pct(p[:, j]),
                "speed_frames": count(player_speed_ok[sl, j]), "active_runs": len(active),
                "median_active_s": float(np.median(active) / FPS) if len(active) else 0,
                "max_active_s": float(max(active, default=0) / FPS),
                "max_missing_s": float(max(missing, default=0) / FPS)})
        # One non-overlapping 1-second displacement sample per second and track.
        centers = np.arange(lo + 12, hi - 13 + 1, 25)
        if len(centers):
            v = np.linalg.norm((positions[centers + 13] - positions[centers - 12]) * SCALE, axis=2)
            s = data[centers, :-1, 2]
            continuous = np.ones(v.shape, bool)
            path = np.zeros(v.shape)
            for delta in range(-12, 14):
                continuous &= player_ok[centers + delta]
                if delta < 13:
                    path += player_step[centers + delta]
            eligible = continuous & np.isfinite(s) & (s > 0) & (v >= 1) & (v <= 10) & (v / np.maximum(path, 1e-9) >= .9)
            if count(eligible):
                ratios = s[eligible] / v[eligible]
                unit_rows.append({"source": source, "period": name, "samples": count(eligible),
                    "median_s_div_displacement_mps": float(np.median(ratios)),
                    "median_abs_error_assuming_raw_mps": float(np.median(np.abs(s[eligible] - v[eligible]))),
                    "median_abs_error_assuming_raw_kph": float(np.median(np.abs(s[eligible] / 3.6 - v[eligible])))})
    # Per-minute aggregate timeline: no raw coordinates persisted.
    for lo in range(0, n, FPS * 60):
        hi = min(n, lo + FPS * 60)
        minute_rows.append({"source": source, "video_minute": lo / FPS / 60,
            "frames": hi - lo, "ball_pct": pct(ball_ok[lo:hi]),
            "mean_players": float(total[lo:hi].mean()), "home_mean": float(hc[lo:hi].mean()),
            "away_mean": float(ac[lo:hi].mean()), "homography_pct": pct(hgood[lo:hi]),
            "tag_pct": pct(tag_count[lo:hi] > 0)})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    mins = pd.DataFrame(minute_rows)
    axes[0].plot(mins.video_minute, mins.ball_pct, label="Balón", color="#137c8b")
    axes[0].plot(mins.video_minute, mins.homography_pct, label="Homografía", alpha=.75)
    axes[0].set_ylabel("Cobertura (%)"); axes[0].legend()
    axes[1].plot(mins.video_minute, mins.home_mean, label="Barcelona")
    axes[1].plot(mins.video_minute, mins.away_mean, label="Athletic")
    axes[1].axhline(11, color="grey", ls="--"); axes[1].set_ylabel("Tracks medios"); axes[1].legend()
    axes[2].plot(mins.video_minute, mins.tag_pct, color="#aa3377")
    axes[2].set_ylabel("Frames con tag (%)"); axes[2].set_xlabel("Minuto del vídeo (no reloj del partido)")
    for ax in axes:
        if source == "tactical":
            for _, lo, hi in half_video_blocks("BAR-ATH"):
                ax.axvspan(lo / FPS / 60, (hi + 1) / FPS / 60, color="green", alpha=.06)
    fig.suptitle(f"{source}: agregados en ventanas de un minuto; verde = bloques evaluados")
    savefig(fig, out, f"{source}_timeline")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    groups = periods[1:3] if source == "tactical" else periods
    for name, lo, hi in groups:
        vals, freq = np.unique(total[lo:hi + 1], return_counts=True)
        axes[0].plot(vals, freq / (hi - lo + 1) * 100, marker=".", label=name)
        _, _, gl = runs(~ball_ok[lo:hi + 1])
        axes[1].hist(gl / FPS, bins=[0, 1, 5, 10, 20, 40, 80, 160, 320, 640], histtype="step", label=name)
    axes[0].set(xlabel="Tracks simultáneos", ylabel="Frames (%)"); axes[0].legend()
    axes[1].set(xlabel="Duración racha sin balón (s)", ylabel="Número de rachas"); axes[1].legend()
    savefig(fig, out, f"{source}_counts_gaps")
    # Slot availability aggregated by minute; no individual trajectory is drawn.
    minute_coverage = np.array([player_ok[a:min(a + 1500, n)].mean(axis=0) * 100
                               for a in range(0, n, 1500)]).T
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(minute_coverage, origin="lower", aspect="auto", vmin=0, vmax=100,
                   extent=[0, n / FPS / 60, .5, k + .5], cmap="viridis")
    ax.set(xlabel="Minuto del vídeo", ylabel="Slot de tracking", title=f"{source}: disponibilidad por slot y minuto")
    fig.colorbar(im, ax=ax, label="Frames con posición (%)")
    savefig(fig, out, f"{source}_slot_coverage")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, lo, hi in groups:
        distances = np.sort(nearest[lo:hi + 1])
        distances = distances[np.isfinite(distances)]
        axes[0].plot(distances, np.arange(1, len(distances) + 1) / len(distances), label=name)
        steps = np.sort(ball_step[lo:hi])
        steps = steps[np.isfinite(steps)]
        axes[1].plot(steps, np.arange(1, len(steps) + 1) / len(steps), label=name)
    axes[0].set(xlabel="Balón a jugador más cercano (m)", ylabel="Fracción acumulada", xscale="symlog")
    axes[1].set(xlabel="Desplazamiento balón entre frames (m)", ylabel="Fracción acumulada", xscale="symlog")
    axes[0].legend(); axes[1].legend()
    savefig(fig, out, f"{source}_ball_geometry")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name, lo, hi in [(axes[i], *g) for i, g in enumerate(groups)]:
        vals = hom.reprojection_error.to_numpy()[lo:hi + 1]
        vals = np.sort(vals[np.isfinite(vals)])
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals))
        ax.set(xlabel="reprojection_error (unidad no declarada)", ylabel="Fracción acumulada", title=name, xscale="symlog")
    if len(groups) == 1:
        axes[1].axis("off")
    savefig(fig, out, f"{source}_homography")
    result = {"source": source, "video": info, "slots": k,
        "team_names": [t.findtext("Name") for t in teams],
        "slots_per_team": [count(home), count(~home)],
        "xml_speed_unit": root.findtext(".//Channel[@id='s']/Unit"),
        "metadata_pitch": [root.findtext(".//FieldSize/Width"), root.findtext(".//FieldSize/Height")],
        "nonfinite": {"nan": count(np.isnan(data)), "inf": count(np.isinf(data)),
            "partial_player_xy": count(np.isfinite(positions[:, :, 0]) != np.isfinite(positions[:, :, 1])),
            "partial_ball_xy": count(np.isfinite(ball[:, 0]) != np.isfinite(ball[:, 1])),
            "negative_speeds": count(data[:, :, 2] < 0)},
        "events": {"count": len(events), "invalid_intervals": invalid_events,
            "codes": dict(Counter(e["code"] for e in events)),
            "groups": {key: dict(Counter(e.get(key, "<absent>") for e in events)) for key in ["Team", "Half", "Type", "Side", "Direction of ball entry", "Max Players in the box"]},
            "duration_s": stats([e["duration_s"] for e in events]), "max_simultaneous": int(tag_count.max()),
            "durations_by_code": {code: stats([e["duration_s"] for e in events if e["code"] == code]) for code in sorted({e["code"] for e in events})}},
        "periods": periods_summary, "speed_unit_diagnostic": unit_rows}
    for rows, filename in [(slot_rows, f"{source}_slots.csv"), (minute_rows, f"{source}_minutes.csv"),
                           (gap_rows, f"{source}_gap_lengths.csv"), (unit_rows, f"{source}_speed_units.csv")]:
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    # Cross-check the earlier independent audit if present, without depending on it.
    previous = out.parent / "tracking-coverage" / "coverage.json"
    if previous.exists():
        old = next(s for s in json.loads(previous.read_text(encoding="utf-8"))["sources"] if s["source"] == source)
        for name, old_name in [("video_completo", "whole_video"), ("primera_parte", "half_1"), ("segunda_parte", "half_2")]:
            if name in periods_summary:
                assert abs(periods_summary[name]["coverage"]["ball_xy"] - old["periods"][old_name]["coverage"]["ball_xy"]["percent"]) < .0001
    print(f"Completed {source}: {n} rows, header and coverage checks passed.", flush=True)
    return result


def build_report(results, inventory, environment, out):
    tactical, broadcast = results
    ps = tactical["periods"]
    main = ["primera_parte", "segunda_parte"]
    sections = ["# EDA de los datos iniciales — More Than A Hack 2027\n",
        f"Fecha: {date(2026, 11, 1).isoformat()}. Partido: FC Barcelona–Athletic Club. Análisis local de los archivos entregados, sin modificar datos originales.\n",
        "## 1. Conclusiones principales\n",
        "Los datos permiten construir un baseline de posesión, pero no constituyen una observación completa y corregida del partido. La posición del balón está disponible en el 59,31 % de los frames de los bloques oficiales. Balón + exactamente 11 tracks por equipo + homografía coexisten en el 31,00 %. La homografía táctica existe en todos los frames, pero eso no certifica su precisión. No se entregan cajas de detección ni identidad real estable.\n",
        "Las prioridades técnicas son manejar ausencias prolongadas del balón, evaluar la coherencia geométrica y temporal y utilizar eventos como evidencias. Reentrenar o sustituir detectores es una opción que este EDA ayuda a dirigir, no una mejora demostrada por estos recuentos. No se ha medido F1 ni elegido parámetros contra etiquetas públicas.\n",
        "Hallazgos adicionales respecto a la auditoría de cobertura:\n\n"
        "- **Unidad de velocidad:** en 40.432 y 40.699 muestras de las partes, la mediana s/(desplazamiento en m/s) es 1,0008 y 1,0004. Favorece claramente m/s frente a kph, aunque el proveedor no lo ha confirmado.\n"
        "- **Huecos largos:** casi todos los frames sin balón pertenecen a rachas de al menos cinco segundos (aproximadamente 99,9 % y 99,5 % de los frames ausentes, respectivamente). Una interpolación de huecos breves no resolverá la mayor parte de la ausencia.\n"
        "- **Tiempo sin partido:** dentro del bloque oficial de descanso hay posición de balón en el 94,07 % de frames y Smart Tagging en el 38,48 %. Son señales automáticas, no prueba de juego activo; conviene respetar los bloques oficiales y revisar la coherencia temporal antes de inferir estados.\n"
        "- **Broadcast:** el vídeo contiene un frame declarado menos que sus tablas; el índice de la última fila ATD no se pudo decodificar. No queda demostrado dónde se origina la discrepancia.\n"
        "- **Saltos de jugadores:** 77 y 223 pares consecutivos superan 1 m por frame (equivalente a más de 25 m/s aparentes). Son candidatos a error/identidad cambiante, no correcciones ya realizadas.\n",
        "## 2. Alcance, trazabilidad y método\n",
        "Unidad de observación principal: frame de vídeo. En jugadores, una observación es (frame, slot); en eventos, un intervalo. Se analizan completos ambos ATD y ambas homografías; Smart Tagging solo existe para táctica. Vídeos: lectura de cabecera y decodificación de cinco fotogramas espaciados por vídeo, sin decodificación exhaustiva ni anotación visual.\n",
        "Posición disponible = x e y finitos; velocidad disponible = s finita además de posición. No se interpolan huecos, se suavizan trayectorias ni se recortan coordenadas. Distancias: escala 105,3 × 68,0 m de games.py; el XML declara 105 × 68 m (diferencia longitudinal de 0,286 %). Desplazamientos: solo pares consecutivos observados, sin cruzar huecos ni límites de parte.\n",
        "Inventario con tamaños físicos; input_sha256.csv identifica exactamente las entradas no audiovisuales. Los vídeos se identifican por nombre, tamaño y propiedades de cabecera; no se calculó su hash completo. Los informes y gráficos permanecen locales en una carpeta ignorada por Git. No se han consultado ficheros de ground truth, servicios externos ni datos humanos adicionales del partido.\n",
        table(inventory), "Entorno ejecutado:\n", table([environment]),
        "## 3. Vídeos y alineación temporal\n",
        table([{"fuente": s["source"], **{key: s["video"][key] for key in ["frames_header", "frames_atd", "fps_header", "width", "height", "duration_seconds_header"]}} for s in results]),
        table([{"fuente": s["source"], **{key: s["video"][key] for key in ["sample_decodes", "samples_requested", "atd_minus_video_frames", "last_atd_index_decodes"]}} for s in results]),
        "La táctica coincide en cabecera, ATD y homografía (199.319 filas/frames). Broadcast presenta una discrepancia: vídeo con 164.487 frames declarados frente a 164.488 filas ATD y homografías. La tabla incluye el intento de decodificar el índice de la última fila ATD; esta prueba de extremo no determina si el desfase está solo al final o en otro punto. No se elimina una fila ni se aplica un offset arbitrario. La muestra decodificada no prueba la integridad de todos los fotogramas. La retransmisión tiene cortes/repeticiones y NO está sincronizada frame a frame con la táctica; no se trasplantan sus límites de partes.\n",
        table([{"periodo": name, **{key: p[key] for key in ["first_frame", "last_frame", "frames", "duration_s"]}} for name, p in ps.items()]),
        table([{"periodo": name, "balon_pct": p["coverage"]["ball_xy"], "tracks_medios": p["mean_tracks"], "H_valida_pct": p["coverage"]["homography"], "tags_pct": p["coverage"]["any_tag"]} for name, p in ps.items()]),
        "Los bloques de primera/segunda parte proceden de half_video_blocks en games.py, incluyen interrupciones y no son dos ventanas de 45 minutos. La segunda ventana oficial comienza antes del saque inicial de esa parte: no debe confundirse inicio del bloque evaluado con instante de reanudación. Los parámetros de partes del XML ATD son genéricos (primera abarca el vídeo y segunda vacía); no deben sustituir a los límites oficiales.\n",
        "Claves: ATD frame_id desde 1 → video_frame = frame_id − 1; homografía frame desde 0; eventos en segundos desde el inicio del vídeo táctico → round(segundos × 25). Los offsets grandes de games.py pertenecen al contador de origen del ground truth y no se suman al ATD entregado.\n",
        "## 4. Diccionario de datos y esquema\n",
        "ATD TXT: `frame_id:slot_1;slot_2;…;slot_k:ball`, cada grupo `x,y,s`. Coordenadas x,y normalizadas del campo, no píxeles. El XML aporta equipo y atributos por slot. Matrices CSV: frame, h00…h22, reprojection_error, valid. La homografía transforma campo normalizado → píxeles; la dirección inversa requiere invertirla y dividir por la coordenada homogénea.\n",
        table([{"fuente": s["source"], "slots": s["slots"], "Barcelona": s["slots_per_team"][0], "Athletic": s["slots_per_team"][1], "unidad_s_XML": s["xml_speed_unit"]} for s in results]),
        "Hay dos slots portero en cada fuente; los demás tienen position_type Unknown. En táctica son 24 y 46. Track n y ShirtNumber=n son índices del slot, no dorsal ni identidad real. El equipo está definido para el 100 % de slots, pero no se ha contrastado que la clasificación sea correcta. No se exportan boxes, máscaras, keypoints corporales, confianza por detección, altura del balón, pose 3D, vector de velocidad, poseedor por frame o balón vivo/muerto. El último estado es parte del objetivo del challenge.\n",
        "## 5. Integridad estructural y nulos\n",
        "Comprobado: IDs ATD únicos, consecutivos, dentro del rango esperado; ancho de cada fila (slots+balón)×3; equipo válido para cada slot; CSV de homografía exactamente alineado con frames de vídeo. Un NaN de velocidad no implica ausencia de posición. No se interpreta un slot no observado como jugador ausente del partido.\n",
        table([{"fuente": s["source"], **s["nonfinite"]} for s in results]),
        "## 6. Cobertura en las partes oficiales\n",
        "Todos los valores de la siguiente tabla son porcentajes de frames del bloque completo, no de tiempo efectivo de juego.\n",
        table([{"señal": key, **{name: ps[name]["coverage"][key] for name in main}} for key in ps[main[0]]["coverage"]]),
        "`exact_11_each` cuenta tracks, no 22 identidades certificadas. `ball_11_each_homography` añade balón e H válida; `full_with_speeds` exige también las 23 velocidades y es 0 % en ambas partes. `all_observed_player_speeds` solo exige velocidad de los tracks que sí tienen posición. Homografía válida no significa calibración exacta.\n",
        "![Disponibilidad temporal táctica](figures/tactical_timeline.png)\n",
        "## 7. Jugadores, continuidad de slots y fragmentación\n",
        table([{"periodo": name, "media_tracks": ps[name]["mean_tracks"], "media_H": ps[name]["mean_home"], "media_A": ps[name]["mean_away"], "menos_22_pct": ps[name]["less_22_pct"], "mas_22_pct": ps[name]["more_22_pct"], "min": ps[name]["tracks"]["min"], "max": ps[name]["tracks"]["max"]} for name in main]),
        "La continuidad se describe por rachas consecutivas con posición de cada slot, sin unir huecos. La fragmentación observada no identifica por sí sola cambios de identidad: puede reflejar oclusión, salida de cámara o un track diferente. Dos tracks próximos tampoco prueban duplicación.\n"]
    slots = pd.read_csv(out / "tactical_slots.csv")
    for name in main:
        selected = slots[slots.period.eq(name)].sort_values("active_runs", ascending=False).head(10)
        sections += [f"### Slots con más rachas — {name}\n", table(selected[["slot", "team", "role", "coverage_pct", "active_runs", "median_active_s", "max_active_s"]].to_dict("records"))]
    sections += ["La tabla completa por slot y bloque está en tactical_slots.csv; los intervalos largos sin un slot incluyen tiempo en el que ese track no estaba activo y no deben interpretarse como una persona perdida durante todo el hueco.\n",
        "![Disponibilidad por slot y minuto](figures/tactical_slot_coverage.png)\n",
        "![Recuentos y ausencias](figures/tactical_counts_gaps.png)\n",
        "## 8. Balón: ausencias, dinámica y proximidad\n",
        table([{"periodo": name, "rachas_sin_balon": ps[name]["ball_absence"]["runs"], "mediana_s": ps[name]["ball_absence"]["length_s"].get("p50"), "p95_s": ps[name]["ball_absence"]["length_s"].get("p95"), "max_s": ps[name]["ball_absence"]["length_s"].get("max"), "frames_en_huecos_ge5s_pct": ps[name]["ball_absence"]["frames_in_gaps_ge_5s_pct"]} for name in main]),
        "Las ausencias incluyen paradas y periodos sin observación; no se puede deducir balón muerto de un NaN. Las rachas están truncadas por el bloque analizado. Completar automáticamente huecos largos por interpolación puede crear trayectorias ficticias.\n",
        table([{"periodo": name, "pares_balon": ps[name]["ball_step_m"]["n"], "salto_mediana_m": ps[name]["ball_step_m"].get("p50"), "salto_p99_m": ps[name]["ball_step_m"].get("p99"), "salto_max_m": ps[name]["ball_step_m"].get("max"), "saltos_gt5m": ps[name]["ball_step_over_5m"], "dist_cercano_mediana_m": ps[name]["nearest_player_m"].get("p50"), "dist_cercano_p95_m": ps[name]["nearest_player_m"].get("p95")} for name in main]),
        "Las distancias al más cercano se calculan solo donde hay balón y al menos un jugador observado. Proximidad no equivale a control. Un balón aéreo proyectado sobre el plano del césped puede mostrar posiciones y velocidades aparentes distorsionadas. Los umbrales descriptivos 1 m/frame de jugador y 5 m/frame de balón no se han usado para borrar datos ni ajustar un modelo.\n",
        "![Geometría y continuidad del balón](figures/tactical_ball_geometry.png)\n",
        "## 9. Coordenadas, inmovilidad y candidatos a anomalía\n",
        table([{"periodo": name, "obs_jugador": ps[name]["player_observations"], "obs_fuera_campo": ps[name]["outside_player_observations"], "frames_balon_fuera": ps[name]["outside_ball_frames"], "frames_par_mismo_equipo_lt025m": ps[name]["same_team_pair_lt_025m_frames"], "pct_par_lt025m": ps[name]["same_team_pair_lt_025m_pct"], "saltos_jugador_gt1m": ps[name]["player_step_over_1m"], "pares_jugador": ps[name]["player_step_m"]["n"], "pares_xy_identicos": ps[name]["identical_player_adjacent_pairs"]} for name in main]),
        "Fuera de [0,1] puede ser una salida real del campo o una extrapolación/error; no se recorta. Cercanía <0,25 m entre tracks del mismo equipo es un candidato de revisión, no una etiqueta de duplicado. Coordenadas consecutivas idénticas pueden ser inmovilidad, redondeo o congelación del tracking; no se atribuye automáticamente una causa.\n",
        table([{"periodo": name, "objeto": obj, "eje": axis, **ps[name][obj][axis]} for name in main for obj in ["player_coordinates", "ball_coordinates"] for axis in ["x", "y"]]),
        "## 10. Velocidades y discrepancia de unidades\n",
        "El XML declara kph y DATA.md describe m/s. Para investigarlo se comparan muestras no solapadas de un segundo: trayectoria observada durante los 26 puntos, desplazamiento de 1 a 10 m/s y rectitud ≥0,9 (distancia neta / longitud recorrida). Se compara s en el centro de la ventana con desplazamiento neto/1 s. El filtro es diagnóstico, no una depuración ni selección para entrenar.\n",
        table(tactical["speed_unit_diagnostic"] + broadcast["speed_unit_diagnostic"]),
        "Una razón s/v cercana a 1 favorece interpretación m/s; cercana a 3,6 favorecería km/h. El contraste observado favorece m/s en las partes. Errores menores bajo una hipótesis son evidencia de coherencia interna, no confirmación del proveedor: suavizado, discretización e identidad cambiante afectan el contraste. Se conservan los valores originales y no se resuelve la discrepancia por decreto. Este diagnóstico usa jugadores; no confirma independientemente la unidad del canal escaso de velocidad del balón.\n",
        table([{"periodo": name, "objeto": key, **ps[name][key]} for name in main for key in ["player_s_raw", "ball_s_raw"]]),
        "## 11. Homografías\n",
        table([{"fuente": s["source"], "periodo": name, "valid_pct": p["coverage"]["homography"], "error_mediana": p["reprojection_error"].get("p50"), "error_p95": p["reprojection_error"].get("p95"), "error_max": p["reprojection_error"].get("max"), "rachas_invalidas": p["homography_invalid_runs"], "racha_invalida_max_s": p["homography_invalid_max_s"]} for s in results for name, p in s["periods"].items() if name in main + ["video_completo"]]),
        "Se verifica finitud y se calcula condicionamiento numérico. La escala y las coordenadas elegidas influyen en ese número, por lo que no se usa un umbral para declarar incorrecta una matriz. El error de reproyección no declara unidad en el CSV: se informa en su escala original. No hay referencias verdaderas para medir error absoluto de posiciones o calibración.\n",
        "![Distribución acumulada de error de homografía](figures/tactical_homography.png)\n",
        "## 12. Smart Tagging\n",
        f"Hay {tactical['events']['count']} intervalos; {tactical['events']['invalid_intervals']} con límites inválidos; máximo de {tactical['events']['max_simultaneous']} intervalos simultáneos.\n",
        table([{"codigo": code, "n": tactical["events"]["codes"][code], **values} for code, values in tactical["events"]["durations_by_code"].items()]),
        "Distribuciones de etiquetas; `<absent>` significa que ese atributo no aplica/no aparece en ese evento, no un valor imputado.\n"]
    for group, values in tactical["events"]["groups"].items():
        sections += [f"### {group}\n", table([{"valor": key, "eventos": val} for key, val in values.items()])]
    sections += ["La cobertura de tags se calcula por unión temporal; sumar duraciones de eventos produce doble conteo. Tags de ambos equipos coexisten en aproximadamente el 42 % de frames de cada parte: ataque y defensa pueden describir la misma jugada. No se interpreta Team como poseedor ni cada evento como posesión. Los saques sin equipo y las diferencias entre instante de reanudación e intervalo de cinco segundos requieren especial cuidado.\n",
        "## 13. Qué aporta broadcast y qué queda sin resolver\n",
        "![Disponibilidad temporal broadcast](figures/broadcast_timeline.png)\n",
        table([{"señal": key, "broadcast_video_completo_pct": value} for key, value in broadcast["periods"]["video_completo"]["coverage"].items()]),
        "Las dos fuentes no pueden fusionarse por índice de fila ni por un único desfase asumido. No se ha estimado sincronización, recuperación de balón entre cámaras ni ganancia de complementariedad. Las coberturas globales de broadcast incluyen cortes y repeticiones; no son comparables directamente con solo las partes de la táctica.\n",
        "### Diagnósticos adicionales de broadcast\n",
        "![Recuentos y ausencias broadcast](figures/broadcast_counts_gaps.png)\n",
        "![Disponibilidad por slot broadcast](figures/broadcast_slot_coverage.png)\n",
        "![Geometría del balón broadcast](figures/broadcast_ball_geometry.png)\n",
        "![Error de homografía broadcast](figures/broadcast_homography.png)\n",
        "## 14. Implicaciones para posesión y plan de experimentos\n",
        table([
            {"prioridad": 1, "observacion": "Balón falta en ~41 % de los bloques oficiales", "experimento": "Modelo temporal con incertidumbre; detección alternativa del balón en muestras automáticas", "verificacion": "Cobertura y coherencia fuera del GT; evaluación en otros partidos permitidos"},
            {"prioridad": 2, "observacion": "11+11 no está garantizado y hay tracks cercanos", "experimento": "Control de duplicados y consistencia de equipo cerca de la acción", "verificacion": "No aumentar falsos tracks ni cambios espurios"},
            {"prioridad": 3, "observacion": "H existe pero su precisión no se ha certificado", "experimento": "Superposición automática y calibración alternativa donde se justifique", "verificacion": "Referencias geométricas independientes; estabilidad temporal"},
            {"prioridad": 4, "observacion": "Eventos parciales y solapados", "experimento": "Combinar evidencias ofensivas/defensivas y reanudaciones", "verificacion": "No copiar tags como etiquetas ni confundir falta de tag con balón muerto"},
            {"prioridad": 5, "observacion": "Velocidad incompleta y unidad inconsistente", "experimento": "Resolver unidad y derivar movimiento con máscaras de validez", "verificacion": "No cruzar huecos ni identidades sin evidencia"}]),
        "El objetivo es H/A/D por frame, con X para abstención. La posesión persiste durante balón suelto hasta control rival; proximidad sola es insuficiente. spellF1 evalúa segmentos y penaliza fragmentación; macroF1 complementa la perspectiva por frame. X se rellena temporalmente para spellF1, pero cuenta como fallo en macroF1.\n",
        "Los cinco minutos públicos solo comprueban pipeline, alineación y formato: no se usan para entrenar ni ajustar. No se crean etiquetas manuales o reglas ligadas a instantes concretos de este partido. Este EDA no mide la exactitud real de posiciones ni de equipos y no demuestra que un detector alternativo mejore posesión.\n",
        "## 15. Reproducción y entregables\n",
        "Desde la raíz del repositorio, con PowerShell:\n\n```powershell\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements-eda.txt\n.\\.venv\\Scripts\\python.exe starter\\eda_initial_data.py --data-dir '..\\More Than A Hack 2027 - Initial Challenge - Dataset' --out evaluation\\reports\\eda-inicial\n```\n",
        "Requisitos: archivos originales en la ruta indicada; no credenciales ni API. La ruta es un parámetro; no está incrustada en el script. Se preserva la auditoría anterior de tracking-coverage como referencia independiente.\n",
        "- INFORME_EDA.md: informe completo.\n- summary.json: estadísticas agregadas y entorno.\n- inventory.csv: inventario y tamaños.\n- input_sha256.csv: huellas de los archivos no audiovisuales.\n- tactical_slots.csv / broadcast_slots.csv: agregados por slot y periodo.\n- *_minutes.csv: agregados por minuto.\n- *_gap_lengths.csv: duración de rachas sin balón por bloque (sin coordenadas ni instantes).\n- *_speed_units.csv: contraste de unidades.\n- figures/: gráficos PNG, sin fotogramas de vídeo ni trayectorias individuales.\n",
        "No publicar automáticamente el informe o agregados: son resultados locales del dataset licenciado. No se escriben exports de tracking por frame ni se redistribuyen vídeos. La carpeta de salida ya está ignorada por Git.\n",
        "Fuentes internas: DATA.md, CHALLENGE.md, EVALUATION.md, RULES.md, SUBMISSION.md, games.py; ATD XML/TXT, homografías, Smart Tagging y cabeceras de vídeos entregados. No se ejecutó entrenamiento ni inferencia de detectores.\n"]
    (out / "INFORME_EDA.md").write_text("\n".join(sections), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("evaluation/reports/eda-inicial"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    inventory = [{"file": str(p.relative_to(args.data_dir)), "bytes": p.stat().st_size,
                  "MiB": p.stat().st_size / 1024 ** 2} for p in sorted(args.data_dir.rglob("*")) if p.is_file()]
    fingerprints = []
    for p in sorted(args.data_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() != ".mp4":
            with p.open("rb") as handle:
                fingerprints.append({"file": str(p.relative_to(args.data_dir)), "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
    pd.DataFrame(fingerprints).to_csv(args.out / "input_sha256.csv", index=False)
    pd.DataFrame(inventory).to_csv(args.out / "inventory.csv", index=False)
    environment = {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "matplotlib", "opencv-python-headless"]}}
    results = [analyze(source, args.data_dir, args.out) for source in ["tactical", "broadcast"]]
    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(type(value))
    (args.out / "summary.json").write_text(json.dumps({"environment": environment, "sources": results}, default=convert, indent=2, allow_nan=False), encoding="utf-8")
    build_report(results, inventory, environment, args.out)
    print(f"Report: {args.out / 'INFORME_EDA.md'}", flush=True)


if __name__ == "__main__":
    main()
