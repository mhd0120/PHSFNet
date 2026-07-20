"""Select and plot two real OCO-2 samples for an Introduction motivation figure.

Version 2 searches low/high observation-condition candidate pairs, scores them by
XCO2 similarity, state contrast, radiance distance, and SNR distance, writes the
Top-K candidates, and plots the selected pair.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BANDS = [
    ("o2", "O$_2$ A-Band"),
    ("weak_co2", "Weak CO$_2$ Band"),
    ("strong_co2", "Strong CO$_2$ Band"),
]

SPLIT_YEARS = {
    "train": ["2020", "2021"],
    "val": ["2022"],
    "test": ["2023"],
}

GROUP_CANDIDATES = [
    ("solar_zenith_angle", ["solar_zenith_angle", "solar_zenith", "sza"]),
    ("viewing_zenith_angle", ["viewing_zenith_angle", "view_zenith", "vza", "zenith"]),
    ("surface_albedo", ["surface_albedo", "albedo", "surface_albedo_o2", "surface_albedo_weak_co2"]),
]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def configure_fonts():
    names = {font.name for font in fm.fontManager.ttflist}
    has_times = "Times New Roman" in names
    if not has_times:
        print("WARNING: Times New Roman was not found by matplotlib; continuing with fallback serif font.")
    plt.rcParams.update({
        "font.family": "Times New Roman" if has_times else "serif",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
    })


def load_split(split):
    frames = []
    paths = []
    for year in SPLIT_YEARS[split]:
        path = PROJECT_ROOT / "fast_read" / "full" / f"training_data_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        frames.append(pd.read_parquet(path))
        paths.append(path)
    return pd.concat(frames, axis=0), paths


def feature_columns(df):
    if not isinstance(df.columns, pd.MultiIndex):
        return []
    if "features" not in df.columns.get_level_values(0):
        return []
    return list(df["features"].columns)


def resolve_group_variable(df, requested):
    features = feature_columns(df)
    lower_to_real = {str(name).lower(): name for name in features}
    checked = []

    def find_any(names):
        for name in names:
            if name.lower() in lower_to_real:
                return lower_to_real[name.lower()]
        return None

    if requested:
        requested_key = requested.lower()
        checked.append(requested)
        if requested_key in lower_to_real:
            return lower_to_real[requested_key], checked
        for canonical, aliases in GROUP_CANDIDATES:
            if requested_key == canonical or requested_key in aliases:
                checked.extend([name for name in aliases if name not in checked])
                found = find_any(aliases)
                if found is not None:
                    return found, checked

    for _, aliases in GROUP_CANDIDATES:
        checked.extend([name for name in aliases if name not in checked])
        found = find_any(aliases)
        if found is not None:
            return found, checked

    raise KeyError("No usable group variable found. Checked: " + ", ".join(checked))


def xco2_ppm(df):
    values = df[("labels", "xco2")].astype(float).to_numpy()
    if np.nanmedian(np.abs(values)) < 1.0:
        return values * 1e6
    return values


def fill_nan_1d(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.all():
        return values
    if valid.sum() == 0:
        return np.zeros_like(values)
    if valid.sum() == 1:
        return np.full_like(values, values[valid][0])
    x = np.arange(values.size)
    out = values.copy()
    out[~valid] = np.interp(x[~valid], x[valid], values[valid])
    return out


def sample_array(df, row_pos, prefix, band):
    values = df.iloc[row_pos][f"{prefix}_{band}"].to_numpy(dtype=float)
    bad_col = f"bad_sample_list_{band}"
    if bad_col in df.columns.get_level_values(0):
        bad = df.iloc[row_pos][bad_col].to_numpy()
        values = values.copy()
        values[bad != 0] = np.nan
    return fill_nan_1d(values)


def concatenated_sample(df, row_pos, prefix):
    return np.concatenate([sample_array(df, row_pos, prefix, band) for band, _ in BANDS])


def wavelength_axis(df, band, use_wavelength):
    n = df[f"radiances_{band}"].shape[1]
    if not use_wavelength or f"wavelengths_{band}" not in df.columns.get_level_values(0):
        return np.arange(1, n + 1), "Wavelength index"
    wavelengths = df[f"wavelengths_{band}"].to_numpy(dtype=float)
    return np.nanmedian(wavelengths, axis=0), "Wavelength"


def finite_range(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.min()), float(values.max())


def minmax_normalize(values):
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def select_candidates(df, window, fallback_window, min_samples, group_variable, group_quantile, top_k):
    all_xco2 = xco2_ppm(df)
    median_xco2 = float(np.nanmedian(all_xco2))

    used_window = window
    selected = np.where(np.abs(all_xco2 - median_xco2) <= used_window)[0]
    if selected.size < min_samples and fallback_window > used_window:
        used_window = fallback_window
        selected = np.where(np.abs(all_xco2 - median_xco2) <= used_window)[0]
    if selected.size < 2:
        raise ValueError(f"Only {selected.size} samples selected. Increase --xco2_window.")

    group_values_all = df[("features", group_variable)].astype(float).to_numpy()
    group_values = group_values_all[selected]
    low_thr = float(np.nanquantile(group_values, group_quantile))
    high_thr = float(np.nanquantile(group_values, 1.0 - group_quantile))
    low_candidates = selected[group_values <= low_thr]
    high_candidates = selected[group_values >= high_thr]
    if low_candidates.size == 0 or high_candidates.size == 0:
        raise ValueError("Low/high candidate groups are empty. Adjust --group_quantile.")

    spectra_cache = {}

    def get_cached(row_pos, prefix):
        key = (int(row_pos), prefix)
        if key not in spectra_cache:
            spectra_cache[key] = concatenated_sample(df, int(row_pos), prefix)
        return spectra_cache[key]

    rows = []
    for low_pos in low_candidates:
        low_pos = int(low_pos)
        low_rad = get_cached(low_pos, "radiances")
        low_snr = get_cached(low_pos, "snrs")
        for high_pos in high_candidates:
            high_pos = int(high_pos)
            high_rad = get_cached(high_pos, "radiances")
            high_snr = get_cached(high_pos, "snrs")
            rows.append({
                "low_row": low_pos,
                "high_row": high_pos,
                "low_sounding_id": df.index[low_pos],
                "high_sounding_id": df.index[high_pos],
                "xco2_low": float(all_xco2[low_pos]),
                "xco2_high": float(all_xco2[high_pos]),
                "abs_delta_xco2": float(abs(all_xco2[low_pos] - all_xco2[high_pos])),
                "group_variable": group_variable,
                "groupvar_low": float(group_values_all[low_pos]),
                "groupvar_high": float(group_values_all[high_pos]),
                "abs_delta_groupvar": float(abs(group_values_all[low_pos] - group_values_all[high_pos])),
                "radiance_distance": float(np.mean(np.abs(low_rad - high_rad))),
                "snr_distance": float(np.mean(np.abs(low_snr - high_snr))),
            })

    table = pd.DataFrame(rows)
    table["n_group"] = minmax_normalize(table["abs_delta_groupvar"])
    table["n_rad"] = minmax_normalize(table["radiance_distance"])
    table["n_snr"] = minmax_normalize(table["snr_distance"])
    table["n_xco2"] = minmax_normalize(table["abs_delta_xco2"])
    table["score"] = (
        1.5 * table["n_group"]
        + 1.0 * table["n_rad"]
        + 0.5 * table["n_snr"]
        - 1.5 * table["n_xco2"]
    )
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    return {
        "median_xco2": median_xco2,
        "window": used_window,
        "selected": selected,
        "low_threshold": low_thr,
        "high_threshold": high_thr,
        "low_group_size": int(low_candidates.size),
        "high_group_size": int(high_candidates.size),
        "xco2": all_xco2,
        "group_values": group_values_all,
        "candidates": table.head(top_k).copy(),
        "all_candidates": table,
    }


def write_candidates(path, candidates):
    fields = [
        "rank", "low_row", "high_row", "low_sounding_id", "high_sounding_id",
        "xco2_low", "xco2_high", "abs_delta_xco2", "group_variable",
        "groupvar_low", "groupvar_high", "abs_delta_groupvar",
        "radiance_distance", "snr_distance", "score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates[fields].to_csv(path, index=False)


def write_state_comparison(path, df, low_pos, high_pos):
    rows = []
    for name in feature_columns(df):
        low_value = df.iloc[low_pos][("features", name)]
        high_value = df.iloc[high_pos][("features", name)]
        try:
            low_float = float(low_value)
            high_float = float(high_value)
            diff = abs(low_float - high_float)
        except (TypeError, ValueError):
            low_float = low_value
            high_float = high_value
            diff = ""
        rows.append({
            "variable_name": name,
            "low_sample_value": low_float,
            "high_sample_value": high_float,
            "abs_difference": diff,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variable_name", "low_sample_value", "high_sample_value", "abs_difference"])
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_info(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pair(df, low_pos, high_pos, group_variable, output_png, output_pdf, radiance_lw, snr_alpha, snr_outline_lw, use_wavelength):
    low_color = "#2f5f8f"
    high_color = "#b65a45"
    low_snr_color = "#8db8d8"
    high_snr_color = "#d59a8a"

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    letters = ["a", "b", "c"]
    band_stats = []

    for i, (band, band_title) in enumerate(BANDS):
        ax = axes[i]
        ax2 = ax.twinx()
        x, x_label = wavelength_axis(df, band, use_wavelength)

        rad_low = sample_array(df, low_pos, "radiances", band)
        rad_high = sample_array(df, high_pos, "radiances", band)
        snr_low = sample_array(df, low_pos, "snrs", band)
        snr_high = sample_array(df, high_pos, "snrs", band)

        ax2.fill_between(x, 0, snr_low, color=low_snr_color, alpha=snr_alpha, linewidth=0)
        ax2.fill_between(x, 0, snr_high, color=high_snr_color, alpha=snr_alpha, linewidth=0)
        ax2.plot(x, snr_low, color=low_snr_color, alpha=0.7, linewidth=snr_outline_lw)
        ax2.plot(x, snr_high, color=high_snr_color, alpha=0.7, linewidth=snr_outline_lw)
        ax.plot(x, rad_low, color=low_color, lw=radiance_lw, label="Sample A radiance")
        ax.plot(x, rad_high, color=high_color, lw=radiance_lw, label="Sample B radiance")

        ax.set_ylabel("Raw radiance")
        ax2.set_ylabel("Signal to noise ratio")
        ax.grid(True, color="0.88", linewidth=0.5)
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        ax2.set_ylim(bottom=0)
        if i == len(BANDS) - 1:
            ax.set_xlabel("")
            ax.text(
                1.0,
                -0.12,
                x_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
            )

        ax.text(
            0.5,
            -0.12,
            f"({letters[i]}) {band_title}",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

        band_stats.append({
            "band": band_title.replace("$", ""),
            "rad_low": finite_range(rad_low),
            "rad_high": finite_range(rad_high),
            "snr_low": finite_range(snr_low),
            "snr_high": finite_range(snr_high),
        })

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.10, right=0.88, top=0.94, bottom=0.12, hspace=0.48)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return band_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(SPLIT_YEARS), default="test")
    parser.add_argument("--xco2_window", type=float, default=0.5)
    parser.add_argument("--fallback_xco2_window", type=float, default=1.0)
    parser.add_argument("--min_samples", type=int, default=80)
    parser.add_argument("--group_variable", type=str, default="solar_zenith_angle")
    parser.add_argument("--group_quantile", type=float, default=0.25)
    parser.add_argument("--top_k_candidates", type=int, default=5)
    parser.add_argument("--candidate_rank", type=int, default=1, help="1-based rank from the candidate CSV to plot")
    parser.add_argument("--radiance_linewidth", type=float, default=1.8)
    parser.add_argument("--snr_alpha", type=float, default=0.07)
    parser.add_argument("--snr_outline_linewidth", type=float, default=0.8)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "figures")
    parser.add_argument("--use_wavelength", type=parse_bool, default=False)
    args = parser.parse_args()

    if not (0.0 < args.group_quantile < 0.5):
        raise ValueError("--group_quantile must be between 0 and 0.5")
    if args.candidate_rank < 1:
        raise ValueError("--candidate_rank must be 1 or greater")

    configure_fonts()
    df, data_paths = load_split(args.split)
    used_group, checked_fields = resolve_group_variable(df, args.group_variable)
    result = select_candidates(
        df=df,
        window=args.xco2_window,
        fallback_window=args.fallback_xco2_window,
        min_samples=args.min_samples,
        group_variable=used_group,
        group_quantile=args.group_quantile,
        top_k=max(args.top_k_candidates, args.candidate_rank),
    )

    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "introduction_two_sample_observation_patterns_final.png"
    pdf_path = output_dir / "introduction_two_sample_observation_patterns_final.pdf"
    info_path = output_dir / "introduction_two_sample_observation_patterns_final_info.txt"
    candidates_path = output_dir / "introduction_two_sample_candidates.csv"
    states_path = output_dir / "introduction_two_sample_state_comparison.csv"

    candidates = result["candidates"]
    write_candidates(candidates_path, candidates.head(args.top_k_candidates))

    if args.candidate_rank > len(candidates):
        raise ValueError(f"--candidate_rank {args.candidate_rank} is not available; only {len(candidates)} candidates were generated")
    selected_pair = candidates.iloc[args.candidate_rank - 1]
    low_pos = int(selected_pair["low_row"])
    high_pos = int(selected_pair["high_row"])

    state_df = write_state_comparison(states_path, df, low_pos, high_pos)
    band_stats = plot_pair(
        df=df,
        low_pos=low_pos,
        high_pos=high_pos,
        group_variable=used_group,
        output_png=png_path,
        output_pdf=pdf_path,
        radiance_lw=args.radiance_linewidth,
        snr_alpha=args.snr_alpha,
        snr_outline_lw=args.snr_outline_linewidth,
        use_wavelength=args.use_wavelength,
    )

    xco2_min = result["median_xco2"] - result["window"]
    xco2_max = result["median_xco2"] + result["window"]
    lines = [
        "Introduction two-sample observation-conditioned pattern figure, version 2",
        f"Split: {args.split}",
        "Data files: " + ", ".join(rel(path) for path in data_paths),
        f"XCO2 median: {result['median_xco2']:.4f} ppm",
        f"XCO2 window: [{xco2_min:.4f}, {xco2_max:.4f}] ppm, half-width={result['window']:.3f} ppm",
        f"Candidate samples in XCO2 window: {result['selected'].size}",
        f"Requested group variable: {args.group_variable}",
        f"Checked group fields: {', '.join(checked_fields)}",
        f"Used group variable: features/{used_group}",
        f"Group quantile: {args.group_quantile:.3f}",
        f"Low group threshold: <= {result['low_threshold']:.4f}",
        f"High group threshold: >= {result['high_threshold']:.4f}",
        f"Low group candidate samples: {result['low_group_size']}",
        f"High group candidate samples: {result['high_group_size']}",
        "",
        f"Top-{args.top_k_candidates} candidate sample pairs:",
    ]
    for _, row in candidates.head(args.top_k_candidates).iterrows():
        lines.append(
            f"rank {int(row['rank'])}: low {row['low_sounding_id']} / high {row['high_sounding_id']}; "
            f"delta XCO2={row['abs_delta_xco2']:.6f} ppm; "
            f"delta {used_group}={row['abs_delta_groupvar']:.4f}; "
            f"radiance_distance={row['radiance_distance']:.6g}; "
            f"snr_distance={row['snr_distance']:.6g}; score={row['score']:.4f}"
        )
    lines.extend([
        "",
        f"Selected plotting pair: rank {args.candidate_rank}",
        f"Low row/sounding id: {low_pos} / {df.index[low_pos]}",
        f"High row/sounding id: {high_pos} / {df.index[high_pos]}",
        f"Low XCO2: {result['xco2'][low_pos]:.4f} ppm",
        f"High XCO2: {result['xco2'][high_pos]:.4f} ppm",
        f"Low {used_group}: {result['group_values'][low_pos]:.4f}",
        f"High {used_group}: {result['group_values'][high_pos]:.4f}",
        "",
        "Auxiliary physical states with largest absolute differences:",
    ])
    numeric_states = state_df.copy()
    numeric_states["abs_difference"] = pd.to_numeric(numeric_states["abs_difference"], errors="coerce")
    for _, row in numeric_states.sort_values("abs_difference", ascending=False).head(8).iterrows():
        lines.append(
            f"- {row['variable_name']}: low={row['low_sample_value']}, "
            f"high={row['high_sample_value']}, abs_difference={row['abs_difference']:.6g}"
        )
    lines.append("")
    lines.append("Band value ranges for selected pair:")
    for stat in band_stats:
        lines.extend([
            f"- {stat['band']}:",
            f"  low radiance min/max: {stat['rad_low'][0]:.6g}, {stat['rad_low'][1]:.6g}",
            f"  high radiance min/max: {stat['rad_high'][0]:.6g}, {stat['rad_high'][1]:.6g}",
            f"  low SNR min/max: {stat['snr_low'][0]:.6g}, {stat['snr_low'][1]:.6g}",
            f"  high SNR min/max: {stat['snr_high'][0]:.6g}, {stat['snr_high'][1]:.6g}",
        ])
    lines.extend([
        "",
        "Message: Two samples with nearly identical XCO2 values can still exhibit different radiance and signal to noise ratio patterns under different observation conditions.",
    ])
    write_info(info_path, lines)

    print(f"XCO2 range: [{xco2_min:.4f}, {xco2_max:.4f}] ppm")
    print(f"Candidate samples in XCO2 window: {result['selected'].size}")
    print(f"Grouping variable: features/{used_group}")
    print(f"Top-{args.top_k_candidates} candidates saved: {candidates_path}")
    print(candidates[[
        "rank", "low_sounding_id", "high_sounding_id", "xco2_low", "xco2_high",
        "abs_delta_xco2", "groupvar_low", "groupvar_high",
        "abs_delta_groupvar", "radiance_distance", "snr_distance", "score",
    ]].head(args.top_k_candidates).to_string(index=False))
    print(f"Selected plotting pair: rank {args.candidate_rank}")
    print(f"State comparison saved: {states_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved info: {info_path}")


if __name__ == "__main__":
    main()
