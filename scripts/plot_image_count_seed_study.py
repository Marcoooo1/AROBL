
#!/usr/bin/env python3
"""Auswertung der DTU-Bildanzahl-Testreihe mit drei Seeds."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STUDY = (
    ROOT / "outputs" / "image_count_seed_study"
)

SCENES = ("scan24", "scan37", "scan55")
COUNTS = (10, 15, 20, 25, 30)
SEEDS = (0, 1, 2)

GEOMETRY = (
    ("accuracy", "Accuracy (mm)", 1),
    ("completeness", "Completeness (mm)", 1),
    ("chamfer", "Chamfer (mm)", 1),
    ("f_score", "F-Score bei 2 mm (%)", 100),
)

RENDERING = (
    ("psnr", "PSNR (dB)", 1),
    ("ssim", "SSIM", 1),
    ("lpips", "LPIPS", 1),
)

RUNTIME = (
    ("train_s", "Trainingsdauer (s)", 1),
    ("total_s", "Gesamtdauer (s)", 1),
)

MESH = (
    ("n_vertices", "Vertices", 1),
    ("n_faces", "Faces", 1),
)

ALL_METRICS = GEOMETRY + RENDERING + RUNTIME + MESH

RENDER_KEYS = {
    metric for metric, _, _ in RENDERING
}


# --------------------------------------------------
# Read experiment results
# --------------------------------------------------

def read_runs(path):

    if not path.is_file():
        raise SystemExit(
            f"Ergebnisdatei fehlt: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "exp", "scene", "n", "seed", "status"
    }

    if missing := required - set(df.columns):
        raise ValueError(
            f"Fehlende Spalten: {sorted(missing)}"
        )

    # Keep the most recent record per experiment.
    if "updated_utc" in df:
        df = df.sort_values(
            "updated_utc",
            kind="stable"
        )

    df = df.drop_duplicates(
        "exp",
        keep="last"
    ).copy()

    df["n"] = pd.to_numeric(
        df["n"],
        errors="coerce"
    )

    df["seed"] = pd.to_numeric(
        df["seed"],
        errors="coerce"
    )

    # Convert metrics to numeric values.
    for metric, _, _ in ALL_METRICS:

        if metric not in df:
            df[metric] = np.nan

        df[metric] = pd.to_numeric(
            df[metric],
            errors="coerce"
        )

    # Only use selected scenes, counts and seeds.
    df = df[
        df["scene"].isin(SCENES)
        & df["n"].isin(COUNTS)
        & df["seed"].isin(SEEDS)
    ].copy()

    if df.empty:
        raise SystemExit(
            "Keine passenden Datensaetze gefunden."
        )

    # Rendering requires independent evaluation images.
    independent = (
        df.get(
            "independent_eval",
            pd.Series(False, index=df.index)
        )
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    overlap = pd.to_numeric(
        df.get(
            "test_overlap",
            pd.Series(np.nan, index=df.index)
        ),
        errors="coerce"
    )

    good = df["status"].eq("ok")

    # Geometry is valid only when the required
    # metrics are available.
    df["valid_geometry"] = (
        good
        & df[
            [
                "accuracy",
                "completeness",
                "chamfer",
                "f_score"
            ]
        ].notna().all(axis=1)
    )

    # Rendering is valid only for independent
    # test views.
    df["valid_rendering"] = (
        good
        & independent
        & overlap.eq(0)
        & df[
            ["psnr", "ssim", "lpips"]
        ].notna().all(axis=1)
    )

    return df


# --------------------------------------------------
# Calculate seed means and standard deviations
# --------------------------------------------------

def aggregate_seeds(df):

    rows = []

    for scene in SCENES:
        for n in COUNTS:

            part = df[
                (df["scene"] == scene)
                & (df["n"] == n)
            ]

            geom = part[
                part["valid_geometry"]
            ].set_index("seed")

            rend = part[
                part["valid_rendering"]
            ].set_index("seed")

            row = {
                "scene": scene,
                "n": n,
                "seeds_geometry_ok": len(geom),
                "seeds_rendering_ok": len(rend),
            }

            for metric, _, _ in ALL_METRICS:

                source = (
                    rend
                    if metric in RENDER_KEYS
                    else geom
                )

                values = source[
                    metric
                ].reindex(SEEDS)

                # Mean and SD only for all three seeds.
                if values.notna().all():

                    row[f"mean_{metric}"] = (
                        values.mean()
                    )

                    row[f"std_{metric}"] = (
                        values.std(ddof=1)
                    )

                else:

                    row[f"mean_{metric}"] = np.nan
                    row[f"std_{metric}"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# Summary across the three DTU scenes
# --------------------------------------------------

def summarize_counts(stats):

    rows = []

    for n in COUNTS:

        part = stats[
            stats["n"] == n
        ]

        row = {
            "n": n,

            "scenes_geometry_complete": int(
                (
                    part["seeds_geometry_ok"] == 3
                ).sum()
            ),

            "scenes_rendering_complete": int(
                (
                    part["seeds_rendering_ok"] == 3
                ).sum()
            ),
        }

        for metric, _, _ in ALL_METRICS:

            values = part[
                f"mean_{metric}"
            ].dropna()

            # Cross-scene mean only if all three
            # scenes have complete seed results.
            row[f"mean_{metric}"] = (
                values.mean()
                if len(values) == 3
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# General plot function
# --------------------------------------------------

def make_panels(metrics, title, path, series):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        squeeze=False
    )

    for ax, (metric, label, factor) in zip(
        axes.flat,
        metrics
    ):

        for name, values in series(metric):

            ax.plot(
                COUNTS,
                np.asarray(
                    values,
                    dtype=float
                ) * factor,
                marker="o",
                linewidth=1.6,
                label=name
            )

        ax.set_title(label)
        ax.set_xlabel("Trainingsbilder")
        ax.set_xticks(COUNTS)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    # Hide unused subplots.
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=14
    )

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# --------------------------------------------------
# Main plots: mean of three seeds per scene
# --------------------------------------------------

def mean_scene_plots(stats, out):

    def series(metric):

        for scene in SCENES:

            part = stats[
                stats["scene"] == scene
            ].set_index("n").reindex(COUNTS)

            yield (
                f"{scene.replace('scan', 'Scan ')} "
                "",

                part[
                    f"mean_{metric}"
                ].to_numpy()
            )

    configurations = (
        (
            "geometry",
            GEOMETRY,
            "DTU Geometrie: Mittelwerte aus 3 Seeds je Szene"
        ),
        (
            "rendering",
            RENDERING,
            "Rendering: Mittelwerte aus 3 Seeds je Szene"
        ),
        (
            "runtime",
            RUNTIME,
            "Laufzeiten: Mittelwerte aus 3 Seeds je Szene"
        ),
        (
            "mesh_size",
            MESH,
            "Meshgroesse: Mittelwerte aus 3 Seeds je Szene"
        ),
    )

    for name, metrics, title in configurations:

        make_panels(
            metrics,
            title,
            out / f"{name}.png",
            series
        )


# --------------------------------------------------
# Individual comparisons
# --------------------------------------------------

def individual_plots(df, out):

    def plot_group(
        fixed_col,
        fixed_value,
        vary_col,
        vary_values,
        label,
        folder
    ):

        def series(metric):

            flag = (
                "valid_rendering"
                if metric in RENDER_KEYS
                else "valid_geometry"
            )

            for value in vary_values:

                subset = df[
                    (df[fixed_col] == fixed_value)
                    & (df[vary_col] == value)
                    & df[flag]
                ].set_index("n").reindex(COUNTS)

                if vary_col == "seed":
                    name = f"Seed {value}"
                else:
                    name = f"Scan {str(value)[4:]}"

                yield (
                    name,
                    subset[metric].to_numpy()
                )

        configurations = (
            ("geometry", GEOMETRY),
            ("rendering", RENDERING),
        )

        for kind, metrics in configurations:

            if vary_col == "seed":
                filename = (
                    f"{fixed_value}_{kind}_seeds.png"
                )
            else:
                filename = (
                    f"seed{fixed_value}_{kind}_scans.png"
                )

            make_panels(
                metrics,
                (
                    f"{label}: {kind} "
                    "nach Trainingsbildanzahl"
                ),
                out / folder / filename,
                series
            )

    # A: Compare three seeds within each scan.
    for scene in SCENES:

        plot_group(
            fixed_col="scene",
            fixed_value=scene,
            vary_col="seed",
            vary_values=SEEDS,
            label=scene.upper(),
            folder="per_scan"
        )

    # B: Compare three scans within each seed.
    for seed in SEEDS:

        plot_group(
            fixed_col="seed",
            fixed_value=seed,
            vary_col="scene",
            vary_values=SCENES,
            label=f"Seed {seed}",
            folder="per_seed"
        )


# --------------------------------------------------
# Seed variation: mean and standard deviation
# --------------------------------------------------

def errorbar_plot(
    stats,
    metric,
    ylabel,
    path,
    factor=1
):

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for scene in SCENES:

        part = stats[
            stats["scene"] == scene
        ].set_index("n").reindex(COUNTS)

        means = (
            part[f"mean_{metric}"]
            .to_numpy(dtype=float)
            * factor
        )

        deviations = (
            part[f"std_{metric}"]
            .to_numpy(dtype=float)
            * factor
        )

        ax.errorbar(
            COUNTS,
            means,
            yerr=deviations,
            marker="o",
            capsize=4,
            label=scene.replace("scan", "Scan ")
        )

    ax.set_title(
        f"{ylabel}: Mittelwert und "
        "Standardabweichung aus 3 Seeds"
    )

    ax.set_xlabel("Trainingsbilder")
    ax.set_ylabel(ylabel)
    ax.set_xticks(COUNTS)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# --------------------------------------------------
# Summary table as PNG
# --------------------------------------------------

def overview_table(summary, path):

    def number(value, digits):

        return (
            f"{value:.{digits}f}"
            if pd.notna(value)
            else "-"
        )

    cells = []

    for _, row in summary.iterrows():

        cells.append([
            int(row["n"]),

            f"{int(row['scenes_geometry_complete'])}/3",

            number(
                row["mean_chamfer"],
                3
            ),

            number(
                row["mean_f_score"] * 100,
                1
            ),

            number(
                row["mean_psnr"],
                2
            ),

            number(
                row["mean_train_s"],
                0
            ),
        ])

    fig, ax = plt.subplots(
        figsize=(11, 3.5)
    )

    ax.axis("off")

    table = ax.table(
        cellText=cells,
        colLabels=[
            "Bilder",
            "Szenen OK",
            "Chamfer (mm)",
            "F-Score (%)",
            "PSNR (dB)",
            "Training (s)"
        ],
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    ax.set_title(
        "Vergleich nach Bildanzahl: "
        "vollstaendige Seed- und Szenenmittel",
        pad=12
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--study-dir",
        type=Path,
        default=DEFAULT_STUDY
    )

    args = parser.parse_args()

    study = args.study_dir.resolve()

    out = study / "seed_analysis"
    plots = out / "plots"
    tables = out / "tables"

    plots.mkdir(
        parents=True,
        exist_ok=True
    )

    tables.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read the original experiment results.
    df = read_runs(
        study / "metrics.csv"
    )

    # Calculate statistics.
    stats = aggregate_seeds(df)
    summary = summarize_counts(stats)

    # ------------------------------------------
    # Export tables
    # ------------------------------------------

    df.drop(
        columns=[
            "valid_geometry",
            "valid_rendering"
        ]
    ).sort_values(
        ["scene", "n", "seed"]
    ).to_csv(
        tables / "all_seed_runs.csv",
        index=False
    )

    stats.to_csv(
        out / "seed_variation.csv",
        index=False,
        float_format="%.6f"
    )

    stats.to_csv(
        tables / "mean_by_scene_and_n.csv",
        index=False,
        float_format="%.6f"
    )

    summary.to_csv(
        tables / "summary_by_n.csv",
        index=False,
        float_format="%.6f"
    )

    # ------------------------------------------
    # Mean plots: 3 scenes
    # ------------------------------------------

    mean_scene_plots(
        stats,
        plots
    )

    # ------------------------------------------
    # Individual comparisons
    # ------------------------------------------

    individual_plots(
        df,
        plots
    )

    # ------------------------------------------
    # Standard deviation plots
    # ------------------------------------------

    errorbar_plot(
        stats,
        "chamfer",
        "Chamfer (mm)",
        plots / "chamfer_seed_spread.png"
    )

    errorbar_plot(
        stats,
        "f_score",
        "F-Score bei 2 mm (%)",
        plots / "fscore_seed_spread.png",
        factor=100
    )

    # ------------------------------------------
    # Summary table
    # ------------------------------------------

    overview_table(
        summary,
        tables / "summary_by_n.png"
    )

    # ------------------------------------------
    # Terminal output
    # ------------------------------------------

    print(
        "\nMittelwerte nur bei jeweils "
        "3 erfolgreichen Seeds pro Szene."
    )

    print(
        summary[
            [
                "n",
                "scenes_geometry_complete",
                "mean_chamfer",
                "mean_f_score",
                "mean_psnr"
            ]
        ].to_string(
            index=False,
            na_rep="-",
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print(f"\nGrafiken: {plots}")
    print(f"Tabellen: {tables}")


if __name__ == "__main__":
    main()
