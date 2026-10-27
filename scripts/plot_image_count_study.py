
#!/usr/bin/env python3
"""Grafiken und Tabellen fuer die DTU-Bildanzahl-Testreihe."""

from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Kein grafisches WSL-Fenster erforderlich
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs" / "image_count_study"

SCENES = ("scan24", "scan37", "scan55")
COUNTS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 49)

FIELDS = (
    "accuracy", "completeness", "chamfer", "f_score",
    "precision", "recall", "psnr", "ssim", "lpips",
    "train_s", "total_s", "n_vertices", "n_faces"
)


# --------------------------------------------------
# Daten einlesen
# --------------------------------------------------

def read_data(path):
    df = pd.read_csv(path)

    required = {"exp", "scene", "n", "status"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Fehlende CSV-Spalten: "
            f"{sorted(required - set(df.columns))}"
        )

    for key in FIELDS:
        if key not in df:
            df[key] = np.nan

        df[key] = pd.to_numeric(
            df[key], errors="coerce"
        )

    df["n"] = pd.to_numeric(
        df["n"], errors="coerce"
    )

    for key in ("independent_eval", "test_overlap"):
        if key not in df:
            df[key] = np.nan

    if "updated_utc" in df:
        df = df.sort_values(
            "updated_utc", kind="stable"
        )

    df = df.drop_duplicates("exp", keep="last")

    df = df[
        df["scene"].isin(SCENES)
        & df["n"].isin(COUNTS)
    ].copy()

    # Renderingmetriken nur auf unabhaengigen Testbildern.
    df["heldout"] = (
        df["independent_eval"]
        .astype(str).str.lower().eq("true")
        & pd.to_numeric(
            df["test_overlap"], errors="coerce"
        ).eq(0)
        & df["n"].le(40)
    )

    df["valid_geom"] = (
        df["status"].eq("ok")
        & df["chamfer"].notna()
        & df["f_score"].notna()
    )

    df["valid_render"] = (
        df["status"].eq("ok")
        & df["heldout"]
    )

    return df


# --------------------------------------------------
# Mittelwerte berechnen
# --------------------------------------------------

def metric_values(df, metric, n, render=False):
    eligible = (
        df["valid_render"]
        if render else df["valid_geom"]
    )

    subset = df[
        eligible & df["n"].eq(n)
    ]

    return subset[metric].dropna()


def summary_table(df):
    result = []

    for n in sorted(COUNTS, reverse=True):

        row = {
            "n": n,
            "n_geometry_ok": int(
                (df["valid_geom"] & df["n"].eq(n)).sum()
            ),
            "n_render_ok": int(
                (
                    df["valid_render"]
                    & df["n"].eq(n)
                    & df["psnr"].notna()
                ).sum()
            )
        }

        for metric in FIELDS:
            values = metric_values(
                df,
                metric,
                n,
                render=metric in ("psnr", "ssim", "lpips")
            )

            # Mittelwert nur bei drei gueltigen Szenen.
            if len(values) == len(SCENES):
                row[f"mean_{metric}"] = values.mean()
            else:
                row[f"mean_{metric}"] = np.nan

            if metric == "chamfer":
                row["std_chamfer"] = (
                    values.std(ddof=1)
                    if len(values) == len(SCENES)
                    else np.nan
                )

        result.append(row)

    return pd.DataFrame(result)


# --------------------------------------------------
# Diagramme erstellen
# --------------------------------------------------

def draw_metrics(df, summary, specs, filename, title):

    cols = 2 if len(specs) == 4 else len(specs)
    rows = math.ceil(len(specs) / cols)

    fig, axs = plt.subplots(
        rows, cols,
        figsize=(6.4 * cols, 4.0 * rows),
        squeeze=False
    )

    for ax, (metric, label, factor) in zip(
        axs.flat, specs
    ):

        render = metric in ("psnr", "ssim", "lpips")

        eligible = (
            df["valid_render"]
            if render else df["valid_geom"]
        )

        # Kurve fuer jede Szene.
        for scene in SCENES:
            values = []

            for n in COUNTS:

                found = df[
                    eligible
                    & df["scene"].eq(scene)
                    & df["n"].eq(n)
                ]

                if (
                    len(found)
                    and pd.notna(found[metric].iloc[0])
                ):
                    value = float(
                        found[metric].iloc[0]
                    ) * factor
                else:
                    value = np.nan

                values.append(value)

            ax.plot(
                COUNTS,
                values,
                marker="o",
                label=scene
            )

        # Mittelwert nur wenn alle Szenen vorhanden.
        means = (
            summary.set_index("n")
            .reindex(COUNTS)[f"mean_{metric}"]
            * factor
        )

        ax.plot(
            COUNTS,
            means,
            linestyle="--",
            color="black",
            linewidth=1.7,
            label="Mittel (nur 3/3)"
        )

        ax.set_title(label)
        ax.set_xlabel("Trainingsbilder")
        ax.set_xticks(COUNTS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in list(axs.flat)[len(specs):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# --------------------------------------------------
# Uebersichtstabelle als Bild
# --------------------------------------------------

def make_table_picture(summary, filename):

    def fmt(value, digits):
        if pd.notna(value):
            return f"{value:.{digits}f}"
        return "-"

    data = []

    for _, row in summary.iterrows():

        data.append([
            int(row["n"]),
            f'{int(row["n_geometry_ok"])}/3',
            fmt(row["mean_chamfer"], 3),
            fmt(row["mean_f_score"] * 100, 1),
            fmt(row["mean_psnr"], 2),
            fmt(row["mean_train_s"], 0)
        ])

    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=[
            "Bilder",
            "Geometrie",
            "Chamfer (mm)",
            "F-Score (%)",
            "PSNR (dB)",
            "Training (s)"
        ],
        loc="center",
        cellLoc="center",
        colLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)

    ax.set_title(
        "DTU Bildanzahl-Testreihe: Mittelwerte aus 3 Szenen",
        pad=18
    )

    fig.text(
        0.5, 0.05,
        "- = fehlende Daten oder weniger als 3 Szenen. "
        "PSNR nur auf unabhaengigen Testbildern.",
        ha="center",
        fontsize=9
    )

    fig.savefig(
        filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)


# --------------------------------------------------
# Hauptprogramm
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--study-dir",
        type=Path,
        default=DEFAULT
    )

    args = parser.parse_args()

    study = args.study_dir.resolve()
    path = study / "metrics.csv"

    if not path.is_file():
        raise SystemExit(f"Nicht gefunden: {path}")

    df = read_data(path)

    if df.empty:
        raise SystemExit(
            "Keine passenden Experimente gefunden."
        )

    plots = study / "plots"
    tables = study / "tables"

    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    summary = summary_table(df)

    # Einzelne Experimentdaten exportieren.
    df.drop(
        columns=[
            "heldout",
            "valid_geom",
            "valid_render"
        ]
    ).sort_values(
        ["scene", "n"],
        ascending=[True, False]
    ).to_csv(
        tables / "all_runs.csv",
        index=False
    )

    # Zusammenfassung nach Bildanzahl.
    summary.to_csv(
        tables / "summary_by_n.csv",
        index=False,
        float_format="%.5f"
    )

    # Geometrie.
    draw_metrics(
        df,
        summary,
        [
            (
                "accuracy",
                "Accuracy (mm), kleiner ist besser",
                1
            ),
            (
                "completeness",
                "Completeness (mm), kleiner ist besser",
                1
            ),
            (
                "chamfer",
                "Chamfer (mm), kleiner ist besser",
                1
            ),
            (
                "f_score",
                "F-Score bei 2 mm (%), groesser ist besser",
                100
            )
        ],
        plots / "geometry.png",
        "DTU Geometrieauswertung"
    )

    # Renderingqualitaet.
    draw_metrics(
        df,
        summary,
        [
            ("psnr", "PSNR (dB)", 1),
            ("ssim", "SSIM", 1),
            (
                "lpips",
                "LPIPS, kleiner ist besser",
                1
            )
        ],
        plots / "rendering.png",
        "Rendering auf unabhaengigen Testansichten"
    )

    # Laufzeiten.
    draw_metrics(
        df,
        summary,
        [
            ("train_s", "Training (Sekunden)", 1),
            ("total_s", "Gesamtdauer (Sekunden)", 1)
        ],
        plots / "runtime.png",
        "Laufzeiten"
    )

    # Meshgroesse.
    draw_metrics(
        df,
        summary,
        [
            ("n_vertices", "Mesh-Vertices", 1),
            ("n_faces", "Mesh-Faces", 1)
        ],
        plots / "mesh_size.png",
        "Meshgroesse (kein Qualitaetsmass)"
    )

    # Tabelle als PNG.
    make_table_picture(
        summary,
        tables / "summary_by_n.png"
    )

    # Tabelle im Terminal.
    print(
        "\nAuswertung nach Bildanzahl "
        "(Mittel nur bei 3/3 Szenen):"
    )

    print(
        summary[
            [
                "n",
                "n_geometry_ok",
                "mean_chamfer",
                "mean_f_score",
                "mean_psnr",
                "mean_train_s"
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
