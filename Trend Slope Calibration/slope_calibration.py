"""Calculate 5-frame quality-score slopes for clear-sky camera data."""

from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SIZE = 5
OUTPUT_FILE = "slope_calibration.csv"
SENSORS = {
    "CSI": "CSI_Quality_Score",
    "RGB": "RGB_Quality_Score",
}


def find_input_file(project_dir: Path, layout_id: int) -> Path:
    """Find the requested CSV, while also supporting the current filenames."""
    preferred_names = [f"Clear_T{layout_id}(1).csv", f"Clear_T{layout_id}.csv"]
    for name in preferred_names:
        matches = sorted(project_dir.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {preferred_names[0]} or {preferred_names[1]} "
        f"under {project_dir}"
    )


def calculate_layout_slopes(csv_path: Path, layout_id: int) -> list[dict]:
    data = pd.read_csv(csv_path)
    required_columns = {"Frame_ID", "Weather", *SENSORS.values()}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    clear_data = data.loc[data["Weather"].eq("clear_skies")].copy()
    clear_data = clear_data.sort_values("Frame_ID", kind="stable").reset_index(drop=True)
    x = np.arange(WINDOW_SIZE, dtype=float)
    results = []

    for sensor, score_column in SENSORS.items():
        scores = pd.to_numeric(clear_data[score_column], errors="coerce")

        for start in range(len(clear_data) - WINDOW_SIZE + 1):
            end = start + WINDOW_SIZE
            q = scores.iloc[start:end].to_numpy(dtype=float)

            # A linear regression needs all five quality scores to be present.
            if np.isnan(q).any():
                continue

            slope = np.polyfit(x, q, 1)[0]
            results.append(
                {
                    "Layout_ID": layout_id,
                    "Sensor": sensor,
                    "Start_Frame": clear_data.iloc[start]["Frame_ID"],
                    "End_Frame": clear_data.iloc[end - 1]["Frame_ID"],
                    "Slope": slope,
                }
            )

    return results


def print_statistics(results: pd.DataFrame) -> None:
    for sensor in SENSORS:
        slopes = results.loc[results["Sensor"].eq(sensor), "Slope"].to_numpy()
        absolute_slopes = np.abs(slopes)

        print(f"\n{sensor} clear-sky 5-frame slope distribution")
        print(f"  number of windows:                 {len(slopes)}")
        if len(slopes) == 0:
            print("  No valid windows available.")
            continue

        print(f"  mean slope:                        {np.mean(slopes):.10f}")
        print(f"  standard deviation:               {np.std(slopes):.10f}")
        print(f"  minimum slope:                     {np.min(slopes):.10f}")
        print(f"  maximum slope:                     {np.max(slopes):.10f}")
        print(f"  mean absolute slope:               {np.mean(absolute_slopes):.10f}")
        print(f"  95th percentile of absolute slope: {np.percentile(absolute_slopes, 95):.10f}")
        print(f"  99th percentile of absolute slope: {np.percentile(absolute_slopes, 99):.10f}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    all_results = []

    for layout_id in (1, 2, 3):
        csv_path = find_input_file(project_dir, layout_id)
        print(f"Layout {layout_id}: {csv_path}")
        all_results.extend(calculate_layout_slopes(csv_path, layout_id))

    results = pd.DataFrame(
        all_results,
        columns=["Layout_ID", "Sensor", "Start_Frame", "End_Frame", "Slope"],
    )
    output_path = script_dir / OUTPUT_FILE
    results.to_csv(output_path, index=False, float_format="%.10f")

    print(f"\nSaved {len(results)} windows to: {output_path}")
    print_statistics(results)


if __name__ == "__main__":
    main()
