import argparse
import os

import pandas as pd


GRADE_TO_WEIGHT = {
    "A": 1.0,
    "B": 0.4,
    "C": -10,
}


def _normalize_grade(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Encountered missing project grade while building weight files.")
    grade = str(value).strip().upper()
    if grade not in GRADE_TO_WEIGHT:
        raise ValueError(f"Unsupported project grade '{value}'. Expected one of {sorted(GRADE_TO_WEIGHT)}.")
    return grade


def build_weight_files(
        projects_csv: str,
        schedblocks_csv: str,
        project_weights_out: str,
        sb_weights_out: str,
) -> None:
    projects_df = pd.read_csv(projects_csv)
    schedblocks_df = pd.read_csv(schedblocks_csv)

    required_project_columns = {"OBSPROJECT_UID", "CODE", "PRJ_LETTER_GRADE"}
    required_schedblock_columns = {"SB_UID", "OBSPROJECT_UID"}
    missing_project_columns = required_project_columns.difference(projects_df.columns)
    missing_schedblock_columns = required_schedblock_columns.difference(schedblocks_df.columns)

    if missing_project_columns:
        raise ValueError(f"projects CSV is missing required columns: {sorted(missing_project_columns)}")
    if missing_schedblock_columns:
        raise ValueError(f"schedblocks CSV is missing required columns: {sorted(missing_schedblock_columns)}")

    projects_for_weights = projects_df[["OBSPROJECT_UID", "CODE", "PRJ_LETTER_GRADE"]].copy()
    projects_for_weights["PRJ_LETTER_GRADE"] = projects_for_weights["PRJ_LETTER_GRADE"].map(_normalize_grade)

    conflicting_codes = (
        projects_for_weights.groupby("CODE")["PRJ_LETTER_GRADE"].nunique().loc[lambda s: s > 1]
    )
    if not conflicting_codes.empty:
        raise ValueError(
            "Found project codes with multiple grades: "
            f"{sorted(conflicting_codes.index.tolist())[:10]}"
        )

    project_weights_df = (
        projects_for_weights[["CODE", "PRJ_LETTER_GRADE"]]
        .drop_duplicates()
        .rename(columns={"PRJ_LETTER_GRADE": "grade"})
        .sort_values("CODE")
        .reset_index(drop=True)
    )
    project_weights_df["weight"] = project_weights_df["grade"].map(GRADE_TO_WEIGHT)

    sb_weights_df = (
        schedblocks_df[["SB_UID", "OBSPROJECT_UID"]]
        .merge(
            projects_for_weights[["OBSPROJECT_UID", "CODE", "PRJ_LETTER_GRADE"]].drop_duplicates(),
            on="OBSPROJECT_UID",
            how="left",
        )
        .rename(columns={"PRJ_LETTER_GRADE": "grade"})
        .sort_values("SB_UID")
        .reset_index(drop=True)
    )

    missing_project_links = sb_weights_df["CODE"].isna()
    if missing_project_links.any():
        missing_sb_uids = sb_weights_df.loc[missing_project_links, "SB_UID"].astype(str).tolist()
        raise ValueError(
            "Some SBs could not be matched to a project grade: "
            f"{missing_sb_uids[:10]}"
        )

    sb_weights_df["grade"] = sb_weights_df["grade"].map(_normalize_grade)
    sb_weights_df["weight"] = sb_weights_df["grade"].map(GRADE_TO_WEIGHT)
    sb_weights_df = sb_weights_df[["SB_UID", "CODE", "grade", "weight"]]

    os.makedirs(os.path.dirname(project_weights_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(sb_weights_out) or ".", exist_ok=True)
    project_weights_df.to_csv(project_weights_out, index=False)
    sb_weights_df.to_csv(sb_weights_out, index=False)

    print(f"Wrote project weights to {project_weights_out}")
    print(f"Wrote SB weights to {sb_weights_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grade-based project and SB weight CSVs."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing projects_c10.csv and schedblocks_c10.csv.")
    parser.add_argument("--projects_csv", type=str, default=None,
                        help="Override path to projects CSV. Defaults to <data_dir>/projects_c10.csv.")
    parser.add_argument("--schedblocks_csv", type=str, default=None,
                        help="Override path to schedblocks CSV. Defaults to <data_dir>/schedblocks_c10.csv.")
    parser.add_argument("--project_weights_out", type=str, default=None,
                        help="Override path for project_weights.csv. Defaults to <data_dir>/project_weights.csv.")
    parser.add_argument("--sb_weights_out", type=str, default=None,
                        help="Override path for sb_weights.csv. Defaults to <data_dir>/sb_weights.csv.")
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    projects_csv = os.path.abspath(os.path.expanduser(args.projects_csv or os.path.join(data_dir, "projects_c10.csv")))
    schedblocks_csv = os.path.abspath(os.path.expanduser(args.schedblocks_csv or os.path.join(data_dir, "schedblocks_c10.csv")))
    project_weights_out = os.path.abspath(os.path.expanduser(args.project_weights_out or os.path.join(data_dir, "project_weights.csv")))
    sb_weights_out = os.path.abspath(os.path.expanduser(args.sb_weights_out or os.path.join(data_dir, "sb_weights.csv")))

    build_weight_files(
        projects_csv=projects_csv,
        schedblocks_csv=schedblocks_csv,
        project_weights_out=project_weights_out,
        sb_weights_out=sb_weights_out,
    )


if __name__ == "__main__":
    main()
