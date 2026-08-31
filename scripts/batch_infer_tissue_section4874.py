#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candycrunch.prediction import wrap_inference


#SUPPORTED_SUFFIXES = {".xlsx", ".mzML", ".mzXML", ".pkl"}
SUPPORTED_SUFFIXES = {".mzML"}


def collect_spectra_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("df_mz_"):
            continue
        if path.suffix in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CandyCrunch inference on every supported spectra file in a directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "TissueSection4874",
        help="Directory containing spectra files to annotate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "predictions" / "TissueSection4874",
        help="Directory where one prediction file per input file will be written.",
    )
    parser.add_argument("--glycan-class", default="N")
    parser.add_argument("--model", default="CNN")
    parser.add_argument("--mode", default="negative")
    parser.add_argument("--modification", default="reduced")
    parser.add_argument("--mass-tag", type=float, default=None)
    parser.add_argument("--lc", default="PGC")
    parser.add_argument("--trap", default="linear")
    parser.add_argument("--rt-min", type=float, default=0)
    parser.add_argument("--rt-max", type=float, default=0)
    parser.add_argument("--rt-diff", type=float, default=1.0)
    parser.add_argument("--rt-max-default", type=float, default=30.0)
    parser.add_argument("--pred-thresh", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--spectra", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--get-missing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mass-tolerance", type=float, default=0.5)
    parser.add_argument("--extra-thresh", type=float, default=None)
    parser.add_argument("--crumbs-thresh", type=float, default=3)
    parser.add_argument("--ppm-thresh", type=float, default=300)
    parser.add_argument("--supplement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--experimental", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-prep", default="underivatized")
    parser.add_argument("--taxonomy-level", default="Class")
    parser.add_argument("--taxonomy-filter", default="Mammalia")
    parser.add_argument("--plot-glycans", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--struct-mass-tol", type=float, default=0.6)
    parser.add_argument("--bin-num", type=int, default=2048)
    parser.add_argument("--max-charge", type=int, default=3)
    parser.add_argument("--frag-num", type=int, default=100)
    parser.add_argument(
        "--filter-out",
        nargs="*",
        default=["Ac", "Kdn", "HexA", "Pen", "HexN", "Me", "PCho", "PEtN"],
        help="Monosaccharides/modifications to filter out.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    spectra_files = collect_spectra_files(input_dir)
    if not spectra_files:
        raise SystemExit(f"No supported spectra files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(spectra_files)} files in {input_dir}")
    print(f"Writing predictions to {output_dir}")

    for spectra_path in spectra_files:
        print(f"Running {spectra_path.name}")
        df_out = wrap_inference(
            str(spectra_path),
            glycan_class=args.glycan_class,
            model=args.model,
            bin_num=args.bin_num,
            max_charge=args.max_charge,
            frag_num=args.frag_num,
            mode=args.mode,
            modification=args.modification,
            mass_tag=args.mass_tag,
            lc=args.lc,
            trap=args.trap,
            rt_min=args.rt_min,
            rt_max=args.rt_max,
            rt_diff=args.rt_diff,
            rt_max_default=args.rt_max_default,
            pred_thresh=args.pred_thresh,
            temperature=args.temperature,
            spectra=args.spectra,
            get_missing=args.get_missing,
            mass_tolerance=args.mass_tolerance,
            extra_thresh=args.extra_thresh,
            crumbs_thresh=args.crumbs_thresh,
            ppm_thresh=args.ppm_thresh,
            filter_out=set(args.filter_out),
            supplement=args.supplement,
            experimental=args.experimental,
            sample_prep=args.sample_prep,
            taxonomy_level=args.taxonomy_level,
            taxonomy_filter=args.taxonomy_filter,
            plot_glycans=args.plot_glycans,
            struct_mass_tol=args.struct_mass_tol,
        )

        out_path = output_dir / f"{spectra_path.stem}_predictions.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
