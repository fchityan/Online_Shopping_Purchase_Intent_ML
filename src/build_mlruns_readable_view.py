from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def safe_slug(value: str) -> str:
    return ''.join(c.lower() if c.isalnum() else '_' for c in value).strip('_')


def load_meta(meta_path: Path) -> dict:
    data: dict[str, str] = {}
    with meta_path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip().strip("'").strip('"')
    return data


def build_readable_view(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f'Source MLflow directory not found: {source_dir}')

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for exp_dir in sorted(source_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        if not exp_dir.name.isdigit():
            continue

        exp_meta_path = exp_dir / 'meta.yaml'
        if not exp_meta_path.exists():
            continue

        exp_meta = load_meta(exp_meta_path)
        exp_name = exp_meta.get('name', f'experiment_{exp_dir.name}')
        exp_slug = safe_slug(exp_name) or f'experiment_{exp_dir.name}'
        exp_view_dir = target_dir / f'experiment__{exp_slug}__id_{exp_dir.name}'
        exp_view_dir.mkdir(parents=True, exist_ok=True)

        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name == '.trash':
                continue
            if len(run_dir.name) < 8:
                continue

            run_meta_path = run_dir / 'meta.yaml'
            if not run_meta_path.exists():
                continue

            run_meta = load_meta(run_meta_path)
            run_name = run_meta.get('run_name', f'run_{run_dir.name[:8]}')
            run_slug = safe_slug(run_name) or f'run_{run_dir.name[:8]}'
            start_time = str(run_meta.get('start_time', 'unknown'))
            run_view_name = f'run__{start_time}__{run_slug}__id_{run_dir.name[:8]}'

            link_path = exp_view_dir / run_view_name
            link_path.symlink_to(run_dir.resolve(), target_is_directory=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a human-readable MLflow folder view using symlinks.')
    parser.add_argument('--source-dir', default='mlruns_purchase_intent', help='Source MLflow folder.')
    parser.add_argument('--target-dir', default='mlops', help='Target folder for readable symlink view.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_readable_view(Path(args.source_dir), Path(args.target_dir))
    print(f'Readable MLflow view created at: {args.target_dir}')


if __name__ == '__main__':
    main()
