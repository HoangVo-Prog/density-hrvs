"""Freeze and independently verify the existing benchmark without model inference.

Usage: python -m algorithm.tools.experiment_baseline SOURCE OUTPUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_iou(first, second) -> float:
    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    overlap = area((max(first[0], second[0]), max(first[1], second[1]),
                    min(first[2], second[2]), min(first[3], second[3])))
    union = area(first) + area(second) - overlap
    return overlap / union if union else 0.0


def inspect_batch(source: Path) -> tuple[dict, list, list[Path]]:
    summary = json.loads((source / 'summary.json').read_text())
    semantic = json.loads((source / 'graded_semantic.json').read_text())
    strict = json.loads((source / 'graded.json').read_text())
    selection = json.loads((source / 'selection.json').read_text())
    expected_ids = {row['question_id'] for row in selection}
    grade = {row['question_id']: int(row['correct']) for row in semantic['rows']}
    strict_grade = {row['question_id']: int(row['correct']) for row in strict['rows']}
    if len(expected_ids) != len(selection) or set(grade) != expected_ids or set(strict_grade) != expected_ids:
        raise ValueError('Selection and grade question IDs do not match uniquely')
    files = [source / name for name in ('summary.json', 'selection.json', 'config.json',
                                       'graded.json', 'graded_semantic.json')]
    rows = []
    for question in selection:
        qid = question['question_id']
        base = source / qid
        inspection = json.loads((base / 'inspection/inspection.json').read_text())
        completion = json.loads((base / 'completion.json').read_text())
        candidates = inspection['candidates']
        overlaps = [box_iou(c['bbox'], inspection['target_bbox']) for c in candidates]
        top = overlaps[0] if overlaps else 0.0
        best = max(overlaps, default=0.0)
        for key, actual in (('top1_iou', top), ('best_iou', best)):
            if abs(completion[key] - actual) > 1e-8:
                raise ValueError(f'{qid}: stored {key} disagrees with exported boxes')
        rows.append({**question, 'top1_iou': top, 'best_iou': best,
                     'semantic_correct': grade[qid], 'strict_correct': strict_grade[qid],
                     'joint_correct': bool(grade[qid] and top >= .5)})
        files.extend(base / name for name in ('completion.json', 'run/results.json',
                                             'inspection/inspection.json'))
    metrics = {'questions': len(rows), 'images': len({r['image_id'] for r in rows}),
               'top1_iou_at_50': sum(r['top1_iou'] >= .5 for r in rows),
               'any_iou_at_50': sum(r['best_iou'] >= .5 for r in rows),
               'semantic_correct': sum(r['semantic_correct'] for r in rows),
               'strict_correct': sum(r['strict_correct'] for r in rows),
               'joint_correct': sum(r['joint_correct'] for r in rows)}
    if metrics['questions'] != summary['total'] or metrics['semantic_correct'] != semantic['correct']:
        raise ValueError('Summary totals disagree with individual records')
    return metrics, rows, files


def verify_snapshot(output: Path) -> dict:
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha256(output / name) != expected:
            raise ValueError(f'Frozen baseline changed: {name}')
    return manifest['metrics']


def freeze(source: Path, output: Path, repo: Path) -> dict:
    if output.exists():
        return verify_snapshot(output)
    metrics, rows, files = inspect_batch(source)
    output.mkdir(parents=True)
    hashes = {}
    for path in files:
        relative = Path('records') / path.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(target)
        target.chmod(0o444)
    archive = output / 'source_snapshot.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted((repo / 'algorithm').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                tar.add(path, arcname=path.relative_to(repo))
    hashes[archive.name] = sha256(archive)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'source_directory': str(source.resolve()), 'source_commit': commit,
                'metrics': metrics, 'files': hashes,
                'scope': 'Frozen source and JSON evidence; original images and masks remain in source directories.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'cases.json').write_text(json.dumps(rows, indent=2) + '\n')
    regression_ids = [r['question_id'] for r in rows if r['joint_correct']]
    diagnostic_ids = ['0063_q02', '0288_q01', '0411_q02', '0467_q02', '0649_q02',
                      '0719_q02', '1497_q02', '2329_q02', '2633_q01', '2783_q01', '3194_q02']
    (output / 'diagnostic_selection.json').write_text(json.dumps(
        [r for r in selection_inputs(rows) if r['question_id'] in set(regression_ids + diagnostic_ids)], indent=2) + '\n')
    return verify_snapshot(output)


def selection_inputs(rows: list[dict]) -> list[dict]:
    """Return inference inputs only: labels and target boxes are excluded."""
    return [{k: row[k] for k in ('image_id', 'question_id', 'question')} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.output, Path(__file__).resolve().parents[2]), indent=2))


if __name__ == '__main__':
    main()
