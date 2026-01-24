"""
Aggregate evaluation results across experiments.

Produces:
- experiments/aggregate_summary.csv : one row per experiment with summary metrics
- experiments/aggregate_metrics.csv : concatenation of all per-pair entries with experiment name
- experiments/aggregate_report.html : simple HTML table summary

Usage:
    python run_aggregate.py --experiments_dir experiments --output experiments/aggregate
"""

import argparse
import json
from pathlib import Path
import csv


def gather(experiments_dir):
    experiments_dir = Path(experiments_dir)
    rows = []
    pair_rows = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        eval_dir = exp_dir / 'evaluation'
        if not eval_dir.exists():
            continue
        summary_path = eval_dir / 'summary.json'
        metrics_path = eval_dir / 'metrics.csv'
        if not summary_path.exists():
            continue
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        row = {
            'experiment': exp_dir.name,
            'n_pairs': summary.get('n_pairs'),
            'content_lpips_mean': summary.get('content_lpips_mean'),
            'style_lpips_mean': summary.get('style_lpips_mean'),
            'content_vgg_mean': summary.get('content_vgg_mean'),
            'style_vgg_mean': summary.get('style_vgg_mean'),
            'clip_content_mean': summary.get('clip_content_mean'),
            'clip_style_mean': summary.get('clip_style_mean')
        }
        rows.append(row)

        if metrics_path.exists():
            with open(metrics_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r['experiment'] = exp_dir.name
                    pair_rows.append(r)
    return rows, pair_rows


def write_csv(rows, path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _img_to_datauri(img_path, size=(160, 160)):
    try:
        im = PILImage.open(img_path).convert('RGB')
        im.thumbnail(size)
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def write_html(rows, path, experiments_dir):
    path = Path(path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('<html><head><meta charset="utf-8"><title>Aggregate Report</title></head><body>')
        f.write('<h2>Experiment Summary</h2>')
        f.write('<table border="1" cellpadding="6" cellspacing="0">')
        # header
        if not rows:
            f.write('<tr><td>No experiments found</td></tr>')
        else:
            headers = list(rows[0].keys())
            # add thumbnails header
            headers_display = headers + ['sample_results']
            f.write('<tr>' + ''.join([f'<th>{h}</th>' for h in headers_display]) + '</tr>')
            for r in rows:
                f.write('<tr>')
                for h in headers:
                    f.write(f'<td>{r.get(h)}</td>')
                # find sample images
                exp_eval_dir = Path(experiments_dir) / r['experiment'] / 'evaluation'
                imgs = []
                if exp_eval_dir.exists():
                    imgs = sorted([p for p in exp_eval_dir.glob('*.jpg')])[:6]
                if not imgs:
                    f.write('<td>—</td>')
                else:
                    f.write('<td>')
                    for im in imgs:
                        datauri = _img_to_datauri(im)
                        if datauri:
                            f.write(f'<img src="{datauri}" style="width:120px;margin:4px;border:1px solid #ccc;"/>')
                    f.write('</td>')
                f.write('</tr>')
        f.write('</table></body></html>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', type=str, default='experiments')
    parser.add_argument('--output', type=str, default='experiments/aggregate')
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, pair_rows = gather(experiments_dir)

    # Write aggregate summary
    fieldnames = ['experiment','n_pairs','content_lpips_mean','style_lpips_mean','content_vgg_mean','style_vgg_mean','clip_content_mean','clip_style_mean','clip_text_mean']
    write_csv(rows, out_dir / 'aggregate_summary.csv', fieldnames)

    # Write pairwise metrics combined
    if pair_rows:
        pair_fieldnames = list(pair_rows[0].keys())
        write_csv(pair_rows, out_dir / 'aggregate_metrics.csv', pair_fieldnames)

    # HTML report with thumbnails
    write_html(rows, out_dir / 'aggregate_report.html', experiments_dir)

    print(f"Wrote aggregate summary to: {out_dir}")

if __name__ == '__main__':
    main()
