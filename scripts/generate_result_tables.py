import os
import json
from pathlib import Path

RESULTS_DIR = Path('SLM_Results') / 'gemma3_4b'

def load_dataset_summaries(results_dir):
    summaries = {}
    for path in results_dir.glob('*/dataset_summary_*.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            summaries[data['dataset']] = data
    return summaries


def accuracy_table(summaries):
    datasets = sorted(summaries.keys())
    header = ['Dataset', 'zero_shot', 'few_shot', 'cot']
    rows = [header]
    for ds in datasets:
        methods = summaries[ds].get('methods', {})
        row = [ds]
        for m in ['zero_shot', 'few_shot', 'cot']:
            acc = methods.get(m, {}).get('accuracy')
            row.append(f"{acc:.2f}" if acc is not None else 'N/A')
        rows.append(row)
    return rows


def metrics_table(summaries):
    datasets = sorted(summaries.keys())
    header = ['Dataset', 'Method', 'total_questions', 'correct_answers', 'total_time(s)', 'total_tokens']
    rows = [header]
    for ds in datasets:
        methods = summaries[ds].get('methods', {})
        for m in ['zero_shot', 'few_shot', 'cot']:
            md = methods.get(m, {})
            rows.append([
                ds,
                m,
                md.get('total_questions', 'N/A'),
                md.get('correct_answers', 'N/A'),
                f"{md.get('total_time', 0):.2f}" if 'total_time' in md else 'N/A',
                md.get('total_tokens', 'N/A')
            ])
    return rows


def md_table(rows):
    # Convert rows to markdown table string
    if not rows:
        return ''
    header = rows[0]
    lines = []
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '|'.join([' --- ' for _ in header]) + '|')
    for r in rows[1:]:
        lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    summaries = load_dataset_summaries(RESULTS_DIR)
    acc_rows = accuracy_table(summaries)
    met_rows = metrics_table(summaries)
    print('# Accuracy by dataset and method\n')
    print(md_table(acc_rows))
    print('\n# Per-method metrics (time, tokens)\n')
    print(md_table(met_rows))
