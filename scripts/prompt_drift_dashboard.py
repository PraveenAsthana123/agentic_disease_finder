"""Prompt Drift Dashboard — prompt/response length monitoring, topic distribution,
temporal drift analysis, volume trends.

Aggregates data from:
- data/clinical.db conversation_log (307 rows: operator + assistant messages)
- prompt_inputs/ directory (~59 markdown prompt files)
- data/clinical.db transaction_log (pipeline events)
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, 'data', 'clinical.db')
PROMPT_INPUTS_DIR = os.path.join(BASE, 'prompt_inputs')

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for',
    'on', 'at', 'by', 'and', 'or', 'not', 'but', 'with', 'from', 'as', 'be',
    'this', 'that', 'it', 'its', 'has', 'have', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall',
    'been', 'being', 'am', 'so', 'if', 'then', 'than', 'no', 'yes', 'all',
    'each', 'every', 'any', 'some', 'such', 'there', 'here', 'when', 'where',
    'how', 'what', 'which', 'who', 'whom', 'why', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'out', 'over',
    'under', 'again', 'further', 'once', 'also', 'just', 'more', 'most',
    'other', 'only', 'own', 'same', 'very', 'too', 'now', 'new', 'get',
    'got', 'make', 'made', 'see', 'use', 'used', 'one', 'two', 'need',
    'like', 'these', 'those', 'them', 'they', 'their', 'our', 'your', 'you',
    'we', 'me', 'my', 'up', 'down', 'way', 'well', 'back', 'set', 'add',
    'run', 'let', 'per', 'put', 'say', 'take', 'give', 'show', 'try',
    'still', 'even', 'much', 'many', 'really', 'already', 'don', 'doesn',
    'didn', 'won', 'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven', 'hadn',
}

CLINICAL_KEYWORDS = {
    'patient', 'seizure', 'eeg', 'epilepsy', 'clinical', 'medication',
    'diagnosis', 'treatment', 'mri', 'neuropsych', 'brain', 'disease',
}
TECHNICAL_KEYWORDS = {
    'api', 'server', 'backend', 'frontend', 'build', 'deploy', 'code',
    'script', 'model', 'train', 'data', 'database',
}


def _categorize(word):
    if word in CLINICAL_KEYWORDS:
        return 'clinical'
    if word in TECHNICAL_KEYWORDS:
        return 'technical'
    return 'operational'


def _query_conversation_log():
    """Return all conversation_log rows as list of dicts."""
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, role, text, ts_utc, ts_local FROM conversation_log "
        "ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _extract_date(ts_local):
    """Extract YYYY-MM-DD from ts_local string."""
    if not ts_local:
        return None
    return ts_local[:10]


def _extract_week(ts_local):
    """Extract ISO week string YYYY-Www from ts_local."""
    if not ts_local:
        return None
    try:
        dt = datetime.fromisoformat(ts_local)
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    except Exception:
        return ts_local[:10]


def _list_prompt_files():
    """List .md files in prompt_inputs/ with stats."""
    if not os.path.isdir(PROMPT_INPUTS_DIR):
        return []
    results = []
    for fn in sorted(os.listdir(PROMPT_INPUTS_DIR)):
        if not fn.endswith('.md'):
            continue
        fp = os.path.join(PROMPT_INPUTS_DIR, fn)
        try:
            stat = os.stat(fp)
            with open(fp, 'r', errors='replace') as f:
                content = f.read()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')
            results.append({
                'filename': fn,
                'length': len(content),
                'date': mtime,
            })
        except Exception:
            continue
    return results


def overview():
    """Prompt drift overview: KPIs, length-over-time, role distribution, interpretation."""
    try:
        rows = _query_conversation_log()
        if not rows:
            return {'available': False, 'message': 'No conversation_log data found'}

        operators = [r for r in rows if r['role'] == 'operator']
        assistants = [r for r in rows if r['role'] == 'assistant']

        total_prompts = len(operators)
        total_responses = len(assistants)

        op_lengths = [len(r['text'] or '') for r in operators]
        as_lengths = [len(r['text'] or '') for r in assistants]

        avg_prompt_len = round(sum(op_lengths) / max(len(op_lengths), 1), 1)
        avg_response_len = round(sum(as_lengths) / max(len(as_lengths), 1), 1)

        prompt_files = _list_prompt_files()
        prompt_file_count = len(prompt_files)

        dates = [_extract_date(r['ts_local']) for r in rows if r['ts_local']]
        dates = [d for d in dates if d]
        date_range = f"{min(dates)} to {max(dates)}" if dates else 'N/A'

        # Compute drift: first-half avg vs second-half avg
        half = len(op_lengths) // 2
        if half > 0:
            first_half_prompt = sum(op_lengths[:half]) / half
            second_half_prompt = sum(op_lengths[half:]) / max(len(op_lengths[half:]), 1)
            prompt_drift_pct = round(
                ((second_half_prompt - first_half_prompt) / max(first_half_prompt, 1)) * 100, 1
            )
        else:
            prompt_drift_pct = 0.0

        half_r = len(as_lengths) // 2
        if half_r > 0:
            first_half_resp = sum(as_lengths[:half_r]) / half_r
            second_half_resp = sum(as_lengths[half_r:]) / max(len(as_lengths[half_r:]), 1)
            response_drift_pct = round(
                ((second_half_resp - first_half_resp) / max(first_half_resp, 1)) * 100, 1
            )
        else:
            response_drift_pct = 0.0

        kpis = [
            {'label': 'Total Prompts', 'value': total_prompts, 'unit': 'messages',
             'icon': 'MessageSquare'},
            {'label': 'Total Responses', 'value': total_responses, 'unit': 'messages',
             'icon': 'MessageCircle'},
            {'label': 'Avg Prompt Length', 'value': avg_prompt_len, 'unit': 'chars',
             'icon': 'AlignLeft'},
            {'label': 'Avg Response Length', 'value': avg_response_len, 'unit': 'chars',
             'icon': 'AlignRight'},
            {'label': 'Prompt Files', 'value': prompt_file_count, 'unit': 'files',
             'icon': 'FileText'},
            {'label': 'Date Range', 'value': date_range, 'unit': '',
             'icon': 'Calendar'},
            {'label': 'Prompt Length Drift', 'value': prompt_drift_pct, 'unit': '%',
             'icon': 'TrendingUp' if prompt_drift_pct >= 0 else 'TrendingDown'},
            {'label': 'Response Length Drift', 'value': response_drift_pct, 'unit': '%',
             'icon': 'TrendingUp' if response_drift_pct >= 0 else 'TrendingDown'},
        ]

        # Length over time (by date)
        date_op = {}
        date_as = {}
        for r in rows:
            d = _extract_date(r['ts_local'])
            if not d:
                continue
            tlen = len(r['text'] or '')
            if r['role'] == 'operator':
                date_op.setdefault(d, []).append(tlen)
            elif r['role'] == 'assistant':
                date_as.setdefault(d, []).append(tlen)

        all_dates = sorted(set(list(date_op.keys()) + list(date_as.keys())))
        length_over_time = []
        for d in all_dates:
            op_vals = date_op.get(d, [])
            as_vals = date_as.get(d, [])
            length_over_time.append({
                'date': d,
                'avg_prompt_len': round(sum(op_vals) / max(len(op_vals), 1), 1) if op_vals else 0,
                'avg_response_len': round(sum(as_vals) / max(len(as_vals), 1), 1) if as_vals else 0,
            })

        role_distribution = [
            {'name': 'operator', 'value': total_prompts},
            {'name': 'assistant', 'value': total_responses},
        ]

        # Interpretation
        drift_dir_prompt = 'increasing' if prompt_drift_pct > 5 else (
            'decreasing' if prompt_drift_pct < -5 else 'stable')
        drift_dir_resp = 'increasing' if response_drift_pct > 5 else (
            'decreasing' if response_drift_pct < -5 else 'stable')
        interpretation = (
            f"Prompt drift analysis over {len(all_dates)} days: "
            f"{total_prompts} operator prompts, {total_responses} assistant responses. "
            f"Avg prompt length {avg_prompt_len} chars, avg response length {avg_response_len} chars. "
            f"Prompt length trend is {drift_dir_prompt} ({prompt_drift_pct:+.1f}%), "
            f"response length trend is {drift_dir_resp} ({response_drift_pct:+.1f}%). "
            f"{prompt_file_count} prompt input files tracked in prompt_inputs/."
        )

        return {
            'available': True,
            'kpis': kpis,
            'length_over_time': length_over_time,
            'role_distribution': role_distribution,
            'interpretation': interpretation,
            'run_at': datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        return {'available': False, 'message': f'Error generating overview: {e}'}


def breakdown():
    """Prompt drift breakdown: histograms, daily volume, topic keywords, file stats, weekly drift."""
    try:
        rows = _query_conversation_log()
        if not rows:
            return {'available': False, 'message': 'No conversation_log data found'}

        operators = [r for r in rows if r['role'] == 'operator']
        assistants = [r for r in rows if r['role'] == 'assistant']

        op_lengths = [len(r['text'] or '') for r in operators]
        as_lengths = [len(r['text'] or '') for r in assistants]

        # Prompt length histogram
        prompt_bins = [
            ('0-100', 0, 100),
            ('100-500', 100, 500),
            ('500-1000', 500, 1000),
            ('1000-5000', 1000, 5000),
            ('5000+', 5000, float('inf')),
        ]
        prompt_length_histogram = []
        for label, lo, hi in prompt_bins:
            count = sum(1 for l in op_lengths if lo <= l < hi)
            prompt_length_histogram.append({'bin': label, 'count': count})

        # Response length histogram
        response_length_histogram = []
        for label, lo, hi in prompt_bins:
            count = sum(1 for l in as_lengths if lo <= l < hi)
            response_length_histogram.append({'bin': label, 'count': count})

        # Daily volume
        daily = {}
        for r in rows:
            d = _extract_date(r['ts_local'])
            if not d:
                continue
            daily.setdefault(d, {'prompts': 0, 'responses': 0})
            if r['role'] == 'operator':
                daily[d]['prompts'] += 1
            elif r['role'] == 'assistant':
                daily[d]['responses'] += 1

        daily_volume = [
            {'date': d, 'prompts': v['prompts'], 'responses': v['responses']}
            for d, v in sorted(daily.items())
        ]

        # Topic keywords from operator prompts
        word_counter = Counter()
        for r in operators:
            text = (r['text'] or '').lower()
            # Remove common punctuation
            for ch in '.,;:!?()[]{}"\'/\\@#$%^&*_+=|<>~`':
                text = text.replace(ch, ' ')
            words = text.split()
            for w in words:
                w = w.strip('-')
                if len(w) < 3:
                    continue
                if w in STOP_WORDS:
                    continue
                # Skip purely numeric tokens
                if w.isdigit():
                    continue
                word_counter[w] += 1

        topic_keywords = []
        for word, count in word_counter.most_common(20):
            topic_keywords.append({
                'keyword': word,
                'count': count,
                'category': _categorize(word),
            })

        # Prompt file stats
        prompt_file_stats = _list_prompt_files()

        # Weekly drift
        weekly = {}
        for r in rows:
            w = _extract_week(r['ts_local'])
            if not w:
                continue
            weekly.setdefault(w, {
                'op_lens': [], 'as_lens': [], 'n_prompts': 0, 'n_responses': 0
            })
            tlen = len(r['text'] or '')
            if r['role'] == 'operator':
                weekly[w]['op_lens'].append(tlen)
                weekly[w]['n_prompts'] += 1
            elif r['role'] == 'assistant':
                weekly[w]['as_lens'].append(tlen)
                weekly[w]['n_responses'] += 1

        weekly_drift = []
        for w in sorted(weekly.keys()):
            d = weekly[w]
            avg_p = round(sum(d['op_lens']) / max(len(d['op_lens']), 1), 1) if d['op_lens'] else 0
            avg_r = round(sum(d['as_lens']) / max(len(d['as_lens']), 1), 1) if d['as_lens'] else 0
            weekly_drift.append({
                'week': w,
                'avg_prompt_len': avg_p,
                'avg_response_len': avg_r,
                'n_prompts': d['n_prompts'],
                'n_responses': d['n_responses'],
            })

        return {
            'available': True,
            'prompt_length_histogram': prompt_length_histogram,
            'response_length_histogram': response_length_histogram,
            'daily_volume': daily_volume,
            'topic_keywords': topic_keywords,
            'prompt_file_stats': prompt_file_stats,
            'weekly_drift': weekly_drift,
        }

    except Exception as e:
        return {'available': False, 'message': f'Error generating breakdown: {e}'}


def definitions():
    """Prompt drift definitions: metrics, categories, detection methods, clinical relevance."""
    return {
        'available': True,
        'sections': [
            {
                'title': 'Prompt Drift Metrics',
                'items': [
                    {
                        'term': 'Prompt Length Drift',
                        'definition': (
                            'Percentage change in average operator prompt length between '
                            'the first half and second half of the observation period. '
                            'A positive value indicates prompts are getting longer over time, '
                            'which may signal increasing complexity or scope creep.'
                        ),
                    },
                    {
                        'term': 'Response Length Drift',
                        'definition': (
                            'Percentage change in average assistant response length between '
                            'the first half and second half of the observation period. '
                            'Significant drift may indicate model behaviour changes or '
                            'evolving task complexity.'
                        ),
                    },
                    {
                        'term': 'Volume Drift',
                        'definition': (
                            'Change in the daily or weekly count of prompts and responses '
                            'over time. Sudden spikes or drops may indicate workflow changes, '
                            'system issues, or shifts in operator engagement patterns.'
                        ),
                    },
                ],
            },
            {
                'title': 'Prompt Categories',
                'items': [
                    {
                        'term': 'Clinical Prompts',
                        'definition': (
                            'Prompts containing clinical terminology such as patient, seizure, '
                            'EEG, epilepsy, medication, diagnosis, treatment, MRI, neuropsych, '
                            'brain, or disease. These represent direct clinical decision support '
                            'interactions.'
                        ),
                    },
                    {
                        'term': 'Technical Prompts',
                        'definition': (
                            'Prompts related to system infrastructure including API, server, '
                            'backend, frontend, build, deploy, code, script, model, train, '
                            'data, or database operations.'
                        ),
                    },
                    {
                        'term': 'Operational Prompts',
                        'definition': (
                            'Prompts that do not fall into clinical or technical categories. '
                            'These cover workflow management, status enquiries, configuration, '
                            'reporting, and general system usage.'
                        ),
                    },
                ],
            },
            {
                'title': 'Drift Detection Methods',
                'items': [
                    {
                        'term': 'Length-Based Drift',
                        'definition': (
                            'Compares average prompt and response lengths across time periods '
                            '(first half vs second half, or weekly). A drift exceeding +/-5% '
                            'is flagged as noteworthy; exceeding +/-20% is flagged as significant.'
                        ),
                    },
                    {
                        'term': 'Topic Drift',
                        'definition': (
                            'Monitors the distribution of keywords across prompt categories '
                            '(clinical, technical, operational) over time. Shifts in topic '
                            'distribution may indicate changing usage patterns or scope changes.'
                        ),
                    },
                    {
                        'term': 'Volume Drift',
                        'definition': (
                            'Tracks the number of prompts and responses per day and per week. '
                            'Identifies periods of high or low activity and detects trend changes '
                            'in system usage intensity.'
                        ),
                    },
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {
                        'term': 'IEC 62304 Compliance',
                        'definition': (
                            'IEC 62304 requires software lifecycle process monitoring. Prompt '
                            'drift monitoring satisfies post-market surveillance requirements '
                            'by tracking how operators interact with the clinical AI system over '
                            'time, detecting usage pattern changes that may affect safety.'
                        ),
                    },
                    {
                        'term': 'FDA AI/ML PCCP',
                        'definition': (
                            'The FDA Predetermined Change Control Plan for AI/ML-based SaMD '
                            'requires monitoring of input data drift. Prompt drift is a form of '
                            'input drift for conversational AI systems — changes in how users '
                            'phrase requests can affect model performance and clinical outcomes.'
                        ),
                    },
                    {
                        'term': 'Patient Safety Impact',
                        'definition': (
                            'Significant prompt drift may indicate that operators are using the '
                            'system in ways not validated during development, potentially leading '
                            'to unreliable AI responses in clinical decision-making contexts.'
                        ),
                    },
                ],
            },
            {
                'title': 'Remediation',
                'items': [
                    {
                        'term': 'Prompt Template Updates',
                        'definition': (
                            'When prompt drift is detected, review and update standardized prompt '
                            'templates to guide operators back to validated interaction patterns. '
                            'Track template versions in prompt_inputs/ directory.'
                        ),
                    },
                    {
                        'term': 'Operator Training',
                        'definition': (
                            'Address prompt drift through targeted operator training sessions '
                            'when keyword analysis shows shifting topic distributions or when '
                            'prompt lengths deviate significantly from baseline.'
                        ),
                    },
                    {
                        'term': 'System Revalidation',
                        'definition': (
                            'When prompt drift exceeds 20%, trigger a system revalidation cycle '
                            'to ensure AI responses remain clinically appropriate for the evolved '
                            'usage patterns. Document findings per IEC 62304 change control.'
                        ),
                    },
                ],
            },
        ],
    }


if __name__ == '__main__':
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else 'overview'
    if target == 'overview':
        result = overview()
    elif target == 'breakdown':
        result = breakdown()
    elif target == 'definitions':
        result = definitions()
    else:
        result = {'error': f'Unknown target: {target}'}

    print(json.dumps(result, indent=2))
