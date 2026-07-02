"""Conversational AI Dashboard — clinical conversation analysis, turn-taking
patterns, topic extraction, and interaction quality metrics.

Maps clinical.db tables to conversational AI concepts:
- conversation_log          -> operator/assistant turns, response lengths
- patients                  -> patient context for conversations
- transaction_log           -> pipeline events
- assessments               -> clinical assessment context
"""

import sqlite3
import os
import re
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Common stop words to exclude from topic extraction
STOP_WORDS = {
    'about', 'after', 'again', 'being', 'below', 'between', 'could',
    'doing', 'during', 'every', 'first', 'found', 'given', 'going',
    'great', 'having', 'here', 'itself', 'known', 'large', 'later',
    'least', 'level', 'light', 'likely', 'local', 'major', 'makes',
    'might', 'model', 'never', 'newer', 'noted', 'often', 'order',
    'other', 'outer', 'overall', 'place', 'point', 'quite', 'rather',
    'ready', 'right', 'shall', 'short', 'shown', 'since', 'small',
    'space', 'start', 'state', 'still', 'taken', 'thank', 'thanks',
    'their', 'there', 'these', 'thing', 'those', 'three', 'times',
    'total', 'truly', 'under', 'until', 'upper', 'using', 'value',
    'wants', 'where', 'which', 'while', 'whole', 'would', 'write',
    'based', 'below', 'build', 'built', 'check', 'clear', 'close',
    'current', 'default', 'defined', 'details', 'different',
    'example', 'expected', 'following', 'function', 'general',
    'helpful', 'however', 'include', 'looking', 'making', 'needed',
    'number', 'output', 'please', 'possible', 'provide', 'provided',
    'running', 'second', 'section', 'should', 'single', 'specific',
    'system', 'through', 'together', 'updated', 'working',
}


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _load_conversations():
    return _db_query(
        "SELECT id, role, text, ts_utc, ts_local "
        "FROM conversation_log ORDER BY ts_utc"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log ORDER BY ts_utc"
    )


def _word_count(text):
    if not text:
        return 0
    return len(text.split())


def _char_count(text):
    if not text:
        return 0
    return len(text)


def _extract_date(ts):
    if not ts:
        return ''
    return ts[:10]


def _extract_hour(ts):
    """Extract hour from timestamp string."""
    if not ts:
        return None
    # Try to match HH:MM pattern
    m = re.search(r'(\d{2}):\d{2}', ts)
    if m:
        return int(m.group(1))
    return None


def _length_bucket(char_count):
    if char_count < 100:
        return 'short (<100 chars)'
    elif char_count < 500:
        return 'medium (100-500)'
    elif char_count < 2000:
        return 'long (500-2000)'
    else:
        return 'very long (>2000)'


def _extract_topics(conversations, top_n=20):
    """Extract topic keywords from conversation text.
    Count words >5 chars, exclude stop words, return top N."""
    word_counter = Counter()
    for c in conversations:
        text = c.get('text') or ''
        words = re.findall(r'[a-zA-Z]+', text.lower())
        for w in words:
            if len(w) > 5 and w not in STOP_WORDS:
                word_counter[w] += 1
    return [
        {'word': word, 'count': count}
        for word, count in word_counter.most_common(top_n)
    ]


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Conversational AI dashboard."""
    conversations = _load_conversations()

    if not conversations:
        return {
            'available': False,
            'message': 'No conversation data available. '
                       'Start clinical conversations to populate the dashboard.',
        }

    total_turns = len(conversations)

    # Role counts
    role_counts = Counter(c['role'] for c in conversations)
    operator_turns = role_counts.get('operator', 0)
    assistant_turns = role_counts.get('assistant', 0)

    # Unique days and sessions (group by date as proxy for sessions)
    dates = [_extract_date(c.get('ts_local') or c.get('ts_utc')) for c in conversations]
    dates = [d for d in dates if d]
    unique_days = len(set(dates))
    total_conversations = unique_days  # sessions approximated by unique days

    # Average turns per day
    avg_turns_per_day = round(total_turns / unique_days, 2) if unique_days else 0

    # Response lengths (assistant only)
    assistant_lengths = [
        _char_count(c.get('text'))
        for c in conversations if c.get('role') == 'assistant'
    ]
    avg_response_length = _avg(assistant_lengths)

    # Total words
    all_words = sum(_word_count(c.get('text')) for c in conversations)

    # --- Chart data ---

    # Role distribution (pie)
    role_distribution = [
        {'name': role, 'value': count}
        for role, count in sorted(role_counts.items())
    ]

    # Daily activity (line)
    daily_map = {}
    for c in conversations:
        day = _extract_date(c.get('ts_local') or c.get('ts_utc'))
        if not day:
            continue
        if day not in daily_map:
            daily_map[day] = {'operator_turns': 0, 'assistant_turns': 0, 'total': 0}
        role = c.get('role', '')
        if role == 'operator':
            daily_map[day]['operator_turns'] += 1
        elif role == 'assistant':
            daily_map[day]['assistant_turns'] += 1
        daily_map[day]['total'] += 1

    daily_activity = [
        {'date': day, **counts}
        for day, counts in sorted(daily_map.items())
    ]

    # Turn length distribution (bar)
    bucket_counts = Counter()
    for c in conversations:
        chars = _char_count(c.get('text'))
        bucket_counts[_length_bucket(chars)] += 1

    bucket_order = [
        'short (<100 chars)',
        'medium (100-500)',
        'long (500-2000)',
        'very long (>2000)',
    ]
    turn_length_distribution = [
        {'bucket': b, 'count': bucket_counts.get(b, 0)}
        for b in bucket_order
        if bucket_counts.get(b, 0) > 0
    ]

    # Response time stats (counts by hour of day)
    hour_counts = Counter()
    for c in conversations:
        hour = _extract_hour(c.get('ts_local') or c.get('ts_utc'))
        if hour is not None:
            hour_counts[hour] += 1

    response_time_stats = [
        {'hour': h, 'count': hour_counts.get(h, 0)}
        for h in range(24)
        if hour_counts.get(h, 0) > 0
    ]

    return {
        'available': True,
        'total_conversations': total_conversations,
        'total_turns': total_turns,
        'operator_turns': operator_turns,
        'assistant_turns': assistant_turns,
        'avg_turns_per_day': avg_turns_per_day,
        'unique_days': unique_days,
        'avg_response_length': avg_response_length,
        'total_words': all_words,
        'role_distribution': role_distribution,
        'daily_activity': daily_activity,
        'turn_length_distribution': turn_length_distribution,
        'response_time_stats': response_time_stats,
        'kpis': [
            {'label': 'Conversations', 'value': str(total_conversations)},
            {'label': 'Total Turns', 'value': str(total_turns)},
            {'label': 'Operator Turns', 'value': str(operator_turns)},
            {'label': 'Assistant Turns', 'value': str(assistant_turns)},
            {'label': 'Avg Turns/Day', 'value': str(avg_turns_per_day)},
            {'label': 'Unique Days', 'value': str(unique_days)},
            {'label': 'Avg Response Length', 'value': f'{avg_response_length:.0f} chars',
             'color': '#10b981' if avg_response_length >= 200 else '#f59e0b'},
            {'label': 'Total Words', 'value': f'{all_words:,}'},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed conversational AI breakdown — conversation inventory, daily
    summaries, role statistics, topic extraction, pipeline events."""
    conversations = _load_conversations()
    pipeline_all = _load_pipeline_events()

    if not conversations:
        return {'available': False}

    # --- Conversation inventory ---
    conversation_inventory = []
    for c in conversations:
        text = c.get('text') or ''
        conversation_inventory.append({
            'id': c['id'],
            'role': c['role'],
            'text': text[:200] + ('...' if len(text) > 200 else ''),
            'ts_local': c.get('ts_local'),
            'word_count': _word_count(text),
            'char_count': _char_count(text),
        })

    # --- Daily summary ---
    daily_map = {}
    for c in conversations:
        day = _extract_date(c.get('ts_local') or c.get('ts_utc'))
        if not day:
            continue
        if day not in daily_map:
            daily_map[day] = {
                'operator_turns': 0,
                'assistant_turns': 0,
                'total_turns': 0,
                'total_words': 0,
            }
        role = c.get('role', '')
        if role == 'operator':
            daily_map[day]['operator_turns'] += 1
        elif role == 'assistant':
            daily_map[day]['assistant_turns'] += 1
        daily_map[day]['total_turns'] += 1
        daily_map[day]['total_words'] += _word_count(c.get('text'))

    daily_summary = [
        {'date': day, **stats}
        for day, stats in sorted(daily_map.items())
    ]

    # --- Role stats ---
    role_groups = {}
    for c in conversations:
        role = c.get('role', 'unknown')
        if role not in role_groups:
            role_groups[role] = {'count': 0, 'lengths': [], 'words': 0}
        role_groups[role]['count'] += 1
        role_groups[role]['lengths'].append(_char_count(c.get('text')))
        role_groups[role]['words'] += _word_count(c.get('text'))

    role_stats = {}
    for role, data in role_groups.items():
        role_stats[role] = {
            'count': data['count'],
            'avg_length': _avg(data['lengths']),
            'total_words': data['words'],
        }

    # --- Pipeline events (last 50) ---
    pipeline_events = [
        {
            'id': ev['id'],
            'component': ev.get('component'),
            'action': ev.get('action'),
            'actor': ev.get('actor'),
            'detail': ev.get('detail'),
            'ts_utc': ev.get('ts_utc'),
            'ts_local': ev.get('ts_local'),
        }
        for ev in pipeline_all[-50:]
    ]

    # --- Topics ---
    topics = _extract_topics(conversations, top_n=20)

    return {
        'available': True,
        'conversations': conversation_inventory,
        'daily_summary': daily_summary,
        'role_stats': role_stats,
        'pipeline_events': pipeline_events,
        'topics': topics,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Conversational AI dashboard."""
    return {
        'concepts': [
            {
                'name': 'Conversational AI',
                'description': 'AI systems that engage in natural-language dialogue with '
                               'users, combining natural language understanding (NLU), '
                               'dialogue management, and natural language generation (NLG) '
                               'to support clinical decision-making, patient interaction, '
                               'and operational queries in healthcare settings.',
            },
            {
                'name': 'Natural Language Understanding (NLU)',
                'description': 'The component that parses user input to extract meaning, '
                               'including intent classification, entity extraction, and '
                               'sentiment analysis. In clinical contexts, NLU must handle '
                               'medical terminology, abbreviations, and domain-specific jargon.',
            },
            {
                'name': 'Dialogue Management',
                'description': 'The orchestration layer that tracks conversation state, '
                               'manages context across turns, and decides the next system '
                               'action. Maintains coherent multi-turn interactions and '
                               'ensures clinical conversations follow safe, structured flows.',
            },
            {
                'name': 'Intent Recognition',
                'description': 'Classification of user utterances into predefined intent '
                               'categories (e.g., clinical query, status check, task '
                               'delegation). Enables routing of conversations to appropriate '
                               'handlers and ensures correct clinical workflows are triggered.',
            },
            {
                'name': 'Named Entity Recognition (NER)',
                'description': 'Identification and extraction of structured entities from '
                               'unstructured text, including patient identifiers, medication '
                               'names, anatomical regions, disease codes, and temporal '
                               'expressions critical for clinical documentation.',
            },
            {
                'name': 'Context Window',
                'description': 'The span of prior conversation turns and contextual '
                               'information available to the AI during response generation. '
                               'Proper context management ensures clinically relevant, '
                               'coherent responses without information loss across long '
                               'multi-turn clinical sessions.',
            },
            {
                'name': 'Turn-Taking',
                'description': 'The mechanism governing alternation between user and system '
                               'utterances. In clinical AI, turn-taking protocols ensure the '
                               'system waits for complete clinical input before responding '
                               'and avoids interrupting critical information delivery.',
            },
            {
                'name': 'Clinical Q&A Pipeline',
                'description': 'End-to-end pipeline that receives clinical questions, '
                               'retrieves relevant patient data and medical knowledge, '
                               'generates evidence-based answers, and logs interactions '
                               'for audit. Combines retrieval-augmented generation (RAG) '
                               'with clinical safety guardrails.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Response Relevance',
                'description': 'Measures how well assistant responses address the '
                               'operator\'s query or clinical need. Evaluated via semantic '
                               'similarity between query intent and response content. '
                               'Target: >0.85 relevance score for clinical queries.',
            },
            {
                'name': 'Turn Completion Rate',
                'description': 'Percentage of conversation turns that reach a satisfactory '
                               'resolution without requiring follow-up clarification. '
                               'Higher rates indicate effective single-turn understanding '
                               'and response generation. Target: >80%.',
            },
            {
                'name': 'Context Retention',
                'description': 'Ability to maintain and correctly reference information '
                               'from earlier turns in a conversation. Measured by accuracy '
                               'of references to prior context in multi-turn sessions. '
                               'Critical for longitudinal patient discussions.',
            },
            {
                'name': 'User Satisfaction Score',
                'description': 'Composite metric derived from conversation completion '
                               'patterns, explicit feedback, and engagement indicators. '
                               'Scores range 1-5, with >4.0 indicating effective clinical '
                               'AI assistance.',
            },
        ],
        'interaction_types': [
            {
                'type': 'Clinical Query',
                'description': 'Questions about patient data, diagnoses, treatment plans, '
                               'lab results, or clinical protocols.',
            },
            {
                'type': 'System Status',
                'description': 'Requests for pipeline health, model performance, job '
                               'status, or infrastructure state.',
            },
            {
                'type': 'Task Delegation',
                'description': 'Instructions to execute analyses, run models, generate '
                               'reports, or trigger automated workflows.',
            },
            {
                'type': 'Troubleshooting',
                'description': 'Debugging conversations addressing errors, failed jobs, '
                               'data quality issues, or unexpected model behavior.',
            },
            {
                'type': 'Configuration',
                'description': 'Setting up parameters, adjusting thresholds, modifying '
                               'pipeline configurations, or managing system settings.',
            },
        ],
        'compliance': [
            {
                'ref': 'FDA AI/ML Framework',
                'note': 'Conversational AI used for clinical decision support must '
                        'follow the Predetermined Change Control Plan (PCCP). '
                        'Dialogue models must be validated for clinical accuracy '
                        'and safety of generated responses.',
            },
            {
                'ref': 'EU AI Act Art. 6',
                'note': 'Clinical conversational AI systems are classified as '
                        'high-risk. Require conformity assessment, human oversight, '
                        'and transparency obligations including disclosure of AI '
                        'involvement in clinical interactions.',
            },
            {
                'ref': 'ISO 14971',
                'note': 'Risk management must address incorrect clinical responses, '
                        'hallucinated medical information, context loss in multi-turn '
                        'sessions, and failure to escalate critical findings.',
            },
            {
                'ref': 'IEC 62304',
                'note': 'Conversational AI software components (NLU, dialogue manager, '
                        'response generator, context tracker) must follow medical '
                        'device software lifecycle processes with documented V&V.',
            },
            {
                'ref': 'HIPAA',
                'note': 'Conversation logs containing clinical information are PHI. '
                        'Systems must encrypt data at rest and in transit, enforce '
                        'role-based access, and maintain audit trails for all '
                        'conversation data access and storage.',
            },
        ],
        'remediation': [
            {
                'strategy': 'Response Quality Monitoring',
                'description': 'Continuously evaluate assistant response relevance and '
                               'accuracy against clinical ground truth. Flag low-confidence '
                               'responses for human review before delivery.',
            },
            {
                'strategy': 'Context Window Optimization',
                'description': 'Implement sliding window and summarization strategies to '
                               'maintain relevant clinical context across long conversations '
                               'without exceeding token limits or losing critical information.',
            },
            {
                'strategy': 'Intent Drift Detection',
                'description': 'Monitor for shifts in conversation topic distributions over '
                               'time. Retrain intent classifiers when new clinical query '
                               'patterns emerge that fall outside existing categories.',
            },
            {
                'strategy': 'Conversation Audit Trail',
                'description': 'Maintain complete, tamper-evident logs of all clinical '
                               'conversations for regulatory compliance. Implement automated '
                               'flagging of conversations containing safety-critical content.',
            },
        ],
    }
