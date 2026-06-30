"""Citation Dashboard — real citation grounding analytics from ChromaDB + clinical.db.

Sources:
- data/vector_db/chroma.sqlite3 — embeddings (75), embedding_metadata (225), embeddings_queue (784)
- data/clinical.db — conversation_log (225), analyses (21), transaction_log (558),
  expert_reviews (3), feedback (1), patients

Metrics:
- Citation rate: % of assistant responses containing source-grounded references
- Source coverage: % of vector store document types referenced in responses
- Per-type citation stats: frequency of each document type cited
- Daily citation volume from conversation timestamps
- Top cited patients and diseases
- Citation gap analysis: patients/documents without citations
- Expert review alignment with cited responses
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DB = _PROJECT_ROOT / "data" / "vector_db" / "chroma.sqlite3"
_CLINICAL_DB = _PROJECT_ROOT / "data" / "clinical.db"


def _chroma_conn():
    if not _CHROMA_DB.exists():
        return None
    return sqlite3.connect(str(_CHROMA_DB))


def _clinical_conn():
    if not _CLINICAL_DB.exists():
        return None
    return sqlite3.connect(str(_CLINICAL_DB))


def _safe(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


# ── Citation pattern helpers ─────────────────────────────────────

# Patterns that indicate grounded/cited content in assistant responses
_PATIENT_ID_RE = re.compile(r"\bP\d{4}\b")
_CONFIDENCE_RE = re.compile(r"\b(?:confidence|probability|score)\s*[:\-]?\s*\d+\.?\d*\s*%?", re.I)
_DISEASE_NAMES = [
    "epilepsy", "seizure", "depression", "anxiety", "adhd",
    "schizophrenia", "parkinson", "alzheimer", "dementia",
    "bipolar", "ocd", "ptsd", "insomnia", "migraine",
]
_DISEASE_RE = re.compile(r"\b(?:" + "|".join(_DISEASE_NAMES) + r")\b", re.I)
_MEDICATION_RE = re.compile(
    r"\b(?:levetiracetam|carbamazepine|valproate|lamotrigine|phenytoin|topiramate|"
    r"gabapentin|oxcarbazepine|lacosamide|clobazam|sertraline|fluoxetine|"
    r"escitalopram|venlafaxine|duloxetine|bupropion|mirtazapine|clonazepam)\b", re.I
)
_ANALYSIS_REF_RE = re.compile(r"\b(?:analysis|report|finding|result|prediction|classification)\b", re.I)
_NUMERIC_DATA_RE = re.compile(r"\b\d+\.\d+\b")


def _count_citations(text):
    """Count distinct citation signals in a single assistant response."""
    if not text:
        return 0
    count = 0
    count += len(_PATIENT_ID_RE.findall(text))
    count += len(_CONFIDENCE_RE.findall(text))
    count += min(len(_DISEASE_RE.findall(text)), 3)  # cap disease mentions
    count += len(_MEDICATION_RE.findall(text))
    count += min(len(_ANALYSIS_REF_RE.findall(text)), 2)
    return count


def _has_citation(text):
    """Whether text contains at least one source-grounded reference."""
    if not text:
        return False
    return (
        bool(_PATIENT_ID_RE.search(text))
        or bool(_CONFIDENCE_RE.search(text))
        or bool(_MEDICATION_RE.search(text))
        or (bool(_DISEASE_RE.search(text)) and bool(_NUMERIC_DATA_RE.search(text)))
    )


def _is_specific(text):
    """Whether a response contains specific data (not vague)."""
    if not text:
        return False
    specificity = 0
    if _PATIENT_ID_RE.search(text):
        specificity += 2
    if _CONFIDENCE_RE.search(text):
        specificity += 2
    if _NUMERIC_DATA_RE.search(text):
        specificity += 1
    if _MEDICATION_RE.search(text):
        specificity += 1
    return specificity >= 2


def _extract_patient_ids(text):
    """Extract all patient IDs mentioned in text."""
    if not text:
        return []
    return _PATIENT_ID_RE.findall(text)


def _extract_diseases(text):
    """Extract disease names mentioned in text."""
    if not text:
        return []
    return [m.lower() for m in _DISEASE_RE.findall(text)]


# ── Overview ──────────────────────────────────────────────────────

def citation_overview():
    """Aggregate citation metrics: rate, coverage, quality, per-type stats, daily volume."""
    clinical = _clinical_conn()
    chroma = _chroma_conn()

    if clinical is None and chroma is None:
        return {"available": False, "note": "Neither clinical.db nor ChromaDB found"}

    result = {"available": True, "generated_at": datetime.now(timezone.utc).isoformat()}

    # ── Vector store document count ──
    total_documents = 0
    doc_types = []
    if chroma:
        cur = chroma.cursor()
        total_documents = _safe(cur, "SELECT count(*) FROM embeddings")
        cur.execute(
            "SELECT string_value, count(*) FROM embedding_metadata "
            "WHERE key='type' GROUP BY string_value ORDER BY count(*) DESC"
        )
        doc_types = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]
        chroma.close()

    doc_type_names = {d["type"] for d in doc_types}

    # ── Assistant responses and citation analysis ──
    total_assistant = 0
    cited_count = 0
    specific_count = 0
    citation_counts = []
    daily_volume = {}
    patient_mentions = Counter()
    disease_mentions = Counter()
    type_citation_counts = Counter()
    responses_with_detail = []

    if clinical:
        ccur = clinical.cursor()
        total_assistant = _safe(
            ccur,
            "SELECT count(*) FROM conversation_log WHERE role='assistant'"
        )

        ccur.execute(
            "SELECT id, text, ts_utc FROM conversation_log "
            "WHERE role='assistant' ORDER BY id"
        )
        rows = ccur.fetchall()

        for row_id, text, ts in rows:
            if not text:
                continue
            has_cite = _has_citation(text)
            cite_count = _count_citations(text)
            is_spec = _is_specific(text)

            if has_cite:
                cited_count += 1
            if is_spec:
                specific_count += 1
            citation_counts.append(cite_count)

            # Daily volume
            if ts:
                day = ts[:10] if len(ts) >= 10 else ts
                daily_volume[day] = daily_volume.get(day, 0) + (1 if has_cite else 0)

            # Patient ID mentions
            for pid in _extract_patient_ids(text):
                patient_mentions[pid] += 1

            # Disease mentions
            for disease in _extract_diseases(text):
                disease_mentions[disease] += 1

            # Infer document type citations
            if _PATIENT_ID_RE.search(text or ""):
                type_citation_counts["patient"] += 1
            if _ANALYSIS_REF_RE.search(text or "") and _NUMERIC_DATA_RE.search(text or ""):
                type_citation_counts["analysis"] += 1
            if _MEDICATION_RE.search(text or ""):
                type_citation_counts["medications"] += 1

            responses_with_detail.append({
                "id": row_id,
                "citation_count": cite_count,
                "has_citation": has_cite,
                "is_specific": is_spec,
                "ts": ts or "",
            })

        # Total patients
        total_patients = _safe(ccur, "SELECT count(*) FROM patients")
        clinical.close()
    else:
        total_patients = 0

    # ── Compute KPIs ──
    citation_rate = round(cited_count / max(total_assistant, 1) * 100, 1)

    # Source coverage: % of vector doc types that appear referenced
    referenced_types = set(type_citation_counts.keys())
    source_coverage = round(
        len(referenced_types & doc_type_names) / max(len(doc_type_names), 1) * 100, 1
    )

    # Citation quality score (composite):
    #   citation_rate weight 40%, specificity 30%, source_coverage 30%
    specificity_pct = round(specific_count / max(total_assistant, 1) * 100, 1)
    quality_score = round(
        citation_rate * 0.4 + specificity_pct * 0.3 + source_coverage * 0.3, 1
    )

    avg_citations = round(
        statistics.mean(citation_counts) if citation_counts else 0, 2
    )
    median_citations = round(
        statistics.median(citation_counts) if citation_counts else 0, 2
    )

    result["summary"] = {
        "total_documents_in_vector_store": total_documents,
        "total_assistant_responses": total_assistant,
        "cited_responses": cited_count,
        "citation_rate_pct": citation_rate,
        "source_coverage_pct": source_coverage,
        "specificity_pct": specificity_pct,
        "citation_quality_score": quality_score,
        "avg_citations_per_response": avg_citations,
        "median_citations_per_response": median_citations,
    }

    # Per-type citation stats
    per_type_stats = []
    for dt in doc_types:
        t = dt["type"]
        cite_n = type_citation_counts.get(t, 0)
        per_type_stats.append({
            "type": t,
            "documents_in_store": dt["count"],
            "times_cited": cite_n,
            "citation_rate_pct": round(cite_n / max(total_assistant, 1) * 100, 1),
        })
    result["per_type_citation_stats"] = per_type_stats

    # Daily citation volume
    result["daily_citation_volume"] = [
        {"date": d, "cited_responses": c}
        for d, c in sorted(daily_volume.items())
    ]

    # Top cited patients
    result["top_cited_patients"] = [
        {"patient_id": pid, "mentions": cnt}
        for pid, cnt in patient_mentions.most_common(15)
    ]

    # Top cited diseases
    result["top_cited_diseases"] = [
        {"disease": d, "mentions": cnt}
        for d, cnt in disease_mentions.most_common(10)
    ]

    # Faithfulness indicators
    result["faithfulness"] = {
        "specific_responses": specific_count,
        "vague_responses": total_assistant - specific_count,
        "specificity_pct": specificity_pct,
    }

    return result


# ── Breakdown ─────────────────────────────────────────────────────

def citation_breakdown():
    """Detailed citation drill-down: per-response detail, per-disease coverage,
    document-to-citation mapping, gap analysis, component rates, temporal trends,
    expert review alignment."""

    clinical = _clinical_conn()
    chroma = _chroma_conn()

    if clinical is None and chroma is None:
        return {"available": False, "note": "Neither clinical.db nor ChromaDB found"}

    result = {"available": True}

    # ── Per-response citation detail (recent 50 assistant messages) ──
    per_response = []
    all_responses = []
    if clinical:
        ccur = clinical.cursor()
        ccur.execute(
            "SELECT id, text, ts_utc FROM conversation_log "
            "WHERE role='assistant' ORDER BY id DESC LIMIT 50"
        )
        for row_id, text, ts in ccur.fetchall():
            cite_count = _count_citations(text)
            pids = _extract_patient_ids(text)
            diseases = _extract_diseases(text)
            per_response.append({
                "id": row_id,
                "ts": ts or "",
                "citation_count": cite_count,
                "has_citation": _has_citation(text),
                "patient_ids_mentioned": pids,
                "diseases_mentioned": diseases,
                "text_preview": (text or "")[:150],
            })

        # Full assistant responses for aggregate analysis
        ccur.execute(
            "SELECT id, text, ts_utc FROM conversation_log "
            "WHERE role='assistant' ORDER BY id"
        )
        all_responses = ccur.fetchall()

    result["per_response_detail"] = per_response

    # ── Per-disease citation coverage ──
    disease_coverage = {}
    if clinical:
        ccur = clinical.cursor()
        # Get analyses grouped by disease
        ccur.execute(
            "SELECT disease, count(*) FROM analyses "
            "WHERE disease IS NOT NULL GROUP BY disease ORDER BY count(*) DESC"
        )
        disease_analysis_counts = dict(ccur.fetchall())

        # Count citations per disease from assistant responses
        disease_cite_counts = Counter()
        for _, text, _ in all_responses:
            for d in _extract_diseases(text):
                disease_cite_counts[d] += 1

        all_diseases = set(disease_analysis_counts.keys()) | set(disease_cite_counts.keys())
        for d in sorted(all_diseases):
            d_lower = d.lower()
            disease_coverage[d] = {
                "analyses_in_db": disease_analysis_counts.get(d, 0),
                "times_cited_in_responses": disease_cite_counts.get(d_lower, disease_cite_counts.get(d, 0)),
                "coverage_ratio": round(
                    disease_cite_counts.get(d_lower, disease_cite_counts.get(d, 0))
                    / max(disease_analysis_counts.get(d, 1), 1), 2
                ),
            }

    result["per_disease_citation_coverage"] = disease_coverage

    # ── Document-to-citation mapping ──
    doc_citation_map = []
    if chroma:
        chcur = chroma.cursor()
        try:
            chcur.execute("SELECT DISTINCT id FROM embedding_metadata ORDER BY id")
            doc_ids = [r[0] for r in chcur.fetchall()]

            # Collect all patient IDs mentioned across responses
            all_cited_pids = Counter()
            for _, text, _ in all_responses:
                for pid in _extract_patient_ids(text):
                    all_cited_pids[pid] += 1

            for doc_id in doc_ids[:100]:
                chcur.execute(
                    "SELECT key, string_value FROM embedding_metadata WHERE id=?",
                    (doc_id,)
                )
                meta = dict(chcur.fetchall())
                pid = meta.get("patient_id", "")
                doc_type = meta.get("type", "unknown")
                times_cited = all_cited_pids.get(pid, 0) if pid else 0

                doc_citation_map.append({
                    "doc_id": doc_id,
                    "patient_id": pid,
                    "type": doc_type,
                    "times_patient_cited": times_cited,
                    "document_preview": (meta.get("chroma:document", ""))[:100],
                })
        except Exception:
            pass
        chroma.close()

    result["document_citation_mapping"] = doc_citation_map

    # ── Citation gap analysis ──
    gap_analysis = {"patients_without_citations": [], "patients_with_citations": []}
    if clinical:
        ccur = clinical.cursor()
        ccur.execute("SELECT patient_id FROM patients WHERE patient_id IS NOT NULL")
        db_patient_ids = [r[0] for r in ccur.fetchall()]

        cited_pids = set()
        for _, text, _ in all_responses:
            cited_pids.update(_extract_patient_ids(text))

        for pid in sorted(db_patient_ids):
            if pid in cited_pids:
                gap_analysis["patients_with_citations"].append(pid)
            else:
                gap_analysis["patients_without_citations"].append(pid)

        gap_analysis["total_patients"] = len(db_patient_ids)
        gap_analysis["cited_patients"] = len(gap_analysis["patients_with_citations"])
        gap_analysis["uncited_patients"] = len(gap_analysis["patients_without_citations"])
        gap_analysis["gap_pct"] = round(
            len(gap_analysis["patients_without_citations"]) / max(len(db_patient_ids), 1) * 100, 1
        )

    result["citation_gap_analysis"] = gap_analysis

    # ── Per-component citation rates ──
    component_rates = []
    if clinical:
        ccur = clinical.cursor()
        ccur.execute(
            "SELECT component, count(*) FROM transaction_log "
            "WHERE component IS NOT NULL GROUP BY component ORDER BY count(*) DESC"
        )
        components = ccur.fetchall()

        # For each component, check how many of its actions correlate with cited responses
        for comp, total in components:
            ccur.execute(
                "SELECT count(DISTINCT ref_id) FROM transaction_log "
                "WHERE component=? AND ref_id IS NOT NULL", (comp,)
            )
            unique_refs = ccur.fetchone()[0]
            component_rates.append({
                "component": comp,
                "total_transactions": total,
                "unique_refs": unique_refs,
            })

    result["per_component_citation_rates"] = component_rates

    # ── Temporal trends ──
    temporal = []
    if all_responses:
        daily_stats = defaultdict(lambda: {"total": 0, "cited": 0})
        for _, text, ts in all_responses:
            if not ts:
                continue
            day = ts[:10] if len(ts) >= 10 else ts
            daily_stats[day]["total"] += 1
            if _has_citation(text):
                daily_stats[day]["cited"] += 1

        for day in sorted(daily_stats.keys()):
            s = daily_stats[day]
            temporal.append({
                "date": day,
                "total_responses": s["total"],
                "cited_responses": s["cited"],
                "citation_rate_pct": round(s["cited"] / max(s["total"], 1) * 100, 1),
            })

    result["temporal_trends"] = temporal

    # ── Expert review alignment ──
    expert_alignment = []
    if clinical:
        ccur = clinical.cursor()
        try:
            ccur.execute(
                "SELECT er.patient_id, er.expert, er.finding, er.agree_with_ai, er.note, "
                "er.analysis_id "
                "FROM expert_reviews er ORDER BY er.id"
            )
            reviews = ccur.fetchall()

            for pid, expert, finding, agree, note, analysis_id in reviews:
                # Check if this patient is cited in assistant responses
                patient_cited = any(
                    pid in _extract_patient_ids(text or "")
                    for _, text, _ in all_responses
                ) if pid else False

                expert_alignment.append({
                    "patient_id": pid or "",
                    "expert": expert or "",
                    "finding": finding or "",
                    "agrees_with_ai": bool(agree),
                    "note": note or "",
                    "analysis_id": analysis_id,
                    "patient_cited_in_responses": patient_cited,
                })
        except Exception:
            pass

    result["expert_review_alignment"] = expert_alignment

    if clinical:
        clinical.close()

    return result


# ── Definitions ───────────────────────────────────────────────────

def citation_definitions():
    """Metric definitions for the Citation Dashboard."""
    return {
        "available": True,
        "metrics": {
            "citation_rate_pct": {
                "label": "Citation Rate",
                "description": (
                    "Percentage of assistant responses containing at least one "
                    "source-grounded reference (patient ID, confidence value, "
                    "medication name, or disease + numeric data). Measures how "
                    "often the AI grounds its answers in real clinical data."
                ),
                "computation": (
                    "count(responses with citation) / count(all assistant responses) * 100"
                ),
                "clinical_relevance": (
                    "Higher citation rates indicate the AI is referencing actual patient "
                    "records and analyses rather than generating unsupported claims."
                ),
            },
            "source_coverage_pct": {
                "label": "Source Coverage",
                "description": (
                    "Percentage of vector store document types (patient, analysis, "
                    "medications, etc.) that appear referenced in assistant responses."
                ),
                "computation": (
                    "count(referenced doc types) / count(all doc types in vector store) * 100"
                ),
                "clinical_relevance": (
                    "Ensures the AI utilizes the full breadth of available clinical "
                    "data, not just a narrow subset."
                ),
            },
            "citation_quality_score": {
                "label": "Citation Quality Score",
                "description": (
                    "Composite score (0-100) combining citation rate (40%), "
                    "specificity (30%), and source coverage (30%)."
                ),
                "computation": (
                    "citation_rate * 0.4 + specificity_pct * 0.3 + source_coverage * 0.3"
                ),
                "clinical_relevance": (
                    "Single metric summarizing overall citation quality — higher "
                    "scores mean the AI is consistently producing well-grounded, "
                    "specific, broadly-sourced responses."
                ),
            },
            "specificity_pct": {
                "label": "Response Specificity",
                "description": (
                    "Percentage of assistant responses containing specific data "
                    "(patient IDs, confidence scores, numeric values) rather than "
                    "vague or generic answers."
                ),
                "computation": (
                    "count(specific responses) / count(all assistant responses) * 100. "
                    "Specific = at least 2 specificity signals (patient ID, confidence, "
                    "numeric data, medication name)."
                ),
                "clinical_relevance": (
                    "Specific responses enable clinicians to verify AI claims against "
                    "source records; vague responses cannot be audited."
                ),
            },
            "citation_gap_pct": {
                "label": "Citation Gap",
                "description": (
                    "Percentage of patients in the clinical database who are never "
                    "referenced in any assistant response."
                ),
                "computation": (
                    "count(uncited patients) / count(total patients) * 100"
                ),
                "clinical_relevance": (
                    "Identifies patients whose data exists in the system but is "
                    "never surfaced in AI responses — potential blind spots."
                ),
            },
            "per_disease_coverage_ratio": {
                "label": "Per-Disease Citation Coverage",
                "description": (
                    "Ratio of citation mentions to analyses for each disease. "
                    "Diseases with low ratios may be under-represented in AI output."
                ),
                "computation": (
                    "count(disease mentions in responses) / count(analyses for disease)"
                ),
                "clinical_relevance": (
                    "Ensures equitable AI attention across disease categories, "
                    "preventing bias toward frequently-studied conditions."
                ),
            },
            "expert_review_alignment": {
                "label": "Expert Review Alignment",
                "description": (
                    "Whether patients with expert reviews are also cited in "
                    "assistant responses, indicating the AI references reviewed cases."
                ),
                "computation": (
                    "Cross-reference expert_reviews.patient_id with patient IDs "
                    "found in assistant response text."
                ),
                "clinical_relevance": (
                    "Cited responses that align with expert-reviewed cases provide "
                    "higher confidence in AI reliability."
                ),
            },
            "avg_citations_per_response": {
                "label": "Average Citations per Response",
                "description": (
                    "Mean number of citation signals per assistant response, "
                    "including patient IDs, confidence values, medication names, "
                    "disease references, and analysis terms."
                ),
                "computation": (
                    "sum(citation signals across all responses) / count(responses)"
                ),
                "clinical_relevance": (
                    "Higher averages suggest denser grounding in source data; "
                    "very high values may indicate verbose rather than precise output."
                ),
            },
        },
        "data_sources": {
            "chroma_db": (
                "data/vector_db/chroma.sqlite3 — embeddings (75 rows), "
                "embedding_metadata (225 rows with keys: chroma:document, "
                "patient_id, type), embeddings_queue (784 rows)"
            ),
            "clinical_db": (
                "data/clinical.db — conversation_log (225 rows, assistant=171, "
                "operator=54), analyses (21 rows), transaction_log (558 rows), "
                "expert_reviews (3 rows), feedback (1 row), patients"
            ),
        },
    }


if __name__ == "__main__":
    ov = citation_overview()
    print(json.dumps(ov, indent=2, default=str))
