'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#4a148c'; // deep purple — BK channel / MaxiK (Ca²⁺-activated; distinct from all Kv channels)
const DANGER = '#b71c1c';
const SUCCESS = '#2e7d32';
const WARN = '#e65100';
const GOF_COLOR = '#6a1b9a'; // purple-GOF
const LOF_COLOR = '#1565c0'; // blue-LOF (Liang-Wang)

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, max = 100, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card shadow-sm mb-3">
      <div className="card-header fw-semibold text-white py-2" style={{ background: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const cohort = data.cohort || {};
  const etiologies = data.etiologies || [];

  return (
    <div>
      {/* Key alert banner */}
      <div className="alert py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5, background: '#f3e5f5' }}>
        <strong>🧬 KCNMA1 (10q22.3) — BK Channel (MaxiK/Slo1) — Largest K⁺ Channel (200–300 pS):</strong>{' '}
        KCNMA1 encodes the <strong>α-subunit (Slo1)</strong> of the BK channel — activated by BOTH voltage AND Ca²⁺.{' '}
        <strong>GOF →</strong> KCNMA1-EPD (epilepsy + paroxysmal dyskinesia — <em>caffeine-triggered, NORMAL EEG</em>).{' '}
        <strong>LOF →</strong> Liang-Wang syndrome (DEE + autism + hypotonia).{' '}
        <span style={{ color: DANGER }} className="fw-bold">
          ⚠️ QUINIDINE (BK blocker): PRECISION THERAPY for GOF; ABSOLUTE CI in LOF (worsens). {' '}
          ⚠️ Paroxysmal dyskinesia = NORMAL EEG — treat with CLB PRN, NOT AED escalation. {' '}
          ⚠️ Caffeine triggers dyskinesia (PKA → BK GOF sensitisation). {' '}
          ⚠️ TGB ABSOLUTE CI. ⚠️ POLG1 before VPA in LOF.
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={cohort.total || 40} />
        <KPI label="GOF Phenotype" value={cohort.gof_phenotype || '—'} color={GOF_COLOR} />
        <KPI label="Liang-Wang (LOF)" value={cohort.lof_liang_wang || '—'} color={LOF_COLOR} />
        <KPI label="Quinidine Users" value={cohort.quinidine_users || '—'} color={COLOR} />
        <KPI label="Seizure-Free ≥6mo" value={cohort.seizure_free_6mo || '—'} color={SUCCESS} />
        <KPI label="Avg Severity" value={cohort.avg_severity_score || '—'} color={WARN} />
      </div>

      {/* Gene info */}
      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <SectionCard title="🔬 Gene & Channel">
            <table className="table table-sm table-borderless mb-0 small">
              <tbody>
                <tr><td className="fw-semibold">Gene</td><td>{data.gene} ({data.locus})</td></tr>
                <tr><td className="fw-semibold">Channel</td><td>{data.channel}</td></tr>
                <tr><td className="fw-semibold">Conductance</td><td>{data.conductance}</td></tr>
                <tr><td className="fw-semibold">GOF Syndrome</td><td>{data.syndromes?.GOF}</td></tr>
                <tr><td className="fw-semibold">LOF Syndrome</td><td>{data.syndromes?.LOF}</td></tr>
                <tr><td className="fw-semibold">OMIM (gene)</td><td>{data.omim_gene}</td></tr>
                <tr><td className="fw-semibold">OMIM (EPD)</td><td>{data.omim_epd}</td></tr>
                <tr><td className="fw-semibold">OMIM (LWS)</td><td>{data.omim_lws}</td></tr>
              </tbody>
            </table>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="💊 Precision Pharmacology">
            <p className="small mb-2"><strong style={{ color: GOF_COLOR }}>GOF Precision:</strong> {data.precision_therapy}</p>
            <p className="small mb-2"><strong style={{ color: DANGER }}>Key CI:</strong> Quinidine <strong>ABSOLUTE CI in LOF</strong> — worsens by blocking residual BK</p>
            <p className="small mb-2"><strong style={{ color: WARN }}>Hallmark Trigger:</strong> {data.hallmark_trigger}</p>
            <p className="small mb-0"><strong>Diagnostic Key:</strong> {data.diagnostic_key}</p>
          </SectionCard>
        </div>
      </div>

      {/* Etiology bars */}
      <SectionCard title="📊 Etiology Distribution">
        {etiologies.map(e => (
          <Bar key={e.class} label={e.class} value={e.pct} color={e.class.includes('LOF') ? LOF_COLOR : GOF_COLOR} />
        ))}
      </SectionCard>

      {/* Key contraindications */}
      <SectionCard title="⛔ Key Contraindications" borderColor={DANGER}>
        {(data.key_contraindications || []).map((ci, i) => (
          <div key={i} className="small mb-1 text-danger fw-semibold">• {ci}</div>
        ))}
      </SectionCard>

      {/* Evidence */}
      <SectionCard title="📖 Landmark Evidence">
        <p className="small mb-1"><strong>First GOF mutation:</strong> {data.first_mutation}</p>
        <p className="small mb-1"><strong>Liang-Wang LOF:</strong> {data.liang_wang}</p>
        <p className="small mb-0"><strong>Quinidine evidence:</strong> {data.quinidine_evidence}</p>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  const [search, setSearch] = useState('');
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const patients = data.patients || [];
  const etiologies = data.etiologies || [];
  const filtered = patients.filter(p =>
    [p.id, p.variant, p.etiology_class, p.phenotype].some(v => v?.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div>
      {/* Etiology cards */}
      <div className="row g-3 mb-4">
        {etiologies.map(e => (
          <div key={e.category} className="col-md-6">
            <SectionCard title={`${e.pct}% — ${e.category}`} borderColor={e.category.includes('LOF') ? LOF_COLOR : GOF_COLOR}>
              <p className="small mb-1">{e.mechanism}</p>
              <p className="small text-muted mb-1"><strong>EEG:</strong> {e.eeg}</p>
              <p className="small text-muted mb-0"><strong>Onset:</strong> {e.onset_months} · <strong>Severity:</strong> {e.severity}</p>
            </SectionCard>
          </div>
        ))}
      </div>

      {/* Patient table */}
      <SectionCard title={`🧑‍⚕️ Patients (${filtered.length}/${patients.length})`}>
        <input
          className="form-control form-control-sm mb-3"
          placeholder="Search ID, variant, etiology…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-hover small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Phenotype</th><th>Variant</th>
                <th>Onset (mo)</th><th>Medications</th><th>Quinidine</th>
                <th>Dyskinesia/mo</th><th>QTc (ms)</th><th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>
                    <span className="badge" style={{ background: p.phenotype?.includes('GOF') ? GOF_COLOR : (p.phenotype?.includes('LOF') ? LOF_COLOR : '#888'), fontSize: '0.65rem' }}>
                      {p.phenotype?.replace('GOF-Epilepsy-', '').replace('LOF-', '') || '—'}
                    </span>
                  </td>
                  <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.variant}</td>
                  <td>{p.onset_months}</td>
                  <td>{(p.current_medications || []).join(', ')}</td>
                  <td>
                    <span className={`badge ${p.quinidine_used ? 'bg-success' : 'bg-secondary'}`} style={{ fontSize: '0.65rem' }}>
                      {p.quinidine_used ? 'Yes' : 'No'}
                    </span>
                  </td>
                  <td>{p.dyskinesia_episodes_per_month ?? '—'}</td>
                  <td>{p.qtc_baseline_ms ?? '—'}</td>
                  <td>
                    <span className="badge" style={{ background: p.severity_score >= 7 ? DANGER : p.severity_score >= 5 ? WARN : SUCCESS, fontSize: '0.65rem' }}>
                      {p.severity_score}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];

  return (
    <div>
      <div className="row g-3 mb-4">
        {seizures.map(s => (
          <div key={s.type} className="col-md-6">
            <SectionCard title={`${s.pct_affected}% — ${s.type}`} borderColor={s.type.includes('Dyskinesia') ? WARN : COLOR}>
              <p className="small mb-1">{s.semiology}</p>
              <p className="small mb-1"><strong>EEG:</strong> <span className="text-muted">{s.eeg_correlate}</span></p>
              <p className="small mb-1"><strong>Duration:</strong> {s.duration_sec}s</p>
              <div className="alert alert-info py-1 px-2 mb-0 small"><strong>💡 Tip:</strong> {s.clinical_tip}</div>
            </SectionCard>
          </div>
        ))}
      </div>

      <SectionCard title="⚡ Seizure Triggers">
        <div className="row g-3">
          {triggers.map(t => (
            <div key={t.trigger} className="col-md-6">
              <div className="border rounded p-2 mb-1 small">
                <div className="d-flex justify-content-between mb-1">
                  <strong>{t.trigger}</strong>
                  <span className="badge" style={{ background: t.pct >= 75 ? DANGER : t.pct >= 50 ? WARN : COLOR }}>
                    {t.pct}%
                  </span>
                </div>
                <p className="text-muted mb-1 small">{t.mechanism}</p>
                <p className="mb-0 small"><strong>Management:</strong> {t.management}</p>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatments || [];
  const contraindications = data.contraindications || [];
  const monitoring = data.monitoring || [];
  const lifecycle = data.lifecycle || [];

  return (
    <div>
      {/* Treatments */}
      {treatments.map(t => (
        <SectionCard key={t.drug} title={`${t.drug} — ${t.evidence}`} borderColor={t.drug === 'Quinidine' ? GOF_COLOR : COLOR}>
          <div className="row small">
            <div className="col-md-6">
              <p className="mb-1"><strong>Class:</strong> {t.class}</p>
              <p className="mb-1"><strong>Dose:</strong> {t.dose}</p>
              <p className="mb-1"><strong>MOA:</strong> {t.moa}</p>
              <p className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
            </div>
            <div className="col-md-6">
              <p className="mb-1"><strong>Monitoring:</strong> {t.monitoring}</p>
              <div className="alert alert-warning py-1 px-2 mb-0 small">
                <strong>🧬 KCNMA1 Note:</strong> {t.kcnma1_note}
              </div>
            </div>
          </div>
        </SectionCard>
      ))}

      {/* Contraindications */}
      <SectionCard title="⛔ Contraindications" borderColor={DANGER}>
        {contraindications.map(ci => (
          <div key={ci.drug} className="mb-3 border-bottom pb-2">
            <div className="fw-semibold text-danger small">{ci.drug}</div>
            <span className="badge mb-1" style={{ background: ci.level.includes('ABSOLUTE') ? DANGER : WARN, fontSize: '0.65rem' }}>
              {ci.level}
            </span>
            <p className="small text-muted mb-0">{ci.reason}</p>
          </div>
        ))}
      </SectionCard>

      {/* Monitoring */}
      <SectionCard title="🔬 Monitoring Items">
        <div className="row g-2">
          {monitoring.map(m => (
            <div key={m.item} className="col-md-6">
              <div className="border rounded p-2 small">
                <div className="fw-semibold">{m.item}</div>
                <div className="text-muted small">{m.frequency}</div>
                <div className="small">{m.why}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Lifecycle */}
      <SectionCard title="📅 Lifecycle Stages">
        {lifecycle.map(l => (
          <div key={l.stage} className="mb-3 border-bottom pb-2">
            <div className="fw-semibold small" style={{ color: COLOR }}>{l.stage} <span className="text-muted">({l.age})</span></div>
            <p className="small mb-1"><strong>Focus:</strong> {l.focus}</p>
            <p className="small mb-0 text-muted">{l.action}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const concepts = data.concepts || [];
  const thresholds = data.thresholds || [];
  const standards = data.evidence_standards || [];
  const distinctions = data.key_pharmacological_distinctions || [];

  return (
    <div>
      {/* Key pharmacological distinctions */}
      <SectionCard title="💊 Key Pharmacological Distinctions (KCNMA1)" borderColor={DANGER}>
        {distinctions.map((d, i) => (
          <div key={i} className="small mb-2 border-bottom pb-2 text-danger fw-semibold">• {d}</div>
        ))}
      </SectionCard>

      {/* Concepts */}
      <SectionCard title="📚 Key Concepts">
        <div className="row g-2">
          {concepts.map(c => (
            <div key={c.term} className="col-md-6">
              <div className="border rounded p-2 small">
                <div className="fw-semibold" style={{ color: COLOR }}>{c.term}</div>
                <div className="text-muted">{c.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="📏 Clinical Thresholds">
        <table className="table table-sm table-hover small mb-0">
          <thead className="table-light">
            <tr><th>Parameter</th><th>Value</th><th>Action</th></tr>
          </thead>
          <tbody>
            {thresholds.map(t => (
              <tr key={t.param}>
                <td className="fw-semibold">{t.param}</td>
                <td><span className="badge" style={{ background: COLOR }}>{t.value}</span></td>
                <td className="text-muted">{t.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </SectionCard>

      {/* Evidence Standards */}
      <SectionCard title="📋 Evidence Standards">
        <div className="row g-2">
          {standards.map(s => (
            <div key={s.standard} className="col-md-6">
              <div className="border rounded p-2 small">
                <div className="fw-semibold" style={{ color: COLOR }}>{s.standard}</div>
                <div className="text-muted">{s.scope}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

export default function KCNMA1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/kcnma1/overview`).then(r => r.json()).then(setOverview),
      fetch(`${API}/api/kcnma1/breakdown`).then(r => r.json()).then(setBreakdown),
      fetch(`${API}/api/kcnma1/definitions`).then(r => r.json()).then(setDefinitions),
    ];
    Promise.all(endpoints)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 KCNMA1 Epilepsy — BK Channel (MaxiK / Slo1)
          </h4>
          <div className="small text-muted">
            GOF → Epilepsy + Paroxysmal Dyskinesia (OMIM #609446) &nbsp;|&nbsp;
            LOF → Liang-Wang Syndrome (OMIM #618729) &nbsp;|&nbsp; 10q22.3 &nbsp;|&nbsp;
            <strong style={{ color: GOF_COLOR }}>Quinidine Precision (GOF)</strong> &nbsp;|&nbsp;
            <strong style={{ color: DANGER }}>Quinidine CI in LOF</strong>
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}
      {loading && <div className="alert alert-info small">Loading KCNMA1 data…</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
