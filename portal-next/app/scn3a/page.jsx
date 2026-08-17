'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#0d47a1';   // deep blue — NaV1.3 sodium channel
const ACCENT2 = '#b71c1c';   // dark red — CI / ABSOLUTE CI / NCSE
const ACCENT3 = '#e65100';   // deep orange — fever / alerts / thresholds
const ACCENT4 = '#1b5e20';   // deep green — seizure freedom / safe
const ACCENT5 = '#4a148c';   // purple — DEE67 / West / LGS / encephalopathy

function KPI({ label, value, color }) {
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

function AlertBox({ color, title, body }) {
  return (
    <div className="p-3 rounded mb-3" style={{ background: color + '18', border: `2px solid ${color}` }}>
      <strong style={{ color }}>{title}</strong>
      <div className="small mt-1" style={{ whiteSpace: 'pre-wrap' }}>{body}</div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="mb-4">
      <h5 className="fw-bold border-bottom pb-1 mb-3">{title}</h5>
      {children}
    </div>
  );
}

export default function SCN3APage() {
  const [tab, setTab] = useState(0);
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [df, setDf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/scn3a/overview`).then(r => r.json()),
      fetch(`${API}/api/scn3a/breakdown`).then(r => r.json()),
      fetch(`${API}/api/scn3a/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-5 text-center text-muted">Loading SCN3A dashboard…</div>;
  if (error) return <div className="p-5 text-danger">Error: {error}</div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="p-3 rounded mb-3 text-white" style={{ background: ACCENT }}>
        <h3 className="mb-0">🧬 SCN3A Epilepsy</h3>
        <div className="small opacity-75 mt-1">
          DEE67 (OMIM #619288) · NaV1.3 Channelopathy · Focal Epilepsy of Infancy · 2q24.3
        </div>
        <div className="small opacity-75">
          {ov.gene} · {ov.locus} · {ov.inheritance}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          <Section title="Protein & Mechanism">
            <div className="alert alert-primary small mb-2"><strong>{ov.protein}</strong></div>
            <p className="small">{ov.mechanism}</p>
            <AlertBox color={ACCENT} title="⚡ Key AHA" body={ov.key_aha} />
          </Section>

          <Section title="Cohort KPIs (n={ov.n_patients})">
            <div className="row g-2">
              <KPI label="Patients" value={ov.n_patients} color={ACCENT} />
              <KPI label="Seizure-Free %" value={`${ov.seizure_free_pct}%`} color={ACCENT4} />
              <KPI label="Drug-Resistant %" value={`${ov.drug_resistant_pct}%`} color={ACCENT2} />
              <KPI label="West History %" value={`${ov.west_history_pct}%`} color={ACCENT5} />
              <KPI label="On ACTH %" value={`${ov.on_acth_pct}%`} color={ACCENT} />
              <KPI label="On KD %" value={`${ov.on_kd_pct}%`} color={ACCENT3} />
              <KPI label="On VPA %" value={`${ov.on_vpa_pct}%`} color={ACCENT} />
              <KPI label="On CBZ/OXC %" value={`${ov.on_cbz_oxc_pct}%`} color={ACCENT5} />
              <KPI label="POLG Done %" value={`${ov.polg_done_pct}%`} color={ACCENT4} />
              <KPI label="VPA w/o POLG" value={ov.vpa_without_polg} color={ACCENT2} />
              <KPI label="PMG (R357Q) %" value={`${ov.pmg_pct}%`} color={ACCENT5} />
              <KPI label="CSWS %" value={`${ov.csws_pct}%`} color={ACCENT3} />
            </div>
          </Section>

          <Section title="Critical Alerts">
            <AlertBox color={ACCENT2} title="⛔ TIAGABINE — ABSOLUTE CI in ALL SCN3A-DEE67" body={ov.tiagabine_alert} />
            <AlertBox color={ACCENT2} title="⚠️ CBZ/OXC — HIGH RISK during West/Spasms" body={ov.cbz_oxc_west_alert} />
            <AlertBox color={ACCENT2} title="🧬 HLA-B*15:02 MANDATORY before CBZ/OXC (Asian ancestry)" body={ov.hla_alert} />
            <AlertBox color={ACCENT3} title="🔬 POLG MANDATORY before VPA" body={ov.polg_alert} />
            <AlertBox color={ACCENT5} title="🧠 R357Q → 3T MRI MANDATORY" body={ov.r357q_alert} />
            <AlertBox color={ACCENT3} title="💊 Quinidine — Investigational (QT monitoring)" body={ov.quinidine_alert} />
          </Section>

          <Section title="Contraindications Summary">
            {ov.contraindications_summary.map((ci, i) => (
              <div key={i} className="p-2 mb-2 rounded small" style={{ background: ACCENT2 + '15', border: `1px solid ${ACCENT2}` }}>
                ⛔ {ci}
              </div>
            ))}
          </Section>

          <Section title="Thresholds">
            <ul className="list-group list-group-flush small">
              {ov.thresholds.map((th, i) => (
                <li key={i} className="list-group-item">{th}</li>
              ))}
            </ul>
          </Section>
        </div>
      )}

      {/* ── TAB 1: Patients & Etiology ── */}
      {tab === 1 && (
        <div>
          <Section title="Etiology Distribution (5 classes)">
            {bk.etiology_distribution.map((e, i) => (
              <div key={i} className="card mb-3 shadow-sm">
                <div className="card-header fw-bold" style={{ background: ACCENT + '18' }}>
                  {e.etiology} — <span style={{ color: ACCENT2 }}>{e.pct}% (n={e.n})</span>
                </div>
                <div className="card-body small">
                  <p className="mb-1"><strong>Mechanism:</strong> {e.mechanism_summary}</p>
                  <p className="mb-1"><strong>EEG:</strong> {e.eeg}</p>
                  <p className="mb-0"><strong>MRI:</strong> {e.mri}</p>
                </div>
              </div>
            ))}
          </Section>

          <Section title="Sample Patients (first 15)">
            <div className="table-responsive">
              <table className="table table-sm table-bordered small">
                <thead style={{ background: ACCENT, color: 'white' }}>
                  <tr>
                    <th>ID</th><th>Etiology</th><th>Onset (mo)</th><th>Sex</th>
                    <th>Sz-Free</th><th>DRE</th><th>West</th><th>On KD</th>
                    <th>On VPA</th><th>POLG</th><th>VPA w/o POLG</th><th>PMG</th><th>CSWS</th>
                  </tr>
                </thead>
                <tbody>
                  {bk.sample_patients.map((p, i) => (
                    <tr key={i} style={{ background: p.vpa_no_polg ? ACCENT2 + '20' : 'inherit' }}>
                      <td>{p.id}</td>
                      <td>{p.etiology_short}</td>
                      <td>{p.onset_months}</td>
                      <td>{p.sex}</td>
                      <td>{p.seizure_free ? '✅' : '—'}</td>
                      <td style={{ color: p.drug_resistant ? ACCENT2 : 'inherit' }}>{p.drug_resistant ? '⚠️ DRE' : '—'}</td>
                      <td>{p.west ? '🧠' : '—'}</td>
                      <td>{p.on_kd ? '🥑' : '—'}</td>
                      <td>{p.on_vpa ? '💊' : '—'}</td>
                      <td style={{ color: p.polg === 'N' ? ACCENT2 : ACCENT4 }}>{p.polg}</td>
                      <td style={{ color: ACCENT2 }}>{p.vpa_no_polg ? '⛔ YES' : '—'}</td>
                      <td>{p.pmg ? '🧬 PMG' : '—'}</td>
                      <td>{p.csws ? '⚡ CSWS' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Lifecycle Windows">
            {bk.lifecycle.map((lc, i) => (
              <div key={i} className="p-2 mb-2 rounded small" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}33` }}>
                <strong>{lc.window} ({lc.ages}):</strong> {lc.events}
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 2: Seizure Types & Triggers ── */}
      {tab === 2 && (
        <div>
          <Section title="Seizure Types">
            {bk.seizure_types.map((st, i) => (
              <div key={i} className="card mb-3 shadow-sm">
                <div className="card-header d-flex justify-content-between align-items-center">
                  <span className="fw-bold">{st.type}</span>
                  <span className="badge" style={{ background: ACCENT5 }}>{st.pct}% of patients</span>
                </div>
                <div className="card-body small">
                  <p className="mb-1"><strong>EEG:</strong> {st.eeg}</p>
                  <p className="mb-0"><strong>Clinical Tip:</strong> {st.tip}</p>
                </div>
              </div>
            ))}
          </Section>

          <Section title="Seizure Triggers">
            {bk.triggers.map((tr, i) => (
              <div key={i} className="card mb-2 shadow-sm">
                <div className="card-header d-flex justify-content-between align-items-center">
                  <span className="fw-bold">{tr.trigger}</span>
                  <span className="badge bg-warning text-dark">{tr.pct}%</span>
                </div>
                <div className="card-body small">
                  <p className="mb-1"><strong>Threshold:</strong> {tr.threshold}</p>
                  <p className="mb-0"><strong>Management:</strong> {tr.mgmt}</p>
                </div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── TAB 3: Treatments ── */}
      {tab === 3 && (
        <div>
          <Section title="Contraindications (Safety First)">
            {bk.contraindications.map((ci, i) => (
              <div key={i} className="card mb-3 border-danger shadow-sm">
                <div className="card-header text-white fw-bold" style={{ background: ci.risk === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3 }}>
                  ⛔ {ci.drug} — {ci.risk}
                </div>
                <div className="card-body small">
                  <p className="mb-1"><strong>Reason:</strong> {ci.reason}</p>
                  <p className="mb-0"><strong>Alternative:</strong> {ci.alternative}</p>
                </div>
              </div>
            ))}
          </Section>

          <Section title="Treatment Protocols (7 drugs)">
            {bk.treatments.map((t, i) => (
              <div key={i} className="card mb-3 shadow-sm">
                <div className="card-header d-flex justify-content-between align-items-center" style={{ background: ACCENT + '15' }}>
                  <span className="fw-bold">{t.drug}</span>
                  <span className="badge" style={{ background: ACCENT }}>{t.evidence}</span>
                </div>
                <div className="card-body small">
                  <p className="mb-1"><strong>Indication:</strong> {t.indication}</p>
                  <p className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
                  <div className="p-2 rounded" style={{ background: ACCENT5 + '12', border: `1px solid ${ACCENT5}` }}>
                    <strong>SCN3A-specific note:</strong> {t.scn3a_note}
                  </div>
                </div>
              </div>
            ))}
          </Section>

          <Section title="Monitoring Checklist">
            <ul className="list-group list-group-flush small">
              {bk.monitoring.map((m, i) => (
                <li key={i} className="list-group-item">
                  <strong>{m.item}</strong>
                  <span className="text-muted ms-2">({m.frequency})</span>
                  <div className="text-muted">{m.rationale}</div>
                </li>
              ))}
            </ul>
          </Section>
        </div>
      )}

      {/* ── TAB 4: Definitions ── */}
      {tab === 4 && (
        <div>
          <Section title="Key Concepts (15)">
            {df.concepts.map((c, i) => (
              <div key={i} className="p-2 mb-2 rounded small" style={{ background: ACCENT + '10', border: `1px solid ${ACCENT}33` }}>
                <strong style={{ color: ACCENT }}>{c.term}:</strong> {c.definition}
              </div>
            ))}
          </Section>

          <Section title="Thresholds">
            <ul className="list-group list-group-flush small">
              {df.thresholds.map((th, i) => (
                <li key={i} className="list-group-item">{th}</li>
              ))}
            </ul>
          </Section>

          <Section title="Clinical Standards">
            <ul className="list-group list-group-flush small">
              {df.standards.map((s, i) => (
                <li key={i} className="list-group-item">{s}</li>
              ))}
            </ul>
          </Section>

          <Section title="References">
            <ul className="list-group list-group-flush small">
              {df.references.map((r, i) => (
                <li key={i} className="list-group-item">{r}</li>
              ))}
            </ul>
          </Section>
        </div>
      )}
    </div>
  );
}
