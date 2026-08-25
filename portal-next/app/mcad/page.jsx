'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// MCAD / ACADM colour scheme — blue-teal (FAO disorder; fasting hazard)
const ACCENT  = '#01579b';   // deep blue — MCAD / FAO primary
const ACCENT2 = '#006064';   // dark teal — C8 primary NBS marker
const ACCENT3 = '#e65100';   // deep orange — FASTING ABSOLUTE CI / hypoketotic crisis
const ACCENT4 = '#1b5e20';   // deep green — KEY NEGATIVES (C14 normal, ketones low)
const ACCENT5 = '#880e4f';   // dark rose — pathognomonic urine markers (HG/SG/PPG)
const ACCENT6 = '#b71c1c';   // deep red — HIGH RISK / ABSOLUTE CI
const ACCENT7 = '#37474f';   // dark slate — NBS / epidemiology
const ACCENT8 = '#4a148c';   // dark purple — founder variant c.985A>G

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

function PctBar({ label, pct, color = ACCENT }) {
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function InfoBox({ title, children, color = ACCENT }) {
  return (
    <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2">
        <div className="fw-bold small mb-1" style={{ color }}>{title}</div>
        <div className="small text-muted">{children}</div>
      </div>
    </div>
  );
}

export default function MCADPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mcad/overview`).then(r => r.json()),
      fetch(`${API}/api/mcad/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mcad/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading MCAD Dashboard…</div>;
  if (err)     return <div className="p-4 text-center text-danger">Error: {err}</div>;

  const kpis = ov?.kpis || {};
  const phDist = ov?.phenotype_distribution || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          &#x1f9ec; MCAD Deficiency Dashboard
        </h4>
        <div className="text-muted small">
          Medium-Chain Acyl-CoA Dehydrogenase Deficiency &mdash; ACADM &middot; 1p31.1 &middot; AR &middot; OMIM #201450
        </div>
        <div className="text-muted small">
          Most common FAO disorder &middot; C8 (octanoylcarnitine) PRIMARY NBS MARKER &middot; HYPOketotic hypoglycaemia hallmark
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

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          <Alert
            text="⚠ FASTING: ABSOLUTE CONTRAINDICATION — primary and most preventable trigger. Never fast >4h (infants) / >8h (children) / >12h (adults). Emergency IV Glucose 10% during any illness + anorexia."
            variant="danger"
          />
          <Alert
            text="⚠ KETOGENIC DIET: ABSOLUTE CI — KD requires fasting intervals → triggers medium-chain FAO crisis → hypoketotic hypoglycaemia. VPA: HIGH RISK — inhibits FAO, depletes carnitine."
            variant="warning"
          />

          {/* KPI strip */}
          <div className="row mb-3">
            <KPI label="Total Patients"     value={kpis.total_patients}   color={ACCENT}  />
            <KPI label="NBS-Detected"       value={kpis.nbs_detected}     color={ACCENT2} />
            <KPI label="Hypoketotic Crisis" value={kpis.crisis_n}         color={ACCENT3} />
            <KPI label="Seizures"           value={kpis.seizures_n}       color={ACCENT6} />
            <KPI label="Avg C8 (µmol/L)"   value={kpis.avg_c8_umol}      color={ACCENT5} />
            <KPI label="Avg C0 (µmol/L)"   value={kpis.avg_c0_umol}      color={ACCENT7} />
          </div>

          {/* Phenotype distribution */}
          <div className="row mb-3">
            <div className="col-md-5">
              <div className="card shadow-sm p-3">
                <div className="fw-bold mb-2" style={{ color: ACCENT }}>Phenotype Distribution</div>
                {Object.entries(phDist).map(([ph, n]) => (
                  <PctBar
                    key={ph}
                    label={ph}
                    pct={Math.round(100 * n / (kpis.total_patients || 40))}
                    color={
                      ph.includes('NBS')      ? ACCENT2 :
                      ph.includes('Crisis')   ? ACCENT3 :
                      ph.includes('SUDS')     ? ACCENT6 :
                      ACCENT7
                    }
                  />
                ))}
              </div>
            </div>
            <div className="col-md-7">
              <div className="card shadow-sm p-3 h-100">
                <div className="fw-bold mb-2" style={{ color: ACCENT }}>Gene &amp; Enzyme</div>
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>{ov?.gene}</td></tr>
                    <tr><td className="fw-bold">Locus</td><td>{ov?.locus}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*{ov?.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#{ov?.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>{ov?.protein}</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>{ov?.prevalence}</td></tr>
                    <tr><td className="fw-bold">Primary NBS</td><td style={{ color: ACCENT2 }}>{ov?.primary_nbs_marker}</td></tr>
                    <tr><td className="fw-bold">Pathognomonic Urine</td><td style={{ color: ACCENT5 }}>{(ov?.pathognomonic_urine || []).join(' + ')}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Key negatives */}
          <div className="card shadow-sm p-3 mb-3">
            <div className="fw-bold mb-2" style={{ color: ACCENT4 }}>KEY NEGATIVES (differentiation)</div>
            <div className="row">
              {Object.entries(ov?.key_negatives || {}).map(([k, v]) => (
                <div key={k} className="col-md-6 mb-2">
                  <InfoBox title={k.replace(/_/g,' ').toUpperCase()} color={ACCENT4}>{v}</InfoBox>
                </div>
              ))}
            </div>
          </div>

          {/* Clinical summary */}
          <InfoBox title="Clinical Summary" color={ACCENT}>
            {ov?.clinical_summary}
          </InfoBox>

          {/* Founder variant */}
          <InfoBox title={`Founder Variant: ${ov?.top_variant} (${ov?.top_variant_pct}% of cohort)`} color={ACCENT8}>
            c.985A&gt;G (p.Lys329Glu) — the dominant allele in Northern European populations; accounts for ~80–90% of Northern European MCAD alleles.
            FAD-binding domain; pathogenic; reliably detected on targeted mutation panel.
          </InfoBox>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && (
        <div>
          <Alert
            text="BIOMARKER PROFILE: C8 ↑↑↑ (PRIMARY NBS) + C6 ↑ + C10:1 ↑ + C10 ↑ + C0 ↓ (secondary depletion) + HYPOketotic hypoglycaemia + Hexanoylglycine + Suberylglycine + PPG (urine pathognomonic)"
            variant="info"
          />

          {/* Biomarker reference table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Biomarker Reference Panel</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Biomarker</th><th>Normal</th><th>MCAD Status</th><th>Direction</th><th>Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(bd?.biomarkers || {}).map(([k, bm]) => (
                    <tr key={k}>
                      <td className="fw-bold">{bm.label}</td>
                      <td className="text-muted">{bm.normal}</td>
                      <td>
                        <span className={`badge bg-${bm.color}`}>{bm.status}</span>
                      </td>
                      <td className="fw-bold" style={{
                        color: bm.direction.startsWith('↑') ? '#b71c1c' :
                               bm.direction.startsWith('↓') ? '#e65100' :
                               '#1b5e20'
                      }}>{bm.direction}</td>
                      <td style={{ maxWidth: 350 }}>{bm.rationale?.substring(0, 180)}…</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Phenotype-biomarker patterns */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Phenotype–Biomarker Patterns</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Phenotype</th><th>Prevalence</th><th>C8</th><th>Glucose</th><th>β-OHB (Ketones)</th><th>Hexanoylglycine</th><th>Prognosis</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.phenotype_patterns || []).map((pp, i) => (
                    <tr key={i}>
                      <td className="fw-bold">{pp.phenotype}</td>
                      <td>{pp.prevalence}</td>
                      <td>{pp.c8}</td>
                      <td style={{ color: pp.glucose?.includes('LOW') ? '#b71c1c' : 'inherit' }}>{pp.glucose}</td>
                      <td style={{ color: pp.bohb?.toLowerCase().includes('low') ? '#e65100' : 'inherit' }}>{pp.bohb}</td>
                      <td>{pp.hg}</td>
                      <td>{pp.prognosis?.substring(0, 80)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Patient sample */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Patient Cohort Sample (n=10 representative)</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr>
                    <th>ID</th><th>Phenotype</th><th>Onset</th><th>C8</th><th>C6</th><th>C10</th><th>C8/C10</th><th>C0</th><th>Glucose</th><th>β-OHB</th><th>HG</th><th>SG</th><th>Variant</th><th>Seizures</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.patient_sample || []).map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td style={{ maxWidth: 120 }}>{p.phenotype?.substring(0,25)}</td>
                      <td>{p.onset_mo === 0 ? 'NBS' : `${p.onset_mo}m`}</td>
                      <td style={{ color: p.c8 > 0.3 ? '#b71c1c' : 'inherit', fontWeight: 'bold' }}>{p.c8}</td>
                      <td>{p.c6}</td>
                      <td>{p.c10}</td>
                      <td style={{ color: p.c8_c10_ratio > 2 ? ACCENT2 : 'inherit' }}>{p.c8_c10_ratio}</td>
                      <td style={{ color: p.c0 < 20 ? ACCENT3 : 'inherit' }}>{p.c0}</td>
                      <td style={{ color: p.glucose < 3.5 ? '#b71c1c' : 'inherit' }}>{p.glucose}</td>
                      <td style={{ color: p.bohb < 0.5 ? ACCENT3 : 'inherit' }}>{p.bohb}</td>
                      <td>{p.hg}</td>
                      <td>{p.sg}</td>
                      <td style={{ color: ACCENT8, fontSize: 11 }}>{p.variant?.substring(0,20)}</td>
                      <td>{p.seizures ? '✓' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Variant table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT8 }}>Pathogenic Variants (ACADM)</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr><th>Variant</th><th>Freq (%)</th><th>Domain</th><th>Phenotype</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {(bd?.variant_table || []).map((v, i) => (
                    <tr key={i} style={{ background: i === 0 ? '#e8f5e9' : 'inherit' }}>
                      <td className="fw-bold" style={{ color: ACCENT8 }}>{v.variant}</td>
                      <td>{v.freq}%</td>
                      <td>{v.domain}</td>
                      <td>{v.phenotype}</td>
                      <td>{v.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="card-footer text-muted small">
              c.985A&gt;G (p.Lys329Glu) highlighted — Northern European founder mutation accounting for ~80–90% of alleles in that population.
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TREATMENTS ── */}
      {tab === 2 && (
        <div>
          <Alert
            text="SEIZURES IN MCAD: Primarily HYPOGLYCAEMIC seizures during acute Reye-like crisis. Between crises: no seizure tendency. Prevent by preventing hypoglycaemia (fasting avoidance + emergency glucose protocol)."
            variant="warning"
          />

          {/* Treatment table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Treatment Protocol &amp; Contraindications</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead className="table-light">
                  <tr><th>Intervention</th><th>Evidence Level</th><th>Rationale</th></tr>
                </thead>
                <tbody>
                  {(bd?.treatment_table || []).map((tx, i) => (
                    <tr key={i} style={{
                      background:
                        tx.contraindication === 'ABSOLUTE CI' ? '#ffebee' :
                        tx.contraindication === 'HIGH RISK'   ? '#fff8e1' :
                        tx.level?.includes('Level A')         ? '#e8f5e9' : 'inherit'
                    }}>
                      <td className="fw-bold" style={{
                        color:
                          tx.contraindication === 'ABSOLUTE CI' ? '#b71c1c' :
                          tx.contraindication === 'HIGH RISK'   ? '#e65100' :
                          tx.level?.includes('Level A')         ? '#1b5e20' : 'inherit'
                      }}>{tx.intervention}</td>
                      <td>
                        <span className={`badge ${
                          tx.contraindication === 'ABSOLUTE CI' ? 'bg-danger' :
                          tx.contraindication === 'HIGH RISK'   ? 'bg-warning text-dark' :
                          tx.level?.includes('Level A')         ? 'bg-success' :
                          tx.level?.includes('Level B')         ? 'bg-primary' : 'bg-secondary'
                        }`}>{tx.level}</span>
                      </td>
                      <td>{tx.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Key differentials */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Key Differentials</div>
            <div className="row p-3">
              {Object.entries(bd?.key_differentials || {}).map(([k, v]) => (
                <div key={k} className="col-md-6 mb-2">
                  <InfoBox title={k.replace(/_/g,' ')} color={ACCENT3}>{v}</InfoBox>
                </div>
              ))}
            </div>
          </div>

          {/* Exam pearls */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT5 }}>Exam Pearls</div>
            <ul className="list-group list-group-flush small">
              {(bd?.exam_pearls || []).map((pearl, i) => (
                <li key={i} className="list-group-item py-1">{pearl}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && (
        <div>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm p-3">
                <div className="fw-bold mb-2" style={{ color: ACCENT }}>Disease Identity</div>
                <table className="table table-sm small mb-0">
                  <tbody>
                    {[
                      ['Disease', def?.disease_name],
                      ['Gene',    def?.gene],
                      ['Locus',   def?.locus],
                      ['OMIM Gene', def?.omim_gene],
                      ['OMIM Disease', def?.omim_disease],
                      ['Inheritance', def?.inheritance],
                      ['Protein', def?.protein],
                      ['Prevalence', def?.prevalence],
                      ['Pathway', def?.pathway],
                      ['Founder Variant', def?.founder_variant],
                    ].map(([label, value]) => (
                      <tr key={label}><td className="fw-bold">{label}</td><td>{value}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm p-3">
                <div className="fw-bold mb-2" style={{ color: ACCENT2 }}>Enzymatic Function</div>
                <p className="small text-muted">{def?.enzymatic_function}</p>
                <div className="fw-bold mb-1" style={{ color: ACCENT3 }}>Metabolic Block</div>
                <p className="small text-muted">{def?.metabolic_block}</p>
                <div className="fw-bold mb-1" style={{ color: ACCENT5 }}>NBS Marker</div>
                <p className="small text-muted">{def?.nbs_marker}</p>
              </div>
            </div>
          </div>

          {/* Confirmatory biomarkers */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT }}>Confirmatory Biomarkers</div>
            <div className="card-body small">
              {Object.entries(def?.confirmatory_biomarkers || {}).map(([k, v]) => (
                <InfoBox key={k} title={k.replace(/_/g,' ').toUpperCase()} color={ACCENT2}>{v}</InfoBox>
              ))}
            </div>
          </div>

          {/* Key exam facts */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT5 }}>Key Exam Facts</div>
            <ul className="list-group list-group-flush small">
              {(def?.key_exam_facts || []).map((fact, i) => (
                <li key={i} className="list-group-item py-1">{fact}</li>
              ))}
            </ul>
          </div>

          {/* Glossary */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT7 }}>Glossary</div>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <tbody>
                  {Object.entries(def?.glossary || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-nowrap" style={{ color: ACCENT }}>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* References */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ color: ACCENT7 }}>References</div>
            <ul className="list-group list-group-flush small">
              {(def?.references || []).map((r, i) => (
                <li key={i} className="list-group-item py-1">{r}</li>
              ))}
            </ul>
          </div>

          <div className="text-muted small text-end mt-2">
            MCAD Dashboard &mdash; ACADM (1p31.1, AR, OMIM #201450) &mdash; 40-patient cohort seed-269 &mdash; 3 endpoints verified
          </div>
        </div>
      )}
    </div>
  );
}
