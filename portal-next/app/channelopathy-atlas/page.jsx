'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// Channelopathy-Atlas color palette — ion channels / electricity
const COLOR  = '#1565c0';  // deep blue — ion channels / electricity
const LIGHT  = '#e3f2fd';  // light blue tint
const COLOR2 = '#b71c1c';  // red — cardiac arrhythmia / danger
const COLOR3 = '#2e7d32';  // dark green — muscle channelopathy / myotonia
const COLOR4 = '#e65100';  // orange — caution / periodic paralysis
const COLOR5 = '#6a1b9a';  // purple — calcium channels
const COLOR6 = '#37474f';  // blue-grey — moderate
const COLOR7 = '#880e4f';  // dark magenta — HERG / LQT2

const GENE_COLORS = {
  SCN4A:   '#1565c0',  // blue — Nav1.4 sodium skeletal
  CACNA1S: '#6a1b9a',  // purple — Cav1.1 calcium
  KCNJ2:   '#2e7d32',  // dark green — Kir2.1 ATS
  CLCN1:   '#e65100',  // orange — ClC-1 chloride
  KCNQ1:   '#b71c1c',  // red — LQT1/JLNS cardiac
  KCNH2:   '#880e4f',  // dark magenta — LQT2/HERG
  SCN5A:   '#37474f',  // blue-grey — Brugada/Nav1.5
  RYR2:    '#bf360c',  // deep orange — CPVT/RyR2
};

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color = COLOR, note }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{typeof pct === 'number' ? `${pct}%` : pct}{note ? ` — ${note}` : ''}</span>
      </div>
      {typeof pct === 'number' && (
        <div className="progress" style={{ height: 8 }}>
          <div className="progress-bar" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
        </div>
      )}
    </div>
  );
}

function AlertBox({ type = 'info', title, children }) {
  const icons = { danger: '🚨', warning: '⚠️', info: 'ℹ️', success: '✅' };
  return (
    <div className={`alert alert-${type} py-2 px-3 mb-3`}>
      <strong>{icons[type]} {title}</strong>
      <div className="small mt-1">{children}</div>
    </div>
  );
}

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border" style={{ color: COLOR }} />
      <div className="mt-2 text-muted small">Loading Channelopathy-Atlas…</div>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger">{msg}</div>;
}

// ── Tab: Overview ──────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const cf = ov.clinical_features_prevalence || {};
  const sev = ov.severity || {};
  const cat = ov.channel_category_breakdown || {};
  const kpis = ov.kpis || [];

  return (
    <div>
      <h5 className="fw-bold mb-1" style={{ color: COLOR }}>{ov.full_name}</h5>
      <p className="text-muted small mb-3">{ov.subtitle}</p>
      <p className="mb-3">{ov.description}</p>

      {/* Drug alerts */}
      {(ov.drug_alerts || []).map((a, i) => (
        <AlertBox key={i}
          type={a.includes('ABSOLUTELY') || a.includes('CONTRAINDICATED') || a.includes('FEVER') || a.includes('PARADOX') ? 'danger' : 'warning'}
          title="Drug / Treatment Alert">
          {a}
        </AlertBox>
      ))}

      {/* KPI cards */}
      <div className="row g-2 mb-4">
        {kpis.map((k, i) => (
          <KPI key={i} label={k.label} value={k.value} color={k.color || COLOR} />
        ))}
      </div>

      <div className="row g-3 mb-4">
        {/* Severity */}
        <div className="col-md-4">
          <div className="card h-100">
            <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Severity Distribution</div>
            <div className="card-body">
              <BarRow label="Mild" pct={sev.mild_pct} color={COLOR3} />
              <BarRow label="Moderate" pct={sev.moderate_pct} color={COLOR4} />
              <BarRow label="Severe" pct={sev.severe_pct} color={COLOR2} />
            </div>
          </div>
        </div>

        {/* Clinical features */}
        <div className="col-md-8">
          <div className="card h-100">
            <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Clinical Features (cohort-wide)</div>
            <div className="card-body">
              <BarRow label="Arrhythmia" pct={cf.arrhythmia_pct} color={COLOR2} />
              <BarRow label="Myotonia" pct={cf.myotonia_pct} color={COLOR3} />
              <BarRow label="Periodic Paralysis" pct={cf.periodic_paralysis_pct} color={COLOR4} />
              <BarRow label="MH Risk" pct={cf.mh_risk_pct} color={COLOR2} />
              <BarRow label="Bidirectional VT" pct={cf.bidirectional_vt_pct} color={COLOR2} />
              <BarRow label="ICD Implanted" pct={cf.icd_implanted_pct} color={COLOR6} />
              <BarRow label="Beta-Blocker" pct={cf.beta_blocker_pct} color={COLOR} />
              <BarRow label="Mexiletine" pct={cf.mexiletine_pct} color={COLOR3} />
              <BarRow label="Flecainide" pct={cf.flecainide_pct} color={COLOR5} />
              <BarRow label="Juvenile Onset (<18y)" pct={cf.juvenile_onset_pct} color={COLOR7} />
            </div>
          </div>
        </div>
      </div>

      {/* Channel Category Breakdown */}
      <div className="card mb-4">
        <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Ion Channel Gene Categories</div>
        <div className="card-body">
          <div className="row g-2">
            {Object.entries(cat).map(([category, genes]) => (
              <div key={category} className="col-md-6">
                <div className="border rounded p-2 h-100"
                  style={{ borderLeft: `4px solid ${GENE_COLORS[genes[0]] || COLOR}` }}>
                  <div className="small fw-semibold mb-1" style={{ color: GENE_COLORS[genes[0]] || COLOR }}>
                    {genes.join(', ')}
                  </div>
                  <div className="small text-muted">{category}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Key Teaching Points */}
      <div className="card mb-4">
        <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Key Teaching Points</div>
        <div className="card-body">
          {(ov.key_teaching_points || []).map((pt, i) => (
            <div key={i} className="mb-2 small border-start ps-2"
              style={{ borderColor: COLOR }}>
              {pt}
            </div>
          ))}
        </div>
      </div>

      {/* Standards */}
      {(ov.standards || []).length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Clinical Standards &amp; References</div>
          <div className="card-body">
            {ov.standards.map((s, i) => (
              <div key={i} className="mb-1 small">⚡ {s}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];
  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Channelopathy-Atlas — 8-Gene Clinical Reference Table</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered table-hover small">
          <thead style={{ backgroundColor: LIGHT }}>
            <tr>
              <th>Gene</th><th>Protein / Size</th><th>Locus</th><th>Inheritance</th>
              <th>Channel Type / Disease</th><th>Onset (y)</th>
              <th>Cardiac?</th><th>Myotonia?</th><th>PP?</th>
              <th>MH Risk?</th><th>BidVT?</th><th>First-Line Drug</th>
            </tr>
          </thead>
          <tbody>
            {genes.map((g) => (
              <tr key={g.gene}>
                <td>
                  <span className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                </td>
                <td>{g.protein}<br /><span className="text-muted">{g.aa} · {g.kDa}</span></td>
                <td><code className="small">{g.locus}</code></td>
                <td className="small">{g.inheritance?.split('.')[0]}</td>
                <td className="small">{g.channel_type?.split('(')[0]?.trim()}</td>
                <td className="text-center">{g.onset_range_y?.[0]}–{g.onset_range_y?.[1]}</td>
                <td className="text-center">{g.cardiac_risk ? '🔴 Yes' : '—'}</td>
                <td className="text-center">{g.myotonia ? '🟠 Yes' : '—'}</td>
                <td className="text-center">{g.periodic_paralysis ? '⚠️ Yes' : '—'}</td>
                <td className="text-center">{g.mh_risk ? '🔴 MH' : '—'}</td>
                <td className="text-center">{g.bidirectional_vt ? '🔴 Yes' : '—'}</td>
                <td className="small fw-semibold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>
                  {g.first_line_drug}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small text-muted">
        PP = Periodic Paralysis; BidVT = Bidirectional VT (pathognomonic in KCNJ2 ATS and RYR2 CPVT);
        MH = Malignant Hyperthermia susceptibility (CACNA1S);
        CLCN1: QUININE/QUINIDINE ABSOLUTELY CONTRAINDICATED.
      </p>

      {/* Pharmacology matrix */}
      {data.pharmacology_matrix && (
        <div className="card mt-3">
          <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Pharmacology Matrix</div>
          <div className="card-body">
            <div className="table-responsive">
              <table className="table table-sm table-bordered small mb-0">
                <thead style={{ backgroundColor: LIGHT }}>
                  <tr><th>Drug / Class</th><th>Indication (gene)</th></tr>
                </thead>
                <tbody>
                  {Object.entries(data.pharmacology_matrix).map(([drug, genes]) => (
                    <tr key={drug}>
                      <td className="fw-semibold"
                        style={{ color: drug.includes('CI') || drug.includes('AVOID') ? '#b71c1c' : '#1565c0' }}>
                        {drug.replace(/_/g, ' ')}
                      </td>
                      <td className="small">{genes.join(' · ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Tab: Clinical Atlas ────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(null);
  const genes = data.genes || [];
  const gene = selected ? genes.find(g => g.gene === selected) : null;

  return (
    <div className="row g-3">
      {/* Gene selector */}
      <div className="col-md-3">
        <div className="list-group">
          {genes.map((g) => (
            <button key={g.gene}
              className={`list-group-item list-group-item-action py-2 ${selected === g.gene ? 'active' : ''}`}
              style={selected === g.gene
                ? { backgroundColor: GENE_COLORS[g.gene] || COLOR, borderColor: GENE_COLORS[g.gene] || COLOR }
                : {}}
              onClick={() => setSelected(g.gene)}>
              <span className="fw-bold">{g.gene}</span>
              <div className="small">{g.channel_type?.split('(')[0]?.trim()}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Gene detail */}
      <div className="col-md-9">
        {!gene ? (
          <div className="text-muted small py-4 text-center">Select a gene to view clinical detail</div>
        ) : (
          <div>
            <h6 className="fw-bold mb-1" style={{ color: GENE_COLORS[gene.gene] || COLOR }}>
              {gene.gene} — {gene.protein}
            </h6>
            <p className="small text-muted mb-2">{gene.alias}</p>

            {/* Critical avoid alert */}
            {gene.critical_avoid && (
              <AlertBox type="danger" title={`Critical: Avoid in ${gene.gene}`}>
                {gene.critical_avoid}
              </AlertBox>
            )}

            <div className="row g-2 mb-3">
              {[
                { label: 'Locus', value: gene.locus },
                { label: 'Size', value: `${gene.aa} · ${gene.kDa}` },
                { label: 'Onset', value: `${gene.onset_range_y?.[0]}–${gene.onset_range_y?.[1]}y` },
                { label: 'OMIM Gene', value: gene.omim_gene },
                { label: 'OMIM Disease', value: gene.omim_disease },
                { label: 'Mean CK', value: `${(gene.mean_ck_iu_l || 0).toLocaleString()} IU/L` },
                { label: 'First-Line Rx', value: gene.first_line_drug },
              ].map(({ label, value }) => (
                <div key={label} className="col-6 col-md-4">
                  <div className="card text-center py-1">
                    <div className="fw-bold small" style={{ color: GENE_COLORS[gene.gene] || COLOR }}>{value}</div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{label}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Feature badges */}
            <div className="row g-2 mb-2">
              {[
                { label: 'Cardiac Risk', val: gene.cardiac_risk },
                { label: 'Arrhythmia', val: gene.arrhythmia_risk },
                { label: 'Myotonia', val: gene.myotonia },
                { label: 'Periodic Paralysis', val: gene.periodic_paralysis },
                { label: 'MH Risk', val: gene.mh_risk },
                { label: 'Deafness Risk', val: gene.deafness_risk },
                { label: 'Juvenile Onset', val: gene.juvenile_onset },
                { label: 'Bidirectional VT', val: gene.bidirectional_vt },
              ].map(({ label, val }) => (
                <div key={label} className="col-auto">
                  <span className={`badge ${val ? 'bg-danger' : 'bg-secondary'}`}>
                    {label}: {val ? 'Yes' : 'No'}
                  </span>
                </div>
              ))}
            </div>

            {/* Severity + clinical features */}
            <div className="row g-3 mb-3">
              <div className="col-md-4">
                <div className="card">
                  <div className="card-header small fw-bold" style={{ backgroundColor: LIGHT }}>Severity</div>
                  <div className="card-body">
                    <BarRow label="Mild" pct={gene.severity_distribution?.mild_pct} color={COLOR3} />
                    <BarRow label="Moderate" pct={gene.severity_distribution?.moderate_pct} color={COLOR4} />
                    <BarRow label="Severe" pct={gene.severity_distribution?.severe_pct} color={COLOR2} />
                  </div>
                </div>
              </div>
              <div className="col-md-8">
                <div className="card">
                  <div className="card-header small fw-bold" style={{ backgroundColor: LIGHT }}>Clinical Features (n=40)</div>
                  <div className="card-body">
                    {Object.entries(gene.clinical_features || {}).map(([k, v]) => (
                      <BarRow key={k}
                        label={k.replace(/_pct$/, '').replace(/_/g, ' ')}
                        pct={v}
                        color={
                          k.includes('arrhythmia') || k.includes('bid') || k.includes('icd') ? COLOR2
                          : k.includes('myotonia') ? COLOR3
                          : k.includes('mexiletine') ? COLOR3
                          : k.includes('flecainide') ? COLOR5
                          : k.includes('beta') ? COLOR
                          : COLOR6
                        }
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Channel molecular biology */}
            <div className="card mb-3">
              <div className="card-header small fw-bold" style={{ backgroundColor: LIGHT }}>Channel Molecular Biology</div>
              <div className="card-body small">{gene.channel_class}</div>
            </div>

            {/* Phenotype */}
            <div className="card mb-3">
              <div className="card-header small fw-bold" style={{ backgroundColor: LIGHT }}>Clinical Phenotype</div>
              <div className="card-body small">{gene.phenotype}</div>
            </div>

            {/* Treatment */}
            <div className="card mb-3">
              <div className="card-header small fw-bold" style={{ backgroundColor: LIGHT }}>Treatment Options</div>
              <div className="card-body">
                {(gene.treatment_options || []).map((t, i) => (
                  <div key={i} className="mb-1 small border-start ps-2"
                    style={{ borderColor: t.includes('ABSOLUTELY') || t.includes('CONTRAINDICATED') || t.includes('AVOID') ? '#b71c1c' : '#1565c0' }}>
                    {t}
                  </div>
                ))}
              </div>
            </div>

            {/* DDx */}
            <div className="card mb-3">
              <div className="card-header small fw-bold" style={{ backgroundColor: LIGHT }}>Key Differential Diagnoses</div>
              <div className="card-body">
                {(gene.key_ddx || []).map((d, i) => (
                  <div key={i} className="mb-1 small">• {d}</div>
                ))}
              </div>
            </div>

            {/* Sample patients */}
            {(gene.sample_patients || []).length > 0 && (
              <div className="card">
                <div className="card-header small fw-bold" style={{ backgroundColor: LIGHT }}>Sample Patients (n=3 of 40)</div>
                <div className="card-body">
                  <div className="table-responsive">
                    <table className="table table-sm table-bordered small mb-0">
                      <thead>
                        <tr>
                          <th>ID</th><th>Sex</th><th>Onset(y)</th><th>Sev</th>
                          <th>Arrh</th><th>Myot</th><th>PP</th><th>BidVT</th>
                          <th>ICD</th><th>BB</th><th>Mex</th><th>Flec</th><th>Treatment</th>
                        </tr>
                      </thead>
                      <tbody>
                        {gene.sample_patients.map((p) => (
                          <tr key={p.id}>
                            <td><code className="small">{p.id}</code></td>
                            <td>{p.sex}</td>
                            <td>{p.onset_age_y}</td>
                            <td>
                              <span className={`badge bg-${p.severity === 'Severe' ? 'danger' : p.severity === 'Moderate' ? 'warning text-dark' : 'success'}`}>
                                {p.severity}
                              </span>
                            </td>
                            <td>{p.arrhythmia ? '✓' : '—'}</td>
                            <td>{p.myotonia ? '✓' : '—'}</td>
                            <td>{p.periodic_paralysis ? '✓' : '—'}</td>
                            <td>{p.bidirectional_vt ? '✓' : '—'}</td>
                            <td>{p.icd_implanted ? '✓' : '—'}</td>
                            <td>{p.beta_blocker ? '✓' : '—'}</td>
                            <td>{p.mexiletine ? '✓' : '—'}</td>
                            <td>{p.flecainide ? '✓' : '—'}</td>
                            <td className="small">{p.current_treatment}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const terms = data.terms || [];
  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Channelopathy-Atlas Clinical Definitions</h6>
      {terms.map((d, i) => (
        <div key={i} className="card mb-3">
          <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT, color: COLOR }}>
            {d.term}
          </div>
          <div className="card-body small">{d.definition}</div>
          {d.clinical_rule && (
            <div className="card-footer small fw-semibold text-danger">
              ⚡ Rule: {d.clinical_rule}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────
export default function ChannelopathyAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    fetch(`${API}/api/channelopathy-atlas/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setErrors(prev => ({ ...prev, overview: e.message })));
    fetch(`${API}/api/channelopathy-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(e => setErrors(prev => ({ ...prev, breakdown: e.message })));
    fetch(`${API}/api/channelopathy-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions)
      .catch(e => setErrors(prev => ({ ...prev, definitions: e.message })));
  }, []);

  const tabContent = [
    <OverviewTab key="ov" data={overview} />,
    <GeneTableTab key="gt" data={breakdown} />,
    <ClinicalAtlasTab key="ca" data={breakdown} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      {/* Critical alert banner */}
      <div className="alert alert-danger py-2 px-3 mb-3 small">
        <strong>⚡ Critical Channelopathy Rules:</strong>{' '}
        <strong>CLCN1:</strong> QUININE/QUINIDINE ABSOLUTELY CONTRAINDICATED — paradoxical ClC-1 block worsens myotonia ·
        <strong>SCN5A Brugada:</strong> FEVER unmasks Type 1 pattern — paracetamol mandatory; no ibuprofen ·
        <strong>KCNQ1 LQT1:</strong> SWIMMING HIGH RISK — competitive swimming prohibited ·
        <strong>KCNH2 LQT2:</strong> SUDDEN AROUSAL trigger (alarm clock) — massive drug DDI list (crediblemeds.org) ·
        <strong>RYR2 CPVT:</strong> Flecainide + Beta-blockers BOTH mandatory; ICD = adjunct NOT replacement (shock → VT storm)
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              onClick={() => setTab(i)}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Error */}
      {errors[['overview', 'breakdown', 'breakdown', 'definitions'][tab]] && (
        <ErrorMsg msg={errors[['overview', 'breakdown', 'breakdown', 'definitions'][tab]]} />
      )}

      {/* Content */}
      {tabContent[tab]}
    </div>
  );
}
