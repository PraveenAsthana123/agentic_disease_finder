'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// MND-Atlas color palette — motor neuron / ALS / neurodegeneration
const COLOR  = '#b71c1c';  // deep red — ALS / neurodegeneration / fatal
const LIGHT  = '#ffebee';  // light red tint
const COLOR2 = '#880e4f';  // dark magenta — aggressive / fatal
const COLOR3 = '#1b5e20';  // deep green — slow progression / SETX
const COLOR4 = '#e65100';  // orange — caution / respiratory
const COLOR5 = '#0d47a1';  // deep blue — gene therapy (SOD1/tofersen)
const COLOR6 = '#37474f';  // blue-grey — moderate / adult onset
const COLOR7 = '#4a148c';  // purple — X-linked dominant (UBQLN2)

const GENE_COLORS = {
  SOD1:    '#0d47a1',  // gene therapy available — blue
  C9orf72: '#b71c1c',  // most common fALS, FTD — red
  TARDBP:  '#6a1b9a',  // TDP-43 universal marker — purple
  FUS:     '#880e4f',  // juvenile aggressive — dark magenta
  VCP:     '#e65100',  // multisystem tetrad — orange
  SETX:    '#1b5e20',  // juvenile slow, best prognosis — green
  NEK1:    '#37474f',  // DNA damage repair — blue-grey
  UBQLN2:  '#4a148c',  // X-linked dominant — deep purple
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
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /><div className="mt-2 text-muted small">Loading MND-Atlas…</div></div>;
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger">{msg}</div>;
}

// ── Tab: Overview ─────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const cf = ov.clinical_features_prevalence || {};
  const sev = ov.severity || {};
  const cat = ov.mnd_category_breakdown || {};

  return (
    <div>
      <h5 className="fw-bold mb-1" style={{ color: COLOR }}>{ov.full_name}</h5>
      <p className="text-muted small mb-3">{ov.subtitle}</p>
      <p className="mb-3">{ov.description}</p>

      {/* Drug alerts */}
      {(ov.drug_alerts || []).map((a, i) => (
        <AlertBox key={i}
          type={a.includes('ABSOLUTELY') || a.includes('STEROIDS') || a.includes('FIRST') ? 'danger' : 'warning'}
          title="Drug / Treatment Alert">
          {a}
        </AlertBox>
      ))}

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={ov.total_patients} color={COLOR} />
        <KPI label="Genes Covered" value={ov.genes_covered} color={COLOR3} />
        <KPI label="Patients/Gene" value={ov.patients_per_gene} color={COLOR5} />
        <KPI label="Mean Onset (y)" value={ov.mean_onset_age_y} color={COLOR4} />
        <KPI label="Mean CK (IU/L)" value={(ov.mean_ck_iu_l || 0).toLocaleString()} color={COLOR6} />
        <KPI label="Seeds" value={ov.seed_range} color={COLOR6} />
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
              <BarRow label="Respiratory Decline" pct={cf.respiratory_decline_pct} color={COLOR4} />
              <BarRow label="FTD Features" pct={cf.ftd_features_pct} color={COLOR2} />
              <BarRow label="Bulbar Onset" pct={cf.bulbar_onset_pct} color={COLOR} />
              <BarRow label="Pseudobulbar Affect (PBA)" pct={cf.pseudobulbar_affect_pct} color={COLOR6} />
              <BarRow label="Juvenile Onset (<30y)" pct={cf.juvenile_onset_pct} color={COLOR7} />
              <BarRow label="NIV Required" pct={cf.niv_required_pct} color={COLOR4} />
              <BarRow label="PEG Required" pct={cf.peg_required_pct} color={COLOR6} />
              <BarRow label="Gene Therapy Offered" pct={cf.gene_therapy_offered_pct} color={COLOR5} />
              <BarRow label="Cognitive Impairment" pct={cf.cognitive_impairment_pct} color={COLOR2} />
            </div>
          </div>
        </div>
      </div>

      {/* MND Gene Category Breakdown */}
      <div className="card mb-4">
        <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT }}>Gene-Pathway Categories</div>
        <div className="card-body">
          <div className="row g-2">
            {Object.entries(cat).map(([category, genes]) => (
              <div key={category} className="col-md-6">
                <div className="border rounded p-2 h-100" style={{ borderLeft: `4px solid ${GENE_COLORS[genes[0]] || COLOR}` }}>
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
            <div key={i} className="mb-2 small border-start border-danger ps-2">{pt}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || [];
  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>MND-Atlas — 8-Gene Clinical Reference Table</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered table-hover small">
          <thead style={{ backgroundColor: LIGHT }}>
            <tr>
              <th>Gene</th><th>Protein / Size</th><th>Locus</th><th>Inheritance</th>
              <th>ALS Subtype</th><th>Pathway</th><th>Onset (y)</th>
              <th>Juvenile?</th><th>FTD Risk</th><th>Gene Rx</th>
              <th>Slow?</th><th>XLD?</th><th>Mean CK</th>
            </tr>
          </thead>
          <tbody>
            {genes.map((g) => (
              <tr key={g.gene}>
                <td><span className="fw-bold" style={{ color: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span></td>
                <td>{g.protein}<br /><span className="text-muted">{g.aa} · {g.kDa}</span></td>
                <td><code className="small">{g.locus}</code></td>
                <td className="small">{g.inheritance?.split('.')[0]}</td>
                <td className="small">{g.mnd_type?.split('—')[0]?.trim()}</td>
                <td className="small text-muted">{g.mnd_group?.split('/')[0]?.trim()}</td>
                <td className="text-center">{g.onset_range_y?.[0]}–{g.onset_range_y?.[1]}</td>
                <td className="text-center">{g.juvenile_onset ? '🔴 Yes' : '—'}</td>
                <td className="text-center">{g.ftd_risk ? '⚠️ Yes' : '—'}</td>
                <td className="text-center">{g.gene_therapy_available ? '✅ Yes' : '—'}</td>
                <td className="text-center">{g.very_slow_progression ? '🟢 Yes' : '—'}</td>
                <td className="text-center">{g.xlinked ? '🔵 XLD' : '—'}</td>
                <td className="text-center">{(g.mean_ck_iu_l || 0).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small text-muted">XLD = X-Linked Dominant; Gene Rx = approved gene-specific therapy; Slow = very slow progression (SETX, VCP-IBM); FTD = frontotemporal dementia risk.</p>
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
              style={selected === g.gene ? { backgroundColor: GENE_COLORS[g.gene] || COLOR, borderColor: GENE_COLORS[g.gene] || COLOR } : {}}
              onClick={() => setSelected(g.gene)}>
              <span className="fw-bold">{g.gene}</span>
              <div className="small">{g.mnd_type?.split('—')[0]?.trim()}</div>
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

            <div className="row g-2 mb-3">
              {[
                { label: 'Locus', value: gene.locus },
                { label: 'Size', value: `${gene.aa} · ${gene.kDa}` },
                { label: 'Onset', value: `${gene.onset_range_y?.[0]}–${gene.onset_range_y?.[1]}y` },
                { label: 'OMIM Gene', value: gene.omim_gene },
                { label: 'OMIM Disease', value: gene.omim_disease },
                { label: 'Mean CK', value: `${(gene.mean_ck_iu_l || 0).toLocaleString()} IU/L` },
                { label: 'Mean Survival', value: `${gene.mean_projected_survival_y}y` },
              ].map(({ label, value }) => (
                <div key={label} className="col-6 col-md-4">
                  <div className="card text-center py-1">
                    <div className="fw-bold small" style={{ color: GENE_COLORS[gene.gene] || COLOR }}>{value}</div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{label}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="row g-2 mb-2">
              {[
                { label: 'FTD Risk', val: gene.ftd_risk },
                { label: 'Respiratory', val: gene.respiratory_risk },
                { label: 'Bulbar Onset', val: gene.bulbar_onset },
                { label: 'Juvenile', val: gene.juvenile_onset },
                { label: 'X-Linked Dom', val: gene.xlinked },
                { label: 'Slow Prog', val: gene.very_slow_progression },
                { label: 'Gene Therapy', val: gene.gene_therapy_available },
              ].map(({ label, val }) => (
                <div key={label} className="col-auto">
                  <span className={`badge ${val ? 'bg-danger' : 'bg-secondary'}`}>{label}: {val ? 'Yes' : 'No'}</span>
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
                      <BarRow key={k} label={k.replace(/_pct$/, '').replace(/_/g, ' ')} pct={v}
                        color={k.includes('gene_therapy') ? COLOR5 : k.includes('ftd') || k.includes('peg') ? COLOR2 : COLOR} />
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Gene class */}
            <div className="card mb-3">
              <div className="card-header small fw-bold" style={{ backgroundColor: LIGHT }}>Molecular Biology</div>
              <div className="card-body small">{gene.gene_class}</div>
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
                  <div key={i} className="mb-1 small border-start border-primary ps-2">{t}</div>
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
                      <thead><tr><th>ID</th><th>Sex</th><th>Onset(y)</th><th>Sev</th><th>Resp</th><th>FTD</th><th>Bulbar</th><th>NIV</th><th>PEG</th><th>Treatment</th></tr></thead>
                      <tbody>
                        {gene.sample_patients.map((p) => (
                          <tr key={p.id}>
                            <td><code className="small">{p.id}</code></td>
                            <td>{p.sex}</td>
                            <td>{p.onset_age_y}</td>
                            <td><span className={`badge bg-${p.severity === 'Severe' ? 'danger' : p.severity === 'Moderate' ? 'warning text-dark' : 'success'}`}>{p.severity}</span></td>
                            <td>{p.respiratory_decline ? '✓' : '—'}</td>
                            <td>{p.ftd_features ? '✓' : '—'}</td>
                            <td>{p.bulbar_onset ? '✓' : '—'}</td>
                            <td>{p.niv ? '✓' : '—'}</td>
                            <td>{p.peg ? '✓' : '—'}</td>
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

// ── Tab: Definitions ──────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || [];
  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>MND-Atlas Clinical Definitions</h6>
      {defs.map((d, i) => (
        <div key={i} className="card mb-3">
          <div className="card-header fw-bold small" style={{ backgroundColor: LIGHT, color: COLOR }}>
            {d.term}
          </div>
          <div className="card-body small">{d.definition}</div>
        </div>
      ))}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────
export default function MNDAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [errors, setErrors] = useState({});

  useEffect(() => {
    fetch(`${API}/api/mnd-atlas/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setErrors(prev => ({ ...prev, overview: e.message })));
    fetch(`${API}/api/mnd-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(e => setErrors(prev => ({ ...prev, breakdown: e.message })));
    fetch(`${API}/api/mnd-atlas/definitions`)
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
        <strong>🚨 Critical Treatment Rules:</strong>{' '}
        <strong>SOD1:</strong> TOFERSEN (QALSODY) FIRST FDA-approved gene-specific ALS therapy (IT ASO, April 2023) — confirm SOD1 variant before prescribing ·
        <strong>C9orf72:</strong> REPEAT-PRIMED PCR MANDATORY — standard NGS misses GGGGCC expansion; FTD in ~50% ·
        <strong>VCP-IBM:</strong> STEROIDS ABSOLUTELY INEFFECTIVE/HARMFUL — confirm VCP molecular diagnosis before any immunosuppression ·
        <strong>ALL ALS:</strong> Riluzole first-line universal
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Content */}
      {errors[['overview', 'breakdown', 'breakdown', 'definitions'][tab]] && (
        <ErrorMsg msg={errors[['overview', 'breakdown', 'breakdown', 'definitions'][tab]]} />
      )}
      {tabContent[tab]}
    </div>
  );
}
