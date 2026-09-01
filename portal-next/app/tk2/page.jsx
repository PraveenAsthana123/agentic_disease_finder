'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'Biopsy & Respiratory', 'Treatments', 'Definitions'];
const COLOR = '#4527a0';   // deep purple — TK2/MDDS4A (myopathic; muscle-selective; NIV/dNs rescue)
const LIGHT = '#ede7f6';

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

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : '#ede7f6';
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : '#4527a0';
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const kpis = data.kpis || {};
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];
  const ddx = data.ddx_table || [];

  return (
    <div>
      {/* Critical VPA+KD Warning Banner */}
      <div className="mb-3 p-3 rounded fw-bold text-center" style={{ background: '#b71c1c', color: 'white', fontSize: '1.05rem' }}>
        ⛔ VPA = ABSOLUTE CONTRAINDICATION IN TK2 MDDS4A — mtDNA DEPLETION HEPATOTOXICITY SHARED WITH ALL MDDS
      </div>
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#e65100', color: 'white', fontSize: '0.95rem' }}>
        🚫 KETOGENIC DIET = CONTRAINDICATED — Forces OXPHOS-Dependent Fat Oxidation That Fails in mtDNA Depletion
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Proximal Weakness" value={`${kpis.proximal_weakness_pct}%`} color={COLOR} />
        <KPI label="Respiratory Failure" value={`${kpis.respiratory_failure_pct}%`} color="#c62828" />
        <KPI label="CK Elevated" value={`${kpis.ck_elevation_pct}%`} color="#e65100" />
        <KPI label="Facial Diplegia" value={`${kpis.facial_diplegia_pct}%`} color="#6a1b9a" />
        <KPI label="Ophthalmoplegia" value={`${kpis.ophthalmoplegia_pct}%`} color="#1565c0" />
        <KPI label="Hepatopathy" value={`${kpis.hepatopathy_pct}%`} color="#2e7d32" />
      </div>

      {/* Identity */}
      <SectionCard title="🧬 Disease Identity">
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Disease:</strong> {data.disease}</div>
          <div className="col-md-6"><strong>Gene:</strong> {data.gene?.split(';')[0]}</div>
          <div className="col-md-4"><strong>Chromosome:</strong> {data.chromosome}</div>
          <div className="col-md-4"><strong>OMIM Gene:</strong> {data.omim_gene} &nbsp; <strong>Disease:</strong> {data.omim_disease}</div>
          <div className="col-md-4"><strong>Inheritance:</strong> {data.inheritance?.split(';')[0]}</div>
          <div className="col-md-6"><strong>Prevalence:</strong> {data.prevalence}</div>
          <div className="col-md-6"><strong>First described:</strong> {data.first_described}</div>
          <div className="col-12"><strong>Category:</strong> {data.category}</div>
          <div className="col-12"><strong>Protein:</strong> <span className="text-muted">{data.gene}</span></div>
        </div>
      </SectionCard>

      {/* Three Clinical Forms */}
      <SectionCard title="⚠️ Three TK2 Phenotypes — Critical Management Distinction" borderColor="#6a1b9a">
        <div className="row g-3 small">
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ background: '#ffebee', border: '2px solid #c62828' }}>
              <div className="fw-bold mb-1" style={{ color: '#c62828' }}>🔴 MDDS4A — Classic Infantile Myopathic (55%)</div>
              <div>Infantile onset; progressive proximal weakness; respiratory failure; normal intellect; NO hepatopathy</div>
              <div className="mt-1 fw-semibold">Major morbidity: respiratory failure → NIV → tracheostomy</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ background: '#fff8e1', border: '2px solid #f57f17' }}>
              <div className="fw-bold mb-1" style={{ color: '#f57f17' }}>🟡 MDDS4B — Encephalomyopathic (20%)</div>
              <div>Muscle + CNS involvement; cognitive regression; white matter changes; epilepsy; more severe</div>
              <div className="mt-1 fw-semibold">Null allele genotypes; earlier death</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ background: '#e8f5e9', border: '2px solid #2e7d32' }}>
              <div className="fw-bold mb-1" style={{ color: '#2e7d32' }}>🟢 MDDS4C — Late-onset PEO (17%)</div>
              <div>Adult onset (20-55 yr); PEO + ptosis + limb-girdle; slow progression; p.Arg214Cys genotype</div>
              <div className="mt-1 fw-semibold">Normal lifespan possible; NIV if respiratory compromise</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Deoxynucleoside Banner */}
      <div className="mb-4 p-3 rounded" style={{ background: '#e8eaf6', border: '2px solid #3949ab' }}>
        <div className="fw-bold mb-1" style={{ color: '#3949ab' }}>💊 Deoxynucleoside Supplementation — First Disease-Modifying MDDS Rescue</div>
        <div className="small">Oral dThd (deoxythymidine) + dCyd (deoxycytidine) 100-200 mg/kg/day each · FDA Orphan Drug designation</div>
        <div className="small mt-1">Substrate bypass: saturates ENT3 transporter → partial dTTP/dCTP restoration without TK2 · Motor improvement in ~70% treated patients</div>
        <div className="small mt-1 fw-semibold text-success">Initiate immediately on confirmed TK2 diagnosis — earliest start = best outcome</div>
      </div>

      {/* Clinical Highlights */}
      <SectionCard title="🏥 Clinical Highlights">
        {highlights.map((h, i) => (
          <Alert key={i}
            variant={i < 2 ? 'danger' : i === 5 ? 'success' : i < 8 ? 'warning' : 'info'}
            text={h}
          />
        ))}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="⛔ Contraindications" borderColor="#c62828">
        {cis.map((ci, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i < 2 ? '#ffebee' : '#fff8e1', border: `1px solid ${i < 2 ? '#c62828' : '#f57f17'}` }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small">{ci.drug}</span>
              <span className="badge" style={{ background: i < 2 ? '#c62828' : '#e65100', fontSize: '0.65rem' }}>{ci.severity}</span>
            </div>
            <div className="text-muted small">{ci.reason}</div>
          </div>
        ))}
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="📏 Clinical Thresholds & Monitoring">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr style={{ background: LIGHT }}>
              <th>Parameter</th><th>Threshold</th><th>Action</th>
            </tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td><strong>{t.parameter}</strong></td>
                  <td><code>{t.threshold}</code></td>
                  <td className="text-muted">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* DDx Table */}
      <SectionCard title="🔀 Differential Diagnosis — TK2 vs Other Myopathies & MDDS">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr style={{ background: LIGHT }}>
              <th>Disease</th><th>Hepatopathy</th><th>Nystagmus</th><th>3-MGA</th><th>Lactic Acidosis</th><th>CK</th><th>Primary Organ</th><th>VPA CI</th>
            </tr></thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i} style={{ background: i === 0 ? LIGHT : undefined, fontWeight: i === 0 ? 'bold' : undefined }}>
                  <td style={{ color: i === 0 ? COLOR : undefined }}>{d.disease}</td>
                  <td><span style={{ color: d.hepatopathy === 'No' ? '#2e7d32' : '#c62828' }}>{d.hepatopathy}</span></td>
                  <td><span style={{ color: d.nystagmus === 'No' ? '#2e7d32' : '#c62828' }}>{d.nystagmus}</span></td>
                  <td><span style={{ color: d.three_mga === 'No' || d['3mga'] === 'No' ? '#2e7d32' : '#c62828' }}>{d.three_mga || d['3mga']}</span></td>
                  <td>{d.lactic_acidosis}</td>
                  <td>{d.ck}</td>
                  <td>{d.primary_organ}</td>
                  <td><span style={{ color: d.vpa_ci?.startsWith('Absolute') ? '#c62828' : '#2e7d32' }}>{d.vpa_ci}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const phenotypes = data.phenotype_distribution || [];
  const genotypes = data.genotype_breakdown || [];
  const features = data.feature_prevalence || [];

  return (
    <div>
      <SectionCard title="👥 Cohort Phenotype Distribution (n=40, seed-553)">
        <div className="row g-3 mb-3">
          {phenotypes.map((g, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${[COLOR, '#f57f17', '#2e7d32', '#c62828'][i] || COLOR}` }}>
                <div className="card-body text-center">
                  <div className="fw-bold fs-3" style={{ color: [COLOR, '#f57f17', '#2e7d32', '#c62828'][i] || COLOR }}>{g.n}</div>
                  <div className="fw-semibold small">{g.name}</div>
                  <div className="text-muted small">{Math.round(g.n / 40 * 100)}% of cohort</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Genotype Breakdown">
        {genotypes.map((v, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #d1c4e9' }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small" style={{ color: COLOR }}>{v.variant}</span>
              <span className="badge" style={{ background: COLOR }}>{v.n} patients</span>
            </div>
            <div className="small text-muted">{v.phenotype}</div>
            <div className="small mt-1"><strong>Residual activity:</strong> {v.residual_activity}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Clinical Feature Prevalence">
        {features.map((f, i) => (
          <div key={i} className="mb-2">
            <Bar label={f.feature} value={f.pct} color={f.pct === 0 ? '#2e7d32' : f.pct === 100 ? COLOR : undefined} />
            <div className="text-muted small mb-2" style={{ marginLeft: 4 }}>{f.note}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function BiopsyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const biopsy = data.muscle_biopsy || {};
  const resp = data.respiratory_outcomes || {};
  const timeline = data.disease_timeline || [];

  return (
    <div>
      <SectionCard title="🔬 Muscle Biopsy Findings">
        <div className="row g-3 mb-3">
          {[
            { label: 'Ragged-Red Fibers (RRF)', val: biopsy.ragged_red_fibers_pct, color: '#c62828' },
            { label: 'COX-Negative Fibers', val: biopsy.cox_negative_fibers_pct, color: COLOR },
          ].map((b, i) => (
            <div key={i} className="col-md-6">
              <div className="card shadow-sm text-center p-3">
                <div className="fw-bold fs-2" style={{ color: b.color }}>{b.val}%</div>
                <div className="small text-muted">{b.label}</div>
              </div>
            </div>
          ))}
        </div>
        <Alert variant="info" text={`mtDNA depletion threshold: ${biopsy.mtdna_depletion_threshold}`} />
        <div className="small text-muted mt-2"><strong>EM:</strong> {biopsy.electron_microscopy}</div>
        <div className="small text-muted mt-1"><strong>OXPHOS enzymes:</strong> {biopsy.oxidative_phosphorylation_enzymes}</div>
      </SectionCard>

      <SectionCard title="🫁 Respiratory Outcomes" borderColor="#c62828">
        <div className="row g-3 mb-3">
          {[
            { label: 'Median NIV Start', val: `${resp.niv_median_start_yr} yr`, color: '#e65100' },
            { label: 'Tracheostomy Rate', val: `${resp.tracheostomy_pct}%`, color: '#c62828' },
            { label: 'Median Survival w/o Tx', val: `${resp.median_survival_without_tx_yr} yr`, color: '#b71c1c' },
            { label: 'Median Survival + NIV', val: `${resp.median_survival_with_niv_yr} yr`, color: COLOR },
          ].map((r, i) => (
            <div key={i} className="col-md-3 col-6">
              <div className="card shadow-sm text-center p-3">
                <div className="fw-bold fs-5" style={{ color: r.color }}>{r.val}</div>
                <div className="small text-muted">{r.label}</div>
              </div>
            </div>
          ))}
        </div>
        <Alert variant="success" text={`dThd+dCyd + NIV combined: ${resp.median_survival_with_dthd_niv_yr}`} />
        <Alert variant="warning" text="Respiratory failure is the leading cause of death in TK2 MDDS4A. FVC surveillance every 3-6 months from diagnosis. Early NIV saves years." />
      </SectionCard>

      <SectionCard title="📅 Disease Natural History Timeline">
        {timeline.map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #d1c4e9' }}>
            <div className="fw-bold small mb-1" style={{ color: COLOR }}>{t.phase}</div>
            <div className="small text-muted">{t.events}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const treatments = data.treatments || [];

  return (
    <div>
      <Alert variant="danger" text="⛔ VPA = ABSOLUTE CONTRAINDICATION in TK2 MDDS4A regardless of indication. Document in allergy alerts. No safe dose in any mtDNA depletion." />
      <Alert variant="warning" text="🚫 Ketogenic Diet = CONTRAINDICATED — forces OXPHOS-dependent fat oxidation that fails in mtDNA depletion." />
      <Alert variant="warning" text="🚫 Propofol AVOID — PRIS risk in mitochondrial disease. Alternative: ketamine + sevoflurane for anaesthesia." />
      <Alert variant="success" text="💊 Deoxynucleoside supplementation (dCyd + dThd oral) = first disease-modifying rescue. Start immediately on diagnosis. FDA orphan." />
      <Alert variant="info" text="🫁 NIV (nocturnal BiPAP) = most life-extending single intervention. Begin when FVC <60% or nocturnal desaturation detected." />

      {treatments.map((t, i) => {
        const isAbsCI = t.level?.includes('ABSOLUTE') || t.level?.includes('CONTRAINDICATED') || t.level?.includes('AVOID');
        const isTopA = t.level?.startsWith('A');
        const bg = isAbsCI ? '#ffebee' : isTopA ? LIGHT : '#fff';
        const border = isAbsCI ? '#c62828' : isTopA ? COLOR : '#ddd';
        return (
          <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${border}`, background: bg }}>
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-start mb-1">
                <span className="fw-bold small">{t.tx}</span>
                <span className="badge" style={{ background: isAbsCI ? '#c62828' : COLOR, fontSize: '0.65rem' }}>
                  {t.level?.split('—')[0].trim()}
                </span>
              </div>
              <div className="small text-muted">{t.note}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const terms = data.terms || [];
  return (
    <div>
      {terms.map((d, i) => (
        <div key={i} className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${COLOR}` }}>
          <div className="card-body">
            <h6 className="fw-bold mb-2" style={{ color: COLOR }}>{d.term}</h6>
            <p className="small text-muted mb-0">{d.definition}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function TK2Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  const fetchData = async (endpoint, setter, key) => {
    if (loading[key]) return;
    setLoading(l => ({ ...l, [key]: true }));
    try {
      const res = await fetch(`${API}${endpoint}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setter(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(l => ({ ...l, [key]: false }));
    }
  };

  useEffect(() => {
    fetchData('/api/tk2/overview', setOverview, 'overview');
  }, []);

  useEffect(() => {
    if (activeTab === 1 || activeTab === 2) {
      if (!breakdown) fetchData('/api/tk2/breakdown', setBreakdown, 'breakdown');
    }
    if (activeTab === 3) {
      if (!breakdown) fetchData('/api/tk2/breakdown', setBreakdown, 'breakdown');
    }
    if (activeTab === 4) {
      if (!definitions) fetchData('/api/tk2/definitions', setDefinitions, 'definitions');
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          💪 TK2 Myopathic mtDNA Depletion Syndrome (MDDS4A)
        </h4>
        <div className="text-muted small">
          Mitochondrial DNA Depletion Syndrome 4A (MDDS4A) ·
          TK2 Mitochondrial Thymidine Kinase · 265 aa · 16q21 ·
          OMIM Gene 188250 · Disease 609560 · AR
        </div>
        <div className="mt-1 small fw-semibold" style={{ color: '#c62828' }}>
          ⛔ VPA ABSOLUTE CI · 🚫 KD CONTRAINDICATED · 💊 dCyd+dThd First MDDS Rescue ·
          🫁 Respiratory Failure 85% — Early NIV · 🚫 No Hepatopathy (DDx DGUOK/MPV17) · 🚫 No Nystagmus
        </div>
      </div>

      {error && <div className="alert alert-danger small">Error: {error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${activeTab === i ? ' active fw-semibold' : ''}`}
              onClick={() => setActiveTab(i)}
              style={activeTab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <PatientsTab data={breakdown} />}
      {activeTab === 2 && <BiopsyTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
