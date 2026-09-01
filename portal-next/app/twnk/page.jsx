'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotypes', 'Liver & OLT', 'Treatments', 'Definitions'];
const COLOR = '#00695c';   // teal — TWNK/MDDS7 (hepatocerebral helicase; liver+brain depletion)
const LIGHT = '#e0f2f1';

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
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : '#e0f2f1';
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
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
      {/* Critical VPA Warning Banner */}
      <div className="mb-3 p-3 rounded fw-bold text-center" style={{ background: '#b71c1c', color: 'white', fontSize: '1.05rem' }}>
        ⛔ VPA = ABSOLUTE CONTRAINDICATION IN TWNK MDDS7 — mtDNA DEPLETION HEPATOTOXICITY (POLG INHIBITION + CoA SEQUESTRATION + EPOXIDE)
      </div>
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#e65100', color: 'white', fontSize: '0.95rem' }}>
        🚫 KETOGENIC DIET = CONTRAINDICATED — Forces OXPHOS-Dependent Fat Oxidation That Fails in mtDNA Depletion
      </div>

      {/* OLT Warning */}
      <div className="mb-4 p-2 rounded fw-semibold text-center" style={{ background: '#1565c0', color: 'white', fontSize: '0.9rem' }}>
        🏥 OLT RULE: Hepatic-Only (25%) → May Cure · Hepatocerebral (75%) → OLT Does NOT Prevent Brain Depletion
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Hepatocerebral" value={`${kpis.hepatocerebral_pct}%`} color={COLOR} />
        <KPI label="Hepatic-Only" value={`${kpis.hepatic_only_pct}%`} color="#2e7d32" />
        <KPI label="Lactic Acidosis" value={`${kpis.lactic_acidosis_pct}%`} color="#c62828" />
        <KPI label="Hypoglycemia" value={`${kpis.hypoglycemia_pct}%`} color="#e65100" />
        <KPI label="Nystagmus" value={`${kpis.nystagmus_pct}%`} color="#2e7d32" />
        <KPI label="3-MGA-uria" value={`${kpis.three_mga_pct}%`} color="#2e7d32" />
      </div>

      {/* Identity */}
      <SectionCard title="🧬 Disease Identity">
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Disease:</strong> {data.disease?.split('/')[0].trim()}</div>
          <div className="col-md-6"><strong>Gene:</strong> {data.gene?.split(';')[0]}</div>
          <div className="col-md-4"><strong>Chromosome:</strong> {data.chromosome}</div>
          <div className="col-md-4"><strong>OMIM Gene:</strong> {data.omim_gene} &nbsp; <strong>MDDS7:</strong> {data.omim_disease_mdds7}</div>
          <div className="col-md-4"><strong>adPEO-2:</strong> {data.omim_disease_adpeo}</div>
          <div className="col-md-6"><strong>Inheritance:</strong> {data.inheritance?.split(';')[0]}</div>
          <div className="col-md-6"><strong>Prevalence:</strong> {data.prevalence}</div>
          <div className="col-12"><strong>First described:</strong> {data.first_described}</div>
          <div className="col-12"><strong>Category:</strong> {data.category}</div>
          <div className="col-12"><strong>Protein:</strong> <span className="text-muted">{data.gene}</span></div>
        </div>
      </SectionCard>

      {/* Three Phenotypes */}
      <SectionCard title="⚠️ Three TWNK Phenotypes — Critical Distinction" borderColor="#00695c">
        <div className="row g-3 small">
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ background: '#ffebee', border: '2px solid #c62828' }}>
              <div className="fw-bold mb-1" style={{ color: '#c62828' }}>🔴 MDDS7 Hepatocerebral (75%)</div>
              <div>Biallelic null/severe; infantile liver failure + brain depletion; lactic acidosis 100%; hypoglycemia 70%</div>
              <div className="mt-1 fw-semibold">OLT does NOT prevent brain mtDNA depletion — neurological decline continues</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ background: '#e8f5e9', border: '2px solid #2e7d32' }}>
              <div className="fw-bold mb-1" style={{ color: '#2e7d32' }}>🟢 MDDS7 Hepatic-Only (25%)</div>
              <div>CNS spared; liver failure treatable; OLT may be curative; neurological monitoring annual</div>
              <div className="mt-1 fw-semibold">Best OLT outcomes when CNS involvement is absent pre-transplant</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="p-3 rounded" style={{ background: '#e8eaf6', border: '2px solid #3949ab' }}>
              <div className="fw-bold mb-1" style={{ color: '#3949ab' }}>🟣 IOSCA / adPEO-2 (allelic)</div>
              <div>IOSCA (p.Y508C, Finnish): ataxia + SNHL; no liver failure; survives to adulthood</div>
              <div className="mt-1">adPEO-2 (heterozygous): multiple deletions, not depletion; adult-onset PEO</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* DDx — No Nystagmus */}
      <div className="mb-4 p-3 rounded" style={{ background: '#fff3e0', border: '2px solid #e65100' }}>
        <div className="fw-bold mb-1" style={{ color: '#e65100' }}>👁️ NO Nystagmus — Critical DDx from DGUOK</div>
        <div className="small">DGUOK MDDS3: nystagmus 90% (rotary/pendular — pathognomonic, first sign in neonatal period). TWNK MDDS7 lacks nystagmus. Use nystagmus to separate DGUOK (present) from TWNK/MPV17/POLG (absent) before genetics available.</div>
      </div>

      {/* Clinical Highlights */}
      <SectionCard title="🏥 Clinical Highlights">
        {highlights.map((h, i) => (
          <Alert key={i}
            variant={i < 2 ? 'danger' : i === 3 ? 'info' : i === 4 ? 'warning' : i < 7 ? 'warning' : 'info'}
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
      <SectionCard title="🔀 Differential Diagnosis — TWNK MDDS7 vs Other Hepatocerebral MDDS">
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
      <SectionCard title="👥 Cohort Phenotype Distribution (n=40, seed-555)">
        <div className="row g-3 mb-3">
          {phenotypes.map((g, i) => (
            <div key={i} className="col-md-6 mb-3">
              <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${[COLOR, '#2e7d32', '#3949ab', '#c62828'][i] || COLOR}` }}>
                <div className="card-body text-center">
                  <div className="fw-bold fs-3" style={{ color: [COLOR, '#2e7d32', '#3949ab', '#c62828'][i] || COLOR }}>{g.n}</div>
                  <div className="fw-semibold small">{g.name?.split('—')[0].trim()}</div>
                  <div className="text-muted small">{g.pct}% of cohort</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Genotype Breakdown">
        {genotypes.map((v, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #b2dfdb' }}>
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-bold small" style={{ color: COLOR }}>{v.variant}</span>
              <span className="badge" style={{ background: COLOR }}>{v.n} patients</span>
            </div>
            <div className="small text-muted">{v.phenotype}</div>
            <div className="small mt-1"><strong>Residual activity:</strong> {v.residual_activity}</div>
            {v.mechanism && <div className="small mt-1 text-muted"><strong>Mechanism:</strong> {v.mechanism}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Clinical Feature Prevalence">
        {features.map((f, i) => (
          <div key={i} className="mb-2">
            <Bar
              label={f.feature}
              value={f.pct}
              color={f.pct === 0 ? '#2e7d32' : f.pct === 100 ? COLOR : undefined}
            />
            <div className="text-muted small mb-2" style={{ marginLeft: 4 }}>{f.note}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function LiverTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading data...</div>;
  const liver = data.liver_pathology || {};
  const olt = data.olt_outcomes || {};
  const timeline = data.disease_timeline || [];

  return (
    <div>
      <SectionCard title="🔬 Liver Pathology & mtDNA Depletion">
        <div className="row g-3 mb-3">
          {[
            { label: 'Liver mtDNA Depletion', val: `${liver.mtdna_depletion_liver_pct}% of normal`, color: '#c62828' },
            { label: 'Brain mtDNA Depletion', val: `${liver.mtdna_depletion_brain_pct}% of normal`, color: COLOR },
          ].map((b, i) => (
            <div key={i} className="col-md-6">
              <div className="card shadow-sm text-center p-3">
                <div className="fw-bold fs-5" style={{ color: b.color }}>{b.val}</div>
                <div className="small text-muted">{b.label}</div>
              </div>
            </div>
          ))}
        </div>
        <Alert variant="info" text={`Diagnostic threshold: ${liver.mtdna_threshold_diagnostic}`} />
        <div className="small text-muted mt-2"><strong>OXPHOS deficiency:</strong> {liver.oxphos_deficiency}</div>
        <div className="small text-muted mt-1"><strong>Histology:</strong> {liver.histology}</div>
        <div className="small text-muted mt-1"><strong>Electron microscopy:</strong> {liver.electron_microscopy}</div>
      </SectionCard>

      <SectionCard title="🏥 Liver Transplant (OLT) Outcomes" borderColor="#1565c0">
        <div className="mb-3 p-3 rounded fw-bold" style={{ background: '#e3f2fd', border: '2px solid #1565c0', fontSize: '0.9rem' }}>
          ⚠️ OLT RULE: Hepatic-Only → May Cure (CNS spared) · Hepatocerebral → Does NOT prevent brain mtDNA depletion
        </div>
        <div className="row g-3 mb-3">
          {[
            { label: 'Hepatic-Only Patients (OLT)', val: olt.hepatic_only_olt_n, color: '#2e7d32' },
            { label: 'OLT Curative (Hepatic-Only)', val: `${olt.hepatic_only_olt_curative_pct}%`, color: '#2e7d32' },
            { label: 'Hepatocerebral Patients (OLT)', val: olt.hepatocerebral_olt_n, color: '#c62828' },
            { label: 'Neurological Progression Post-OLT', val: `${olt.hepatocerebral_olt_neurological_progression_pct}%`, color: '#c62828' },
          ].map((r, i) => (
            <div key={i} className="col-md-3 col-6">
              <div className="card shadow-sm text-center p-3">
                <div className="fw-bold fs-5" style={{ color: r.color }}>{r.val}</div>
                <div className="small text-muted">{r.label}</div>
              </div>
            </div>
          ))}
        </div>
        <Alert variant="success" text={`Hepatic-only median survival post-OLT: ${olt.hepatic_only_median_survival_post_olt_yr} years`} />
        <Alert variant="danger" text={`Hepatocerebral median survival post-OLT: ${olt.hepatocerebral_median_survival_post_olt_yr} years (neurological decline continues)`} />
        <div className="small text-muted mt-2">{olt.note}</div>
      </SectionCard>

      <SectionCard title="📅 Disease Natural History Timeline">
        {timeline.map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: i % 2 === 0 ? LIGHT : '#fff', border: '1px solid #b2dfdb' }}>
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
      <Alert variant="danger" text="⛔ VPA = ABSOLUTE CONTRAINDICATION in TWNK MDDS7 regardless of indication. Three mechanisms: POLG inhibition + CoA sequestration + epoxide hepatotoxicity." />
      <Alert variant="warning" text="🚫 Ketogenic Diet = CONTRAINDICATED — forces OXPHOS-dependent fat oxidation that fails in mtDNA depletion." />
      <Alert variant="warning" text="🚫 Propofol AVOID — PRIS risk in mitochondrial disease. Alternative: ketamine + sevoflurane." />
      <Alert variant="info" text="💉 IV Dextrose GIR 8-10 = mandatory during fasting, procedures, illness. Hypoglycemia 70% in MDDS7." />
      <Alert variant="success" text="🏥 OLT decision: MUST assess neurological status pre-OLT. Hepatic-only → curative intent. Hepatocerebral → OLT does not prevent brain depletion." />

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

export default function TWNKPage() {
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
    fetchData('/api/twnk/overview', setOverview, 'overview');
  }, []);

  useEffect(() => {
    if (activeTab === 1 || activeTab === 2) {
      if (!breakdown) fetchData('/api/twnk/breakdown', setBreakdown, 'breakdown');
    }
    if (activeTab === 3) {
      if (!breakdown) fetchData('/api/twnk/breakdown', setBreakdown, 'breakdown');
    }
    if (activeTab === 4) {
      if (!definitions) fetchData('/api/twnk/definitions', setDefinitions, 'definitions');
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 TWNK Hepatocerebral mtDNA Depletion Syndrome (MDDS7) / IOSCA
        </h4>
        <div className="text-muted small">
          Mitochondrial DNA Depletion Syndrome 7 (MDDS7) · IOSCA ·
          TWNK Mitochondrial DNA Helicase · 684 aa · 10q24.31 ·
          OMIM Gene 606075 · Disease MDDS7 271245 · adPEO-2 609286 · AR (MDDS7/IOSCA) / AD (adPEO)
        </div>
        <div className="mt-1 small fw-semibold" style={{ color: '#c62828' }}>
          ⛔ VPA ABSOLUTE CI · 🚫 KD CONTRAINDICATED · 👁️ NO Nystagmus (DDx DGUOK) ·
          🚫 NO 3-MGA · 🏥 OLT: Hepatic-Only May Cure, Hepatocerebral Does NOT Prevent Brain Depletion
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
      {activeTab === 2 && <LiverTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
