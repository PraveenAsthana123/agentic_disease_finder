'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ITGA2B:  '#1b5e20',  // deep green — GT type 1, αIIb, absent aggregation
  ITGB3:   '#e65100',  // deep orange — GT type 2, β3, HPA-1a/NAIT
  GP1BA:   '#0d47a1',  // deep blue — BSS, GPIbα, giant platelets
  GP1BB:   '#4a148c',  // deep purple — BSS type B, 22q11.2 deletion
  MYH9:    '#880e4f',  // deep pink — MYH9-RD, Döhle bodies
  ANKRD26: '#37474f',  // dark slate — THC2, 5′UTR, AML risk
  ETV6:    '#b71c1c',  // deep red — THC5, ALL predisposition
  RUNX1:   '#004d40',  // dark teal — FPD-AML, δ-granule, AML/MDS
};

const GENE_DISEASE = {
  ITGA2B:  'GT Type 1 AR — αIIb Integrin — Absent αIIbβ3 — Absent Clot Retraction PATHOGNOMONIC — rFVIIa Inhibitors',
  ITGB3:   'GT Type 2 AR — β3 Integrin — HPA-1a/PlA1 Antigen — NAIT First Pregnancy — IVIG + HPA-Compatible Platelets',
  GP1BA:   'BSS AR — GPIbα VWF Receptor — Giant Platelets PATHOGNOMONIC — Absent Ristocetin — NOT ITP',
  GP1BB:   'BSS Type B AR — GPIbβ — 22q11.2 Deletion DiGeorge Overlap — MLPA Mandatory',
  MYH9:    'MYH9-RD AD — Non-Muscle Myosin IIA — Döhle-like Bodies PATHOGNOMONIC — NOT ITP — Nephritis Surveillance',
  ANKRD26: 'THC2 AD — 5′UTR Variants WES-MISS — Normal Platelet Function — AML/MDS 8% Lifetime Risk',
  ETV6:    'THC5 AD — ETS Transcription Factor — ALL Predisposition 25–35% — Donor Screening MANDATORY',
  RUNX1:   'FPD-AML AD — RUNX1 Transcription Factor — δ-Granule Deficiency — AML/MDS 35–44% — Donor Screening MANDATORY',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#94a3b8' }}>Loading…</div>;
}

function ErrorBox({ msg }) {
  return (
    <div style={{ padding: '1rem', background: '#450a0a', borderRadius: 8, color: '#fca5a5', margin: '1rem 0' }}>
      Error: {msg}
    </div>
  );
}

function KPI({ label, value, color }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem',
      borderLeft: `4px solid ${color || '#6366f1'}`, minWidth: 160,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#a5b4fc' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function AlertCard({ alert, color }) {
  const [code, ...rest] = alert.split(':');
  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: '0.85rem 1rem',
      borderLeft: `4px solid ${color || '#f59e0b'}`, marginBottom: 8,
    }}>
      <span style={{ fontWeight: 700, color: color || '#fbbf24', fontSize: 12 }}>{code}:</span>
      <span style={{ color: '#cbd5e1', fontSize: 12 }}>{rest.join(':')}</span>
    </div>
  );
}

// ── OVERVIEW TAB ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats || {};
  const genes = data.genes || [];
  const alerts = data.top_alerts || [];
  return (
    <div>
      <h2 style={{ color: '#e2e8f0', marginBottom: 4 }}>Hereditary Platelet Disorder Atlas</h2>
      <p style={{ color: '#94a3b8', marginBottom: 20, fontSize: 13 }}>
        Complete 8-Gene Hereditary Platelet Disorder Reference — {s.total_patients} patients (8×40, seeds 1606–1613) ·
        Mean diagnosis age {s.mean_dx_age_years}y · Mean delay {s.mean_dx_delay_months} months
      </p>

      {/* KPI row 1 — platelet function disorders */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        Platelet Function Disorders (GT)
      </h3>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <KPI label="GT Clot Retraction Absent" value={`${s.gt_clot_retraction_absent_pct}%`} color="#1b5e20" />
        <KPI label="GT rFVIIa Used (Inhibitors)" value={`${s.gt_rfviia_used_pct}%`} color="#4ade80" />
      </div>

      {/* KPI row 2 — BSS */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        Bernard-Soulier Syndrome (BSS)
      </h3>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <KPI label="BSS Giant Platelets Film" value={`${s.bss_giant_platelets_pct}%`} color="#0d47a1" />
        <KPI label="BSS Misdiagnosed as ITP" value={`${s.bss_misdiagnosed_itp_pct}%`} color="#ef4444" />
      </div>

      {/* KPI row 3 — MYH9 */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        MYH9-Related Disease
      </h3>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <KPI label="MYH9 Misdiagnosed as ITP" value={`${s.myh9_misdiagnosed_itp_pct}%`} color="#880e4f" />
        <KPI label="MYH9 Nephritis Risk" value={`${s.myh9_nephritis_pct}%`} color="#f59e0b" />
      </div>

      {/* KPI row 4 — Malignancy predisposition */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        Malignancy Predisposition Genes
      </h3>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <KPI label="ANKRD26 WES-Missed" value={`${s.ankrd26_wes_missed_pct}%`} color="#37474f" />
        <KPI label="ANKRD26 AML/MDS Lifetime" value={`${s.ankrd26_aml_lifetime_pct}%`} color="#f97316" />
        <KPI label="ETV6 ALL Predisposition" value={`${s.etv6_all_predisposition_pct}%`} color="#b71c1c" />
        <KPI label="RUNX1 AML/MDS Lifetime" value={`${s.runx1_aml_mds_pct}%`} color="#004d40" />
        <KPI label="RUNX1 δ-Granule Deficient" value={`${s.runx1_delta_granule_pct}%`} color="#14b8a6" />
      </div>

      {/* Gene cards */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        8 Genes — Platelet Disorder Reference
      </h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 12, marginBottom: 24 }}>
        {genes.map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '0.9rem 1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 15 }}>{g.gene}</div>
            <div style={{ color: '#94a3b8', fontSize: 11, margin: '2px 0 6px' }}>
              {g.locus} · {g.inheritance} · OMIM {g.omim_disease}
            </div>
            <div style={{ color: '#cbd5e1', fontSize: 11 }}>{GENE_DISEASE[g.gene]}</div>
            <div style={{ marginTop: 8, display: 'flex', gap: 16 }}>
              <span style={{ color: '#94a3b8', fontSize: 11 }}>n={g.n_patients}</span>
              <span style={{ color: '#94a3b8', fontSize: 11 }}>Dx age {g.mean_dx_age}y</span>
            </div>
          </div>
        ))}
      </div>

      {/* Alerts */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        Key Clinical Alerts (Top 2 per Gene)
      </h3>
      {alerts.slice(0, 16).map((a, i) => {
        const gene = a.split('-')[0];
        return <AlertCard key={i} alert={a} color={GENE_COLORS[gene] || '#f59e0b'} />;
      })}
    </div>
  );
}

// ── GENE TABLE TAB ────────────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
            {['Gene', 'Protein (short)', 'Locus', 'aa / kDa', 'Inheritance', 'OMIM Disease', 'Mean Dx Age', 'n Patients'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #334155' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {genes.map((g, i) => (
            <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
              <td style={{ padding: '7px 10px', color: '#cbd5e1', maxWidth: 260 }}>{g.protein_short || g.alias?.slice(0, 80)}</td>
              <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
              <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.aa} / {g.kDa}</td>
              <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{(g.inheritance || '').split(';')[0]}</td>
              <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.omim_disease}</td>
              <td style={{ padding: '7px 10px', color: '#94a3b8' }}>
                {g.patients ? (g.patients.reduce((a, p) => a + p.age_at_diagnosis, 0) / g.patients.length).toFixed(1) + 'y' : '—'}
              </td>
              <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── CLINICAL ATLAS TAB ────────────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const genes = Object.values(data);
  const gene = selected ? data[selected] : null;

  return (
    <div style={{ display: 'flex', gap: 16 }}>
      {/* sidebar */}
      <div style={{ minWidth: 160, background: '#1e293b', borderRadius: 10, padding: '0.8rem', height: 'fit-content' }}>
        {genes.map(g => (
          <div key={g.gene}
            onClick={() => setSelected(g.gene === selected ? null : g.gene)}
            style={{
              padding: '6px 10px', borderRadius: 6, marginBottom: 4, cursor: 'pointer',
              background: selected === g.gene ? (GENE_COLORS[g.gene] || '#6366f1') : 'transparent',
              color: selected === g.gene ? '#fff' : (GENE_COLORS[g.gene] || '#a5b4fc'),
              fontWeight: 700, fontSize: 13,
            }}>
            {g.gene}
          </div>
        ))}
      </div>

      {/* detail */}
      <div style={{ flex: 1 }}>
        {!gene && (
          <div style={{ color: '#94a3b8', fontSize: 13, padding: '2rem' }}>Select a gene to view clinical details.</div>
        )}
        {gene && (
          <div>
            <h2 style={{ color: GENE_COLORS[gene.gene] || '#a5b4fc', marginBottom: 4 }}>{gene.gene}</h2>
            <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 12 }}>
              {gene.locus} · {gene.aa} · {gene.kDa} · OMIM {gene.omim_disease} · {(gene.inheritance || '').split(';')[0]}
            </div>

            {/* Alias / clinical description */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.9rem', marginBottom: 12, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
              {gene.alias}
            </div>

            {/* Gene class */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.9rem', marginBottom: 12 }}>
              <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Molecular Biology</div>
              <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{gene.gene_class}</div>
            </div>

            {/* Key alerts */}
            <div style={{ marginBottom: 12 }}>
              <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Clinical Alerts</div>
              {(gene.key_alerts || []).map((a, i) => <AlertCard key={i} alert={a} color={GENE_COLORS[gene.gene] || '#f59e0b'} />)}
            </div>

            {/* Etiologies */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.9rem', marginBottom: 12 }}>
              <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Etiologies (n={gene.n_patients})</div>
              {Object.entries(gene.etiologies || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #1e293b', fontSize: 12 }}>
                  <span style={{ color: '#cbd5e1' }}>{k}</span>
                  <span style={{ color: GENE_COLORS[gene.gene] || '#a5b4fc', fontWeight: 700 }}>{v}</span>
                </div>
              ))}
            </div>

            {/* Stats */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.9rem', marginBottom: 12 }}>
              <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Cohort Statistics</div>
              {Object.entries(gene.stats || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>{k.replace(/_/g, ' ')}</span>
                  <span style={{ color: GENE_COLORS[gene.gene] || '#a5b4fc', fontWeight: 600 }}>{v}</span>
                </div>
              ))}
            </div>

            {/* Dx delay */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.9rem' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>Diagnosis Delay Distribution</div>
              {Object.entries(gene.dx_delay_distribution || {}).map(([band, n]) => (
                <div key={band} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                  <span style={{ width: 60, color: '#94a3b8', fontSize: 12 }}>{band}</span>
                  <div style={{
                    height: 14, borderRadius: 3,
                    width: `${Math.round(n / gene.n_patients * 100) * 2}px`,
                    background: GENE_COLORS[gene.gene] || '#6366f1', minWidth: 4,
                  }} />
                  <span style={{ color: '#cbd5e1', fontSize: 12 }}>{n}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── DEFINITIONS TAB ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h2 style={{ color: '#e2e8f0', marginBottom: 16 }}>Reference Definitions</h2>
      {Object.entries(data.concepts || {}).map(([title, text]) => (
        <div key={title} style={{ background: '#1e293b', borderRadius: 8, padding: '1rem', marginBottom: 12 }}>
          <h3 style={{ color: '#a5b4fc', marginBottom: 8, fontSize: 14 }}>{title}</h3>
          <p style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{text}</p>
        </div>
      ))}

      <h2 style={{ color: '#e2e8f0', margin: '24px 0 12px' }}>Pharmacological Distinctions</h2>
      {(data.pharmacological_distinctions || []).map((d, i) => {
        const [title, ...rest] = d.split(':');
        return (
          <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: '1rem', marginBottom: 10 }}>
            <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 13, marginBottom: 4 }}>{title}</div>
            <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 }}>{rest.join(':')}</div>
          </div>
        );
      })}

      <h2 style={{ color: '#e2e8f0', margin: '24px 0 12px' }}>Key Standards &amp; Guidelines</h2>
      {(data.key_standards || []).map((s, i) => {
        const [title, ...rest] = s.split(':');
        return (
          <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: '1rem', marginBottom: 10 }}>
            <div style={{ fontWeight: 700, color: '#34d399', fontSize: 13, marginBottom: 4 }}>{title}</div>
            <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 }}>{rest.join(':')}</div>
          </div>
        );
      })}
    </div>
  );
}

// ── MAIN PAGE ─────────────────────────────────────────────────────────────────
export default function HereditaryPlateletDisorderAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-platelet-disorder-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  const PAGE = {
    background: '#0f172a', minHeight: '100vh', color: '#e2e8f0',
    fontFamily: "'Inter', 'Segoe UI', sans-serif",
  };
  const HEADER = {
    background: 'linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%)',
    padding: '1.5rem 2rem', borderBottom: '1px solid #1e293b',
  };
  const CONTENT = { padding: '1.5rem 2rem', maxWidth: 1400, margin: '0 auto' };
  const TAB_BAR = { display: 'flex', gap: 4, marginBottom: 24, borderBottom: '1px solid #1e293b', paddingBottom: 0 };

  return (
    <div style={PAGE}>
      <div style={HEADER}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, color: '#e2e8f0' }}>
          🩸 Hereditary Platelet Disorder Atlas
        </h1>
        <p style={{ margin: '4px 0 0', color: '#94a3b8', fontSize: 12 }}>
          Complete 8-Gene Hereditary Platelet Disorder Reference — ITGA2B · ITGB3 · GP1BA · GP1BB · MYH9 · ANKRD26 · ETV6 · RUNX1 — 320 patients (seeds 1606–1613)
        </p>
      </div>
      <div style={CONTENT}>
        {error && <ErrorBox msg={error} />}
        <div style={TAB_BAR}>
          {TABS.map(t => (
            <button key={t} onClick={() => setActiveTab(t)} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              padding: '8px 16px', fontSize: 13, fontWeight: 600,
              color: activeTab === t ? '#a5b4fc' : '#64748b',
              borderBottom: activeTab === t ? '2px solid #6366f1' : '2px solid transparent',
              marginBottom: -1,
            }}>{t}</button>
          ))}
        </div>
        {activeTab === 'Overview'      && <OverviewTab data={overview} />}
        {activeTab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
        {activeTab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
        {activeTab === 'Definitions'   && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
