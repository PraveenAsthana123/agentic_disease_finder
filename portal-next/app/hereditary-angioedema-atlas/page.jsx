'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SERPING1: '#1b5e20',  // deep green — C1-INH, HAE type I/II, most common
  F12:      '#e65100',  // deep orange — FXII GOF, estrogen-driven HAE type III
  PLG:      '#0d47a1',  // deep blue — plasminogen GOF, HAE-PLG
  ANGPT1:   '#4a148c',  // deep purple — Angiopoietin-1, Tie2 barrier
  MYOF:     '#880e4f',  // deep pink — Myoferlin, membrane repair
  KNG1:     '#37474f',  // dark slate — HMWK, bradykinin precursor
  HS3ST6:   '#b71c1c',  // deep red — heparan sulfate, contact activation
  KLKB1:    '#004d40',  // dark teal — prekallikrein, prolonged APTT no bleeding
};

const GENE_DISEASE = {
  SERPING1: 'HAE Type I/II AD — C1-INH LOF — C4 Low Between Attacks PATHOGNOMONIC — Bradykinin — Laryngeal Risk',
  F12:      'HAE-FXII/Type III AD GOF — Estrogen-Driven — C1-INH Normal — OCP Absolutely CI — Predominantly Females',
  PLG:      'HAE-PLG AD GOF — Lys330Glu — Plasmin Amplifies FXII — Tranexamic Acid Rational — Estrogen-Sensitive',
  ANGPT1:   'HAE-ANGPT1 AD — Angiopoietin-1 LOF — Tie2 Endothelial Barrier — Normal Complement — Genetic Panel Only',
  MYOF:     'HAE-MYOF AD — Myoferlin LOF — Membrane Repair Deficit — Endothelial Permeability — 2057 aa Large Gene',
  KNG1:     'HAE-KNG1 AD — High-MW Kininogen — Bradykinin Precursor — Icatibant First Choice — ACE-I Interaction',
  HS3ST6:   'HAE-HS3ST6 AD — Heparan Sulfate 3-O-Sulfotransferase — Contact Activation Surface — Genetic Panel Only',
  KLKB1:    'Prekallikrein Deficiency AR — Prolonged APTT NO Bleeding — Incubation Correction PATHOGNOMONIC — HAE Subset',
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
      <h2 style={{ color: '#e2e8f0', marginBottom: 4 }}>Hereditary Angioedema Atlas</h2>
      <p style={{ color: '#94a3b8', marginBottom: 20, fontSize: 13 }}>
        Complete 8-Gene Hereditary Angioedema Reference — {s.total_patients} patients (8×40, seeds 1614–1621) ·
        Mean diagnosis age {s.mean_dx_age_years}y · Mean delay {s.mean_dx_delay_months} months
      </p>

      {/* KPI row 1 — SERPING1 HAE type I/II */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        SERPING1-HAE — C1-INH Deficiency (Type I/II)
      </h3>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <KPI label="C4 Low Between Attacks" value={`${s.serping1_c4_low_pct}%`} color="#1b5e20" />
        <KPI label="Laryngeal Attacks (Life-Threatening)" value={`${s.serping1_laryngeal_pct}%`} color="#ef4444" />
        <KPI label="Abdominal Attacks (Mimic Acute Abdomen)" value={`${s.serping1_abdominal_pct}%`} color="#f59e0b" />
        <KPI label="Misdiagnosed as Allergy" value={`${s.serping1_misdiagnosed_allergy_pct}%`} color="#dc2626" />
        <KPI label="Unnecessary Surgery" value={`${s.serping1_unnecessary_surgery_pct}%`} color="#b45309" />
      </div>

      {/* KPI row 2 — Estrogen-driven HAE */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        Estrogen-Driven HAE (F12, PLG) — Normal Complement
      </h3>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <KPI label="F12-HAE Female Predominance" value={`${s.f12_female_pct}%`} color="#e65100" />
        <KPI label="F12-HAE OCP-Triggered" value={`${s.f12_ocp_triggered_pct}%`} color="#f97316" />
        <KPI label="PLG-HAE Tranexamic Effective" value={`${s.plg_tranexamic_effective_pct}%`} color="#0d47a1" />
      </div>

      {/* KPI row 3 — KLKB1 paradox */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        KLKB1 Prekallikrein Deficiency — APTT Trap
      </h3>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <KPI label="Prolonged APTT — NO Bleeding" value={`${s.klkb1_aptt_no_bleeding_pct}%`} color="#004d40" />
        <KPI label="Misdiagnosed as Coagulopathy" value={`${s.klkb1_misdiagnosed_coagulopathy_pct}%`} color="#dc2626" />
      </div>

      {/* Gene cards */}
      <h3 style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
        8 Genes — Hereditary Angioedema Reference
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
      <div style={{ minWidth: 120, background: '#1e293b', borderRadius: 10, padding: '0.8rem', height: 'fit-content' }}>
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
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #0f172a', fontSize: 12 }}>
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
export default function HereditaryAngioedemaAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-angioedema-atlas`;
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
    background: 'linear-gradient(135deg,#0c1a1a 0%,#0f172a 100%)',
    padding: '1.5rem 2rem', borderBottom: '1px solid #1e293b',
  };
  const CONTENT = { padding: '1.5rem 2rem', maxWidth: 1400, margin: '0 auto' };
  const TAB_BAR = { display: 'flex', gap: 4, marginBottom: 24, borderBottom: '1px solid #1e293b', paddingBottom: 0 };

  return (
    <div style={PAGE}>
      <div style={HEADER}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, color: '#e2e8f0' }}>
          💉 Hereditary Angioedema Atlas
        </h1>
        <p style={{ margin: '4px 0 0', color: '#94a3b8', fontSize: 12 }}>
          Complete 8-Gene Hereditary Angioedema Reference — SERPING1 · F12 · PLG · ANGPT1 · MYOF · KNG1 · HS3ST6 · KLKB1 — 320 patients (seeds 1614–1621)
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
        {activeTab === 'Overview'       && <OverviewTab data={overview} />}
        {activeTab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
        {activeTab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
        {activeTab === 'Definitions'    && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
