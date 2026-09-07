'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TPO:     '#dc2626',  // red    — DH2A; perchlorate positive; most common dyshormonogenesis
  TG:      '#ea580c',  // orange — DH3; serum Tg LOW despite TSH HIGH; largest human protein
  SLC26A4: '#ca8a04',  // amber  — Pendred; EVA PATHOGNOMONIC; SNHL + goitre
  SLC5A5:  '#16a34a',  // green  — NIS; RAI uptake ABSENT; iodide transport defect
  DUOX2:   '#2563eb',  // blue   — DH6; H2O2 failure; most common worldwide; transient NH
  TSHR:    '#7c3aed',  // purple — TSH resistance (LOF) or familial hyperthyroidism (GOF)
  PAX8:    '#0f766e',  // teal   — thyroid dysgenesis; renal anomaly 50%
  FOXE1:   '#be185d',  // pink   — Bamforth-Lazarus; agenesis + cleft palate + spiky hair
};

const GENE_DISEASE = {
  TPO:     'AR Dyshormonogenesis-2A — TPO-933aa — 2p25.3 — Thyroid-Peroxidase-Iodide-Oxidation-Coupling — Perchlorate-Discharge-POSITIVE-PATHOGNOMONIC — SNHL-ABSENT-DDx-Pendred — Most-Common-Dyshormonogenesis',
  TG:      'AR Dyshormonogenesis-3 — TG-2767aa — 8q24.22 — Thyroglobulin-Prohormone-Scaffold-Largest-Secreted-Protein — Serum-Tg-LOW-ABSENT-Despite-TSH-HIGH-PATHOGNOMONIC — Afrikaner-Founder-p.G2229R',
  SLC26A4: 'AR Pendred-Syndrome — SLC26A4-780aa — 7q22.3 — Pendrin-Anion-Exchanger-Thyroid-Inner-Ear — EVA-CT-Temporal-Bone-PATHOGNOMONIC — SNHL-5-10pct-Hereditary — EVA-Screen-Before-Cochlear-Implant',
  SLC5A5:  'AR Iodide-Transport-Defect — SLC5A5-643aa — 19p13.11 — NIS-Sodium-Iodide-Symporter-Basolateral — RAI-Uptake-ABSENT-PATHOGNOMONIC — Perchlorate-Discharge-NEGATIVE — High-Iodine-May-Help',
  DUOX2:   'AR-biallelic / Dominant-heterozygous Dyshormonogenesis-6 — DUOX2-1548aa — 15q21.1 — Dual-Oxidase-H2O2-Generation — Most-Common-DH-Worldwide-Japan-Asia — Monoallelic-Transient-Neonatal-Hypothyroidism',
  TSHR:    'AR-LOF TSH-Resistance / AD-GOF Familial-Hyperthyroidism — TSHR-764aa — 14q31.1 — Thyrotropin-Receptor-GPCR — LOF-Elevated-TSH-PLUS-Normal-Gland-No-Goitre — GOF-TRAb-NEGATIVE-Neonatal-Hyperthyroidism',
  PAX8:    'AD Thyroid-Dysgenesis — PAX8-457aa — 2q14.1 — Paired-Box-8-Transcription-Factor — Renal-Anomaly-50pct-PATHOGNOMONIC — Lingual-Thyroid-Ectopic — Do-NOT-Ablate-Ectopic-Thyroid — Variable-Expressivity',
  FOXE1:   'AR Bamforth-Lazarus-Syndrome — FOXE1-373aa — 9q22.33 — Forkhead-Box-E1-TTF2 — Thyroid-Agenesis-Cleft-Palate-Spiky-Hair-PATHOGNOMONIC-TRIAD — Only-Agenesis-WITH-Dysmorphia — Choanal-Atresia-Airway-Emergency',
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

function Alert({ text }) {
  const lv = (text || '').includes('PATHOGNOMONIC') || (text || '').includes('EMERGENCY') || (text || '').includes('MANDATORY') || (text || '').includes('ABSENT') ? 'critical'
           : (text || '').includes('POSITIVE') || (text || '').includes('NOT') || (text || '').includes('DISTINGUISH') ? 'warning' : 'info';
  const colors = { critical: '#fca5a5', warning: '#fcd34d', info: '#93c5fd' };
  const bg     = { critical: '#450a0a', warning: '#451a03', info: '#0c1a3a' };
  return (
    <div style={{
      background: bg[lv], borderRadius: 6, padding: '0.55rem 0.9rem',
      color: colors[lv], fontSize: 12, margin: '4px 0', lineHeight: 1.5,
    }}>{text}</div>
  );
}

function GeneChip({ gene, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: selected ? GENE_COLORS[gene] : '#1e293b',
        color: selected ? '#fff' : '#94a3b8',
        border: `2px solid ${GENE_COLORS[gene] || '#334155'}`,
        borderRadius: 20, padding: '4px 14px', cursor: 'pointer',
        fontWeight: selected ? 700 : 400, fontSize: 13, margin: '3px',
        transition: 'all 0.15s',
      }}
    >{gene}</button>
  );
}

/* ─── TAB: Overview ─────────────────────────────────────────────────────────── */
function OverviewTab() {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-thyroid-disorder-atlas/overview`)
      .then(r => r.json()).then(setData).catch(e => setErr(e.message));
  }, []);

  if (err)  return <ErrorBox msg={err} />;
  if (!data) return <Loading />;

  const s = data.aggregate_stats || {};
  return (
    <div>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: '1.2rem' }}>{data.subtitle}</p>

      {/* KPIs */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
        <KPI label="Total Patients"    value={s.total_patients}        color="#6366f1" />
        <KPI label="Genes Covered"     value={s.genes_covered}         color="#22d3ee" />
        <KPI label="Avg Age Dx (yr)"   value={s.avg_age_at_diagnosis_yr} color="#f59e0b" />
        <KPI label="Severe Cases %"    value={`${s.severe_cases_pct}%`} color="#ef4444" />
        <KPI label="Seed Range"        value={s.seed_range}            color="#a3e635" />
      </div>

      {/* Gene summary cards */}
      <h3 style={{ color: '#e2e8f0', marginBottom: '0.8rem' }}>Gene Overview</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
        {(data.gene_summary || []).map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <span style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', margin: '6px 0 4px' }}>
              <strong style={{ color: '#f59e0b' }}>Biomarker: </strong>{g.key_biomarker}
            </div>
            <div style={{ fontSize: 11, color: '#fca5a5', marginBottom: 6 }}>
              <strong>Pathognomonic: </strong>{g.pathognomonic}
            </div>
            <div style={{ fontSize: 11, color: '#86efac', marginBottom: 8 }}>
              <strong>Treatment: </strong>{g.treatment}
            </div>
            <div style={{ fontSize: 11, color: '#64748b' }}>n = {g.n_patients} patients</div>
          </div>
        ))}
      </div>

      {/* Clinical distinctions */}
      <h3 style={{ color: '#e2e8f0', margin: '1.5rem 0 0.8rem' }}>Key Clinical Distinctions</h3>
      <div style={{ background: '#0f172a', borderRadius: 8, padding: '1rem' }}>
        {(data.key_clinical_distinctions || []).map((d, i) => (
          <div key={i} style={{ fontSize: 12, color: '#cbd5e1', padding: '4px 0', borderBottom: '1px solid #1e293b' }}>
            <span style={{ color: '#f59e0b', marginRight: 8 }}>▸</span>{d}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── TAB: Gene Table ───────────────────────────────────────────────────────── */
function GeneTableTab() {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);
  const [sel, setSel]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-thyroid-disorder-atlas/breakdown`)
      .then(r => r.json()).then(setData).catch(e => setErr(e.message));
  }, []);

  if (err)  return <ErrorBox msg={err} />;
  if (!data) return <Loading />;

  const genes = data.genes || [];
  const active = sel ? genes.find(g => g.gene === sel) : null;

  return (
    <div>
      <div style={{ marginBottom: '1rem' }}>
        {genes.map(g => (
          <GeneChip key={g.gene} gene={g.gene} selected={sel === g.gene}
            onClick={() => setSel(sel === g.gene ? null : g.gene)} />
        ))}
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a', color: '#94a3b8' }}>
              {['Gene','Locus','aa','Inh.','Pathognomonic','Tx','n'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #1e293b' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes.map((g, i) => (
              <tr key={g.gene} onClick={() => setSel(sel === g.gene ? null : g.gene)}
                style={{
                  background: sel === g.gene ? '#1e293b' : i % 2 === 0 ? '#0f172a' : '#111827',
                  cursor: 'pointer',
                  borderLeft: sel === g.gene ? `4px solid ${GENE_COLORS[g.gene]}` : '4px solid transparent',
                }}>
                <td style={{ padding: '7px 10px', color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700 }}>{g.gene}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.protein_size}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                <td style={{ padding: '7px 10px', color: '#fca5a5', maxWidth: 260, fontSize: 11 }}>{g.pathognomonic}</td>
                <td style={{ padding: '7px 10px', color: '#86efac', maxWidth: 220, fontSize: 11 }}>{g.treatment}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.n_patients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Detail panel */}
      {active && (
        <div style={{ marginTop: '1.5rem', background: '#1e293b', borderRadius: 10, padding: '1.2rem',
          borderLeft: `4px solid ${GENE_COLORS[active.gene]}` }}>
          <h3 style={{ color: GENE_COLORS[active.gene], marginBottom: '0.8rem' }}>{active.gene} — Detail</h3>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: '0.8rem' }}>{active.protein}</div>
          <h4 style={{ color: '#e2e8f0', marginBottom: 6 }}>Critical Flags</h4>
          {(active.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
          <h4 style={{ color: '#e2e8f0', margin: '1rem 0 6px' }}>Severity Distribution</h4>
          <div style={{ display: 'flex', gap: 10 }}>
            {Object.entries(active.severity_distribution || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 14px', textAlign: 'center' }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: k === 'severe' ? '#ef4444' : k === 'moderate' ? '#f59e0b' : '#86efac' }}>{v}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>{k}</div>
              </div>
            ))}
          </div>
          <h4 style={{ color: '#e2e8f0', margin: '1rem 0 6px' }}>Sample Patients (first 5)</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ color: '#64748b' }}>
                {['ID','Age Dx','Sex','Severity','Locus'].map(h => (
                  <th key={h} style={{ padding: '4px 8px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(active.patients || []).map(p => (
                <tr key={p.patient_id}>
                  <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                  <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.age_at_diagnosis_yr}y</td>
                  <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.sex}</td>
                  <td style={{ padding: '3px 8px', color: p.severity === 'severe' ? '#ef4444' : p.severity === 'moderate' ? '#f59e0b' : '#86efac' }}>{p.severity}</td>
                  <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.locus}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ─── TAB: Clinical Atlas ───────────────────────────────────────────────────── */
function ClinicalAtlasTab() {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);
  const [sel, setSel]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-thyroid-disorder-atlas/breakdown`)
      .then(r => r.json()).then(setData).catch(e => setErr(e.message));
  }, []);

  if (err)  return <ErrorBox msg={err} />;
  if (!data) return <Loading />;

  const genes = data.genes || [];

  return (
    <div>
      <p style={{ color: '#94a3b8', fontSize: 12, marginBottom: '1rem' }}>
        Select gene to view full clinical profile with all critical flags, biomarkers, and treatment protocols.
      </p>
      <div style={{ marginBottom: '1rem' }}>
        {genes.map(g => (
          <GeneChip key={g.gene} gene={g.gene} selected={sel === g.gene}
            onClick={() => setSel(sel === g.gene ? null : g.gene)} />
        ))}
      </div>

      {!sel && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(300px,1fr))', gap: 12 }}>
          {genes.map(g => (
            <div key={g.gene} onClick={() => setSel(g.gene)}
              style={{ background: '#1e293b', borderRadius: 8, padding: '1rem', cursor: 'pointer',
                borderLeft: `4px solid ${GENE_COLORS[g.gene]}`,
                transition: 'background 0.15s' }}>
              <div style={{ color: GENE_COLORS[g.gene], fontWeight: 700, fontSize: 15, marginBottom: 4 }}>{g.gene}</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>{GENE_DISEASE[g.gene]}</div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>
                {(g.critical_flags || []).length} critical flags · {g.n_patients} patients
              </div>
            </div>
          ))}
        </div>
      )}

      {sel && (() => {
        const g = genes.find(x => x.gene === sel);
        if (!g) return null;
        return (
          <div style={{ background: '#1e293b', borderRadius: 10, padding: '1.5rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene]}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h2 style={{ color: GENE_COLORS[g.gene], margin: 0 }}>{g.gene}</h2>
              <button onClick={() => setSel(null)}
                style={{ background: '#334155', color: '#94a3b8', border: 'none', borderRadius: 6,
                  padding: '4px 12px', cursor: 'pointer', fontSize: 12 }}>← Back</button>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: '1rem' }}>
              {[
                ['Locus', g.locus], ['Size', g.protein_size], ['Inheritance', g.inheritance],
              ].map(([k, v]) => (
                <span key={k} style={{ background: '#0f172a', borderRadius: 4, padding: '3px 10px', fontSize: 11, color: '#94a3b8' }}>
                  <strong style={{ color: '#64748b' }}>{k}: </strong>{v}
                </span>
              ))}
            </div>

            <h4 style={{ color: '#f59e0b', marginBottom: 6 }}>Key Biomarker</h4>
            <div style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', fontSize: 12, color: '#fbbf24', marginBottom: '1rem' }}>
              {g.key_biomarker}
            </div>

            <h4 style={{ color: '#fca5a5', marginBottom: 6 }}>Pathognomonic Finding</h4>
            <div style={{ background: '#450a0a', borderRadius: 6, padding: '8px 12px', fontSize: 12, color: '#fca5a5', marginBottom: '1rem' }}>
              {g.pathognomonic}
            </div>

            <h4 style={{ color: '#86efac', marginBottom: 6 }}>Treatment Protocol</h4>
            <div style={{ background: '#052e16', borderRadius: 6, padding: '8px 12px', fontSize: 12, color: '#86efac', marginBottom: '1rem' }}>
              {g.treatment}
            </div>

            <h4 style={{ color: '#e2e8f0', marginBottom: 8 }}>Critical Clinical Flags</h4>
            {(g.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
          </div>
        );
      })()}
    </div>
  );
}

/* ─── TAB: Definitions ──────────────────────────────────────────────────────── */
function DefinitionsTab() {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-thyroid-disorder-atlas/definitions`)
      .then(r => r.json()).then(setData).catch(e => setErr(e.message));
  }, []);

  if (err)  return <ErrorBox msg={err} />;
  if (!data) return <Loading />;

  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: '1rem' }}>Gene Definitions</h3>
      {(data.genes || []).map(g => (
        <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: '1rem', marginBottom: 10,
          borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ color: GENE_COLORS[g.gene], fontWeight: 700, fontSize: 15 }}>{g.gene}</span>
            <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.protein_size} · {g.inheritance} · onset: {g.age_of_onset}</span>
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6, marginBottom: 8 }}>{g.definition}</div>
          <div style={{ fontSize: 11, color: '#64748b' }}>
            <strong style={{ color: '#475569' }}>Critical flags: </strong>
            {(g.critical_flags || []).join(' | ')}
          </div>
        </div>
      ))}

      <h3 style={{ color: '#e2e8f0', margin: '2rem 0 1rem' }}>Glossary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 10 }}>
        {Object.entries(data.glossary || {}).map(([term, def]) => (
          <div key={term} style={{ background: '#1e293b', borderRadius: 6, padding: '0.8rem' }}>
            <div style={{ color: '#a5b4fc', fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{term}</div>
            <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.5 }}>{def}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Main Page ─────────────────────────────────────────────────────────────── */
export default function HereditaryThyroidDisorderAtlasPage() {
  const [tab, setTab] = useState(0);

  const tabContent = [
    <OverviewTab key="ov" />,
    <GeneTableTab key="gt" />,
    <ClinicalAtlasTab key="ca" />,
    <DefinitionsTab key="df" />,
  ];

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', padding: '1.5rem' }}>
      {/* Header */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', margin: 0 }}>
          🧬 Hereditary-Thyroid-Disorder-Atlas
        </h1>
        <p style={{ color: '#64748b', fontSize: 13, margin: '4px 0 0' }}>
          Complete 8-Gene Congenital Hypothyroidism &amp; Thyroid Development Disorders Atlas —
          TPO · TG · SLC26A4 (Pendred) · SLC5A5 (NIS) · DUOX2 · TSHR · PAX8 · FOXE1 (Bamforth-Lazarus)
          | 320 patients · seeds 1886–1893
        </p>
      </div>

      {/* Gene chips header */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: '1.2rem' }}>
        {Object.entries(GENE_COLORS).map(([gene, color]) => (
          <span key={gene} style={{
            background: color + '22', border: `1px solid ${color}`,
            color, borderRadius: 16, padding: '2px 10px', fontSize: 11, fontWeight: 600,
          }}>{gene}</span>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', borderBottom: '1px solid #1e293b', paddingBottom: 4 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            background: tab === i ? '#6366f1' : 'transparent',
            color: tab === i ? '#fff' : '#64748b',
            border: 'none', borderRadius: '6px 6px 0 0', padding: '8px 18px',
            cursor: 'pointer', fontWeight: tab === i ? 700 : 400, fontSize: 13,
            transition: 'all 0.15s',
          }}>{t}</button>
        ))}
      </div>

      {/* Tab content */}
      <div>{tabContent[tab]}</div>
    </div>
  );
}
