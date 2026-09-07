'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ABCC8:    '#dc2626',  // red — most common CHI, focal/diffuse, surgical
  KCNJ11:   '#b45309',  // amber — CHI + NDM dual phenotype
  GLUD1:    '#7c3aed',  // purple — HI/HA ammonia elevated
  GCK:      '#0f766e',  // teal — mild activating HI
  HADH:     '#1d4ed8',  // blue — SCHAD protein-sensitive
  SLC16A1:  '#c2410c',  // deep orange — exercise-induced EIHI
  HNF4A:    '#065f46',  // dark green — neonatal HI → MODY1
  FOXA2:    '#1e1b4b',  // dark indigo — HI + hypopituitarism absent glucagon
};

const GENE_DISEASE = {
  ABCC8:   'AR/AD CHI1 — SUR1-KATP-Regulatory — 1581aa — 11p15.1 — Most-Common-CHI-40-50pct — Focal-vs-Diffuse-18F-DOPA-PET-MANDATORY — Diazoxide-UNRESPONSIVE — Octreotide → Pancreatectomy-Focal-Curative-95pct',
  KCNJ11:  'AR CHI2 (LOF) + AD NDM (GOF-Activating) — Kir6.2-KATP-Pore — 390aa — 11p15.1 — Sulphonylurea-RESCUES-NDM-90pct — DEND-Syndrome-GOF-Severe — CHI-Same-ABCC8-Management',
  GLUD1:   'AD GOF Hyperinsulinism-Hyperammonaemia-HI/HA — GDH-558aa — 10q23.33 — Leucine-Sensitive — Ammonia-60-200-μmol/L-Fasting-AND-Postprandial-PATHOGNOMONIC — Diazoxide-RESPONSIVE — Valproate-AVOID',
  GCK:     'AD GOF Activating-GCK-HI — Glucokinase-465aa — 7p13 — Glucose-Sensor-Set-Point-LOWERED — Mild-Fasting-Glucose-2.5-3.5-mmol — NOT-MODY2 (LOF-OPPOSITE) — Diazoxide-RESPONSIVE',
  HADH:    'AR SCHAD-Deficiency — 3-OH-Acyl-CoA-Dehydrogenase — 314aa — 4q25 — Protein-Sensitive — 3-Hydroxyglutaric-Acid-Urine-PATHOGNOMONIC — Ammonia-NORMAL-KEY-DDx-GLUD1 — Diazoxide-RESPONSIVE',
  SLC16A1: 'AD GOF Promoter Exercise-Induced-HI-EIHI — MCT1-494aa — 1p13.2 — Pyruvate-Entry-During-Exercise — Hypoglycaemia-30-60min-Post-Exercise — WES-MISSES-Promoter — Diazoxide-INEFFECTIVE',
  HNF4A:   'AD Neonatal-HI → MODY1 — HNF4α-455aa — 20q13.12 — Macrosomia-LGA-PATHOGNOMONIC-Clue — HI-Resolves-60pct-Infancy — MODY1-Inevitable — Annual-Glucose-From-Age-10 — SU-Responsive',
  FOXA2:   'AD HI + Hypopituitarism-Triad-GH-ACTH-TSH — HNF3β-1161aa — 20p11.21 — ABSENT-Glucagon-PATHOGNOMONIC — Pituitary-MRI-MANDATORY — Glucagon-Injection-INEFFECTIVE — IV-Dextrose-ONLY-Emergency',
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

function Alert({ text, color }) {
  return (
    <div style={{
      background: '#0f172a', border: `1px solid ${color || '#334155'}`,
      borderRadius: 8, padding: '0.65rem 1rem', marginBottom: 8,
      fontSize: 12, color: '#e2e8f0', lineHeight: 1.5,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats || {};
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 6 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 20 }}>{data.subtitle}</p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
        <KPI label="Total Patients" value={data.total_patients} color="#6366f1" />
        <KPI label="Genes Covered" value={s.genes_covered} color="#22c55e" />
        <KPI label="Seeds" value={data.seed_range} color="#f59e0b" />
        <KPI label="KATP Channel Genes" value={s.katp_channel_genes} color="#dc2626" />
        <KPI label="Enzyme GOF Genes" value={s.enzyme_gain_of_function_genes} color="#7c3aed" />
        <KPI label="TF Genes" value={s.transcription_factor_genes} color="#065f46" />
      </div>

      <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>🚨 Critical Clinical Alerts</h3>
      {(data.top_alerts || []).map((a, i) => (
        <Alert key={i} text={a} color={Object.values(GENE_COLORS)[i % 8]} />
      ))}

      <h3 style={{ color: '#e2e8f0', marginTop: 24, marginBottom: 10 }}>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 }}>
        {(data.genes || []).map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{g.locus} · {g.aa} aa</div>
            <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 6 }}>
              {GENE_DISEASE[g.gene]?.split(' — ')[0]}
            </div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>n={g.n_patients} patients</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>8-Gene Hereditary Hyperinsulinism Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Protein / Alias', 'Locus', 'AA', 'kDa', 'OMIM Gene', 'Inheritance', 'Gene Class', 'Diazoxide'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 280, wordBreak: 'break-word' }}>{g.protein}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.aa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.kDa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.omim_gene}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 200 }}>{g.inheritance}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 200 }}>{g.gene_class}</td>
                <td style={{ padding: '8px 10px', color: g.gene === 'ABCC8' || g.gene === 'SLC16A1' ? '#dc2626' : '#22c55e', fontWeight: 600 }}>
                  {g.gene === 'ABCC8' || g.gene === 'SLC16A1' ? 'INEFFECTIVE' : 'Responsive'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(0);
  if (!data) return <Loading />;
  const g = data[selected];
  if (!g) return null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 20 }}>
      <div>
        {data.map((gene, i) => (
          <button key={gene.gene} onClick={() => setSelected(i)} style={{
            display: 'block', width: '100%', textAlign: 'left',
            padding: '10px 14px', marginBottom: 6, borderRadius: 8,
            background: selected === i ? GENE_COLORS[gene.gene] : '#1e293b',
            color: selected === i ? '#fff' : '#94a3b8',
            border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
          }}>
            {gene.gene}
            <div style={{ fontSize: 10, fontWeight: 400, marginTop: 2 }}>{gene.locus}</div>
          </button>
        ))}
      </div>
      <div>
        <h3 style={{ color: GENE_COLORS[g.gene] || '#a5b4fc', marginBottom: 4 }}>{g.gene}</h3>
        <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 12 }}>{g.inheritance} · {g.locus} · {g.aa} aa · {g.kDa} kDa</div>

        <h4 style={{ color: '#e2e8f0', marginBottom: 6 }}>🚨 Key Alerts</h4>
        {(g.key_alerts || []).map((a, i) => (
          <Alert key={i} text={a} color={GENE_COLORS[g.gene]} />
        ))}

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Clinical Description</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '1rem',
          fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap',
          maxHeight: 400, overflowY: 'auto',
        }}>
          {g.alias}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Known Etiologies / Variants</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Variant', 'Type', 'Frequency', 'Severity'].map(h => (
                <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(g.etiologies || []).map((e, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                <td style={{ padding: '6px 8px', color: GENE_COLORS[g.gene], fontWeight: 600 }}>{e.variant}</td>
                <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{e.type}</td>
                <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{e.frequency}</td>
                <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{e.severity}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Sample Patients (first 10)</h4>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                <th style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b' }}>ID</th>
                {Object.keys(g.sample_patients?.[0] || {}).filter(k => k !== 'gene' && k !== 'seed').slice(0, 6).map(k => (
                  <th key={k} style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b' }}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(g.sample_patients || []).map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '5px 8px', color: GENE_COLORS[g.gene] }}>{p.patient_id}</td>
                  {Object.entries(p).filter(([k]) => k !== 'gene' && k !== 'seed' && k !== 'patient_id').slice(0, 6).map(([k, v]) => (
                    <td key={k} style={{ padding: '5px 8px', color: '#94a3b8' }}>
                      {typeof v === 'boolean' ? (v ? '✓' : '✗') : String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 16 }}>Definitions & Clinical Concepts</h3>
      {Object.entries(data.concepts || {}).map(([title, body]) => (
        <div key={title} style={{ marginBottom: 20 }}>
          <h4 style={{ color: '#a5b4fc', marginBottom: 6 }}>{title}</h4>
          <div style={{
            background: '#1e293b', borderRadius: 8, padding: '1rem',
            fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap',
          }}>
            {body}
          </div>
        </div>
      ))}
      <h3 style={{ color: '#e2e8f0', marginTop: 24, marginBottom: 12 }}>Pharmacological Distinctions</h3>
      {(data.pharmacological_distinctions || []).map((d, i) => (
        <Alert key={i} text={d} color="#6366f1" />
      ))}
      <h3 style={{ color: '#e2e8f0', marginTop: 24, marginBottom: 12 }}>Key Standards & Guidelines</h3>
      {(data.key_standards || []).map((s, i) => (
        <Alert key={i} text={s} color="#22c55e" />
      ))}
    </div>
  );
}

export default function HeredHIAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-hi-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-hi-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-hi-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary Hyperinsulinism Atlas — Complete 8-Gene Reference
        </h1>
        <p style={{ color: '#64748b', fontSize: 13 }}>
          ABCC8 (SUR1-CHI1) · KCNJ11 (Kir6.2-CHI2/NDM) · GLUD1 (HI/HA) · GCK (Activating) ·
          HADH (SCHAD) · SLC16A1 (EIHI) · HNF4A (→MODY1) · FOXA2 (HI+Hypopituitarism) —
          320 Patients · Seeds 1774–1781
        </p>
      </div>

      {error && <ErrorBox msg={error} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 8,
            background: tab === i ? '#6366f1' : '#1e293b',
            color: tab === i ? '#fff' : '#94a3b8',
            border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div>
        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <GeneTableTab data={breakdown} />}
        {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
        {tab === 3 && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
