'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  G6PC:    '#dc2626',  // red — GSD Ia von Gierke, NO fructose/galactose
  SLC37A4: '#b45309',  // amber — GSD Ib neutropenia, G-CSF mandatory
  AGL:     '#1d4ed8',  // blue — GSD III Cori/Forbes, high protein
  GBE1:    '#7c3aed',  // purple — GSD IV Andersen, polyglucosan, APBD
  PYGM:    '#0f766e',  // teal — GSD V McArdle, second wind, no statins
  PYGL:    '#065f46',  // dark green — GSD VI Hers, benign ketosis
  PHKA2:   '#c2410c',  // deep orange — GSD IXa most common GSD, X-linked
  GYS2:    '#1e1b4b',  // dark indigo — GSD 0a, fasting hypo+postprandial hyper, no hepatomegaly
};

const GENE_DISEASE = {
  G6PC:    'AR GSD Ia von Gierke — Glucose-6-Phosphatase-357aa — 17q21.31 — NO-Fructose-Galactose-Sucrose-ABSOLUTE-CI — Corn-Starch-KEYSTONE — Flat-Glucagon-PATHOGNOMONIC',
  SLC37A4: 'AR GSD Ib — G6P-Translocase-429aa — 11q23.3 — Neutropenia-PATHOGNOMONIC-DDx-Ia — G-CSF-Mandatory-ANC>1.5 — IBD-Like-Empagliflozin-Emerging',
  AGL:     'AR GSD III Cori/Forbes — Debranching-Enzyme-1532aa — 1p21.2 — IIIa-Liver+Muscle-IIIb-Liver-Only — High-Protein-KEY — Liver-Improves-Puberty-UNIQUE',
  GBE1:    'AR GSD IV Andersen — Branching-Enzyme-702aa — 3p12.3 — Polyglucosan-Accumulation — Liver-Transplant-Curative-Classic — APBD-Adult-Neurogenic-Bladder-Ashkenazi',
  PYGM:    'AR GSD V McArdle — Muscle-Phosphorylase-841aa — 11q13.1 — IEFT-Lactate-Flat-PATHOGNOMONIC — Second-Wind-PATHOGNOMONIC — NO-Statins-ABSOLUTE-CI',
  PYGL:    'AR GSD VI Hers — Liver-Phosphorylase-848aa — 14q22.1 — Benign-Ketotic-Hypoglycaemia — No-Lactic-Acidosis-KEY-DDx-I — Spontaneous-Improvement-Puberty',
  PHKA2:   'XLR GSD IXa — Phosphorylase-Kinase-Alpha2-1235aa — Xp22.13 — Most-Common-GSD-Overall-1:100000 — Benign-Liver-Resolves-Puberty — Gene-Panel-Mandatory',
  GYS2:    'AR GSD 0a — Liver-Glycogen-Synthase-703aa — 12p12.1 — Fasting-Hypo+Postprandial-Hyper-UNIQUE — No-Hepatomegaly — NBS-MISSES — Insulin-ABSOLUTE-CI',
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
        <KPI label="AR Genes" value={s.ar_genes} color="#dc2626" />
        <KPI label="X-Linked Genes" value={s.x_linked_genes} color="#7c3aed" />
        <KPI label="Patients / Gene" value={s.patients_per_gene} color="#0f766e" />
        <KPI label="G6PC Hepatomegaly %" value={`${s.g6pc_hepatomegaly_pct}%`} color="#dc2626" />
        <KPI label="GSD Ib Neutropenia %" value={`${s.slc37a4_neutropenia_pct}%`} color="#b45309" />
        <KPI label="McArdle Second Wind %" value={`${s.pygm_second_wind_pct}%`} color="#0f766e" />
        <KPI label="GSD IXa Most Common" value="1:100k" color="#c2410c" />
        <KPI label="GSD 0a NBS Detected %" value={`${s.gys2_nbs_detected_pct ?? 0}%`} color="#1e1b4b" />
        <KPI label="GSD 0a Insulin Error %" value={`${s.gys2_insulin_error_pct}%`} color="#ef4444" />
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
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{g.locus} · {g.aa} aa · {g.inheritance}</div>
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
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>8-Gene Hereditary Glycogen Storage Disease Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Protein', 'Locus', 'AA', 'kDa', 'OMIM Gene', 'Inheritance', 'Gene Class', 'Key Intervention'].map(h => (
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
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 200 }}>{g.gene_class}</td>
                <td style={{ padding: '8px 10px', fontWeight: 600, color:
                  g.gene === 'SLC37A4' ? '#b45309' :
                  g.gene === 'PYGM' ? '#dc2626' :
                  g.gene === 'GYS2' ? '#1e1b4b' :
                  '#94a3b8' }}>
                  {g.gene === 'SLC37A4' ? 'G-CSF + Empagliflozin' :
                   g.gene === 'PYGM' ? 'No Statins / Sucrose pre-exercise' :
                   g.gene === 'GBE1' ? 'Liver Transplant (classic)' :
                   g.gene === 'G6PC' ? 'No fructose/galactose + UCCS' :
                   g.gene === 'GYS2' ? 'No insulin / UCCS bedtime' :
                   'UCCS bedtime / avoid fasting'}
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

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Etiology / Presentation Distribution</h4>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(g.etiologies || {}).map(([k, v]) => (
            <div key={k} style={{
              background: '#1e293b', borderRadius: 8, padding: '8px 12px',
              fontSize: 12, color: '#94a3b8',
            }}>
              <span style={{ fontWeight: 600, color: GENE_COLORS[g.gene] }}>{v}%</span> {k.replace(/_/g, ' ')}
            </div>
          ))}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Sample Patients (first 10)</h4>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                <th style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b' }}>ID</th>
                {Object.keys(g.sample_patients?.[0] || {}).filter(k => k !== 'patient_id').slice(0, 6).map(k => (
                  <th key={k} style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b' }}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(g.sample_patients || []).map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '5px 8px', color: GENE_COLORS[g.gene] }}>{p.patient_id}</td>
                  {Object.entries(p).filter(([k]) => k !== 'patient_id').slice(0, 6).map(([k, v]) => (
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
    </div>
  );
}

export default function HeredGSDAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-gsd-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-gsd-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-gsd-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary GSD Atlas — Complete 8-Gene Glycogen Storage Disease Reference
        </h1>
        <p style={{ color: '#64748b', fontSize: 13 }}>
          G6PC (GSD-Ia-No-Fructose) · SLC37A4 (GSD-Ib-Neutropenia-G-CSF) · AGL (GSD-III-High-Protein) ·
          GBE1 (GSD-IV-APBD-Polyglucosan) · PYGM (McArdle-Second-Wind-No-Statins) ·
          PYGL (GSD-VI-Benign-Ketosis) · PHKA2 (GSD-IXa-Most-Common-XLR) ·
          GYS2 (GSD-0a-No-Hepatomegaly-Insulin-CI) — 320 Patients · Seeds 1806–1813
        </p>
      </div>

      {error && <ErrorBox msg={error} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 8,
            background: tab === i ? '#dc2626' : '#1e293b',
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
