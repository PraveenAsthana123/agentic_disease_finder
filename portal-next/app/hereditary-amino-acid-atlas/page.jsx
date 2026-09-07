'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PAH:    '#dc2626',  // red — PKU most common aminoacidopathy, NBS universal
  CBS:    '#b45309',  // amber — homocystinuria, B6-responsive, ectopia lentis downward
  BCKDHA: '#1d4ed8',  // blue — MSUD, alloisoleucine PATHOGNOMONIC, maple syrup odour
  FAH:    '#7c3aed',  // purple — HT1, succinylacetone PATHOGNOMONIC, NTBC curative
  HGD:    '#0f766e',  // teal — alkaptonuria, urine darkens PATHOGNOMONIC, nitisinone
  GLDC:   '#065f46',  // dark green — NKH, CSF:plasma glycine >0.08 PATHOGNOMONIC
  OAT:    '#c2410c',  // deep orange — gyrate atrophy, plasma ornithine >10×
  TAT:    '#1e1b4b',  // dark indigo — HT2 Richner-Hanhart, keratosis triad, Tyr diet CURATIVE
};

const GENE_DISEASE = {
  PAH:    'AR PKU — Phenylalanine-Hydroxylase-452aa — 12q23.2 — Most-Common-Aminoacidopathy — NBS-Universal-DBS — BH4-Sapropterin-30pct-Responsive — Pegvaliase-Refractory',
  CBS:    'AR Homocystinuria — Cystathionine-Beta-Synthase-551aa — 21q22.3 — B6-Responsive-50pct-MANDATORY-3mo-Trial — Ectopia-Lentis-Downward — Thromboembolism-PATHOGNOMONIC',
  BCKDHA: 'AR MSUD-Type-1A — BCKDHA-445aa — 19q13.2 — Alloisoleucine-PATHOGNOMONIC-Plasma — Leucine-Most-Neurotoxic — Maple-Syrup-Odour — Mennonite-Founder',
  FAH:    'AR HT1-Tyrosinemia-Type-1 — Fumarylacetoacetase-419aa — 15q25.1 — Succinylacetone-PATHOGNOMONIC — NTBC-Nitisinone-Standard-of-Care — HCC-Risk-AFP-Surveillance',
  HGD:    'AR Alkaptonuria-AKU — Homogentisate-1,2-Dioxygenase-445aa — 3q13.33 — Urine-Darkens-Standing-PATHOGNOMONIC — Ochronosis-Cartilage-Pigment — NTBC-SONIA2-2019',
  GLDC:   'AR NKH-Glycine-Encephalopathy — Glycine-Decarboxylase-1020aa — 9p24.1 — CSF:Plasma-Glycine->0.08-PATHOGNOMONIC — Burst-Suppression-EEG — Sodium-Benzoate-Sink',
  OAT:    'AR Gyrate-Atrophy — Ornithine-Aminotransferase-439aa — 10q26.13 — Plasma-Ornithine->10x-Normal — Chorioretinal-Degeneration-Progressive — B6-5pct-Responsive',
  TAT:    'AR HT2-Richner-Hanhart — Tyrosine-Aminotransferase-454aa — 16q22.2 — Palmar-Plantar-Keratosis+Pseudodendritic-Keratitis-TRIAD — Low-Phe-Tyr-Diet-CURATIVE — No-Hepatic-HCC-Risk',
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
        <KPI label="On Diet %" value={`${s.on_diet_pct ?? 0}%`} color="#0f766e" />
        <KPI label="Family Cascade %" value={`${s.family_cascade_pct ?? 0}%`} color="#7c3aed" />
        <KPI label="Mild %" value={`${s.severity_mild_pct ?? 0}%`} color="#22c55e" />
        <KPI label="Moderate %" value={`${s.severity_moderate_pct ?? 0}%`} color="#f59e0b" />
        <KPI label="Severe %" value={`${s.severity_severe_pct ?? 0}%`} color="#dc2626" />
        <KPI label="Patients / Gene" value={s.patients_per_gene} color="#b45309" />
      </div>

      <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>🚨 Critical Treatment Alerts</h3>
      {(data.critical_treatment_alerts || []).slice(0, 16).map((a, i) => (
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
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              {g.locus} · {g.protein_size} aa · {g.inheritance}
            </div>
            <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 6 }}>
              {g.disorder}
            </div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
              Biomarker: {g.key_biomarker} · n={g.n_patients}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Rx: {g.treatment}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes ? Object.values(data.genes) : [];
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>8-Gene Hereditary Aminoacidopathy Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Locus', 'AA', 'Inheritance', 'Key Biomarker', 'Pathognomonic', 'Treatment', 'N'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes.map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.protein_size}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 160, wordBreak: 'break-word' }}>{g.key_biomarker}</td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 200, wordBreak: 'break-word' }}>{g.pathognomonic}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 200, wordBreak: 'break-word' }}>{g.treatment}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.n_patients}</td>
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
  const genes = data.genes ? Object.values(data.genes) : [];
  const g = genes[selected];
  if (!g) return null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 20 }}>
      <div>
        {genes.map((gene, i) => (
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
        <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 12 }}>
          {g.inheritance} · {g.locus} · {g.protein_size} aa · Age of onset: {g.age_of_onset}
        </div>

        <h4 style={{ color: '#e2e8f0', marginBottom: 6 }}>🚨 Critical Flags</h4>
        {(g.critical_flags || []).map((a, i) => (
          <Alert key={i} text={a} color={GENE_COLORS[g.gene]} />
        ))}

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Key Biomarker</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '0.75rem 1rem',
          fontSize: 13, color: '#fbbf24', fontWeight: 600,
        }}>
          {g.key_biomarker}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Pathognomonic</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '0.75rem 1rem',
          fontSize: 13, color: '#f87171',
        }}>
          {g.pathognomonic}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Treatment</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '0.75rem 1rem',
          fontSize: 13, color: '#6ee7b7',
        }}>
          {g.treatment}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Clinical Description</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '1rem',
          fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap',
          maxHeight: 360, overflowY: 'auto',
        }}>
          {g.protein_description}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Cohort ({g.n_patients} patients)</h4>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#22c55e', fontWeight: 700 }}>{g.severity?.mild ?? 0}</span>
            <span style={{ color: '#64748b' }}> mild</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#f59e0b', fontWeight: 700 }}>{g.severity?.moderate ?? 0}</span>
            <span style={{ color: '#64748b' }}> moderate</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#dc2626', fontWeight: 700 }}>{g.severity?.severe ?? 0}</span>
            <span style={{ color: '#64748b' }}> severe</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#6366f1', fontWeight: 700 }}>{g.on_diet}</span>
            <span style={{ color: '#64748b' }}> on diet ({g.on_diet_pct}%)</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#7c3aed', fontWeight: 700 }}>{g.family_cascade}</span>
            <span style={{ color: '#64748b' }}> family cascade</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || {};
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 4 }}>Definitions & Clinical Concepts</h3>
      <p style={{ color: '#64748b', fontSize: 12, marginBottom: 20 }}>
        {data.total_definition_entries} entries · {data.atlas}
      </p>
      {Object.entries(defs).map(([title, body]) => (
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

export default function HeredAminoAcidAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-amino-acid-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-amino-acid-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-amino-acid-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary Amino Acid Atlas — Complete 8-Gene Aminoacidopathy Reference
        </h1>
        <p style={{ color: '#64748b', fontSize: 13 }}>
          PAH (PKU-Most-Common-NBS-Universal-BH4-Sapropterin-Pegvaliase) ·
          CBS (Homocystinuria-B6-50pct-Ectopia-Lentis-Downward-Thromboembolism) ·
          BCKDHA (MSUD-Alloisoleucine-PATHOGNOMONIC-Maple-Syrup-Odour-Mennonite) ·
          FAH (HT1-Succinylacetone-PATHOGNOMONIC-NTBC-HCC-AFP-Surveillance) ·
          HGD (Alkaptonuria-Urine-Darkens-PATHOGNOMONIC-Ochronosis-NTBC-SONIA2) ·
          GLDC (NKH-CSF:Plasma-Glycine->0.08-PATHOGNOMONIC-Burst-Suppression) ·
          OAT (Gyrate-Atrophy-Ornithine->10x-Chorioretinal-B6-5pct) ·
          TAT (HT2-Richner-Hanhart-Keratosis-Triad-Phe-Tyr-Diet-CURATIVE) —
          320 Patients · Seeds 1822–1829
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
