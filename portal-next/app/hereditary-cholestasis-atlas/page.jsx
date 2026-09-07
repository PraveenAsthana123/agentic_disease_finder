'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ATP8B1: '#dc2626',  // red    — PFIC1; FIC1; extrahepatic diarrhea post-LTx worsens
  ABCB11: '#ea580c',  // orange — PFIC2; BSEP; HCC without cirrhosis; odevixibat
  ABCB4:  '#ca8a04',  // amber  — PFIC3/LPAC; MDR3; HIGH GGT; UDCA first-line
  TJP2:   '#16a34a',  // green  — PFIC4; ZO-2; HCC without cirrhosis; neurological
  NR1H4:  '#2563eb',  // blue   — PFIC5; FXR; most severe; FXR agonists CANNOT work
  MYO5B:  '#7c3aed',  // purple — PFIC6/MVID; myosin Vb; 300 mL/kg/day diarrhea
  JAG1:   '#0f766e',  // teal   — Alagille1; butterfly vertebrae PATHOGNOMONIC
  NOTCH2: '#be185d',  // pink   — Alagille2; renal more prominent; Hajdu-Cheney GOF
};

const GENE_DISEASE = {
  ATP8B1: 'AR PFIC1 — ATP8B1-1251aa — 18q21.31 — FIC1-Phospholipid-Flippase — LOW-GGT-PATHOGNOMONIC — Extrahepatic-Diarrhea-Pancreatitis-SNHL — Post-LTx-Diarrhea-WORSENS-NOT-Improves',
  ABCB11: 'AR PFIC2 — ABCB11-1321aa — 2q31.1 — BSEP-Bile-Salt-Export-Pump — LOW-GGT-PATHOGNOMONIC — HCC-WITHOUT-Cirrhosis-MRI+AFP-6-monthly-From-Birth — Odevixibat-FDA2021-E297G-Best-Response',
  ABCB4:  'AR PFIC3 / AD LPAC — ABCB4-1279aa — 7q21.12 — MDR3-Phospholipid-Translocase — HIGH-GGT-PATHOGNOMONIC-Only-PFIC-With-High-GGT — UDCA-First-Line-Most-Effective-All-PFIC — LPAC-Intrahepatic-Stones-Young-Adult',
  TJP2:   'AR PFIC4 — TJP2-1221aa — 9q21.11 — ZO2-Tight-Junction-Scaffold — LOW-GGT — HCC-WITHOUT-Cirrhosis-Surveillance-Mandatory — Neurological-Features-Subset-Extrahepatic-TJP2',
  NR1H4:  'AR PFIC5 — NR1H4-472aa — 12q23.1 — FXR-Farnesoid-X-Receptor-Bile-Acid-Sensor — LOW-GGT-Most-Severe-Neonatal — FXR-Agonists-CANNOT-Work-Receptor-Absent — AFP-Disproportionately-HIGH-Marker',
  MYO5B:  'AR PFIC6/MVID — MYO5B-1852aa — 18q21.1 — Myosin-Vb-Apical-Membrane-Recycling — LOW-GGT — MVID-Neonatal-Secretory-Diarrhea-300mL/kg/day-TPN-Mandatory — Combined-Intestine+Liver-Tx',
  JAG1:   'AD Alagille-Syndrome-1 — JAG1-1218aa — 20p12.2 — NOTCH-Ligand — Bile-Duct-Paucity-PATHOGNOMONIC — Butterfly-Vertebrae-X-Ray-PATHOGNOMONIC — PPS-Most-Common-Cardiac — Odevixibat-FDA2023-ALGS',
  NOTCH2: 'AD Alagille-Syndrome-2 — NOTCH2-2471aa — 1p12 — NOTCH-Receptor-JAG1-Partner — Renal-Anomalies-MORE-Prominent-ALGS1 — Hepatic-MILDER — Hajdu-Cheney-GOF-Acroosteolysis-DISTINCT',
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
  const lv = (text || '').includes('PATHOGNOMONIC') || (text || '').includes('MANDATORY') || (text || '').includes('CANNOT') || (text || '').includes('WORSENS') ? 'critical'
           : (text || '').includes('POSITIVE') || (text || '').includes('NOT') || (text || '').includes('FIRST-LINE') ? 'warning' : 'info';
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
    fetch(`${API}/api/hereditary-cholestasis-atlas/overview`)
      .then(r => r.json()).then(setData).catch(e => setErr(e.message));
  }, []);

  if (err)  return <ErrorBox msg={err} />;
  if (!data) return <Loading />;

  const s = data.aggregate_stats || {};
  return (
    <div>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: '1.2rem' }}>{data.subtitle}</p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
        <KPI label="Total Patients"       value={s.total_patients}        color="#6366f1" />
        <KPI label="Genes Covered"        value={s.genes_covered}         color="#22d3ee" />
        <KPI label="Avg Age at Dx (yr)"   value={s.avg_age_at_diagnosis_yr} color="#a3e635" />
        <KPI label="Severe Cases %"       value={`${s.severe_cases_pct}%`} color="#f97316" />
        <KPI label="Seeds"                value={s.seed_range}            color="#c084fc" />
      </div>

      <h3 style={{ color: '#e2e8f0', marginBottom: '0.8rem' }}>GGT Discriminator (Key Diagnostic Rule)</h3>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
        <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem 1.2rem', borderLeft: '4px solid #dc2626' }}>
          <div style={{ color: '#fca5a5', fontWeight: 700, fontSize: 13 }}>LOW GGT</div>
          <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>PFIC1 (ATP8B1) · PFIC2 (ABCB11) · PFIC4 (TJP2) · PFIC5 (NR1H4) · PFIC6 (MYO5B)</div>
        </div>
        <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem 1.2rem', borderLeft: '4px solid #ca8a04' }}>
          <div style={{ color: '#fcd34d', fontWeight: 700, fontSize: 13 }}>HIGH GGT</div>
          <div style={{ color: '#94a3b8', fontSize: 12, marginTop: 4 }}>PFIC3 (ABCB4) · Alagille JAG1 · Alagille NOTCH2</div>
        </div>
      </div>

      <h3 style={{ color: '#e2e8f0', marginBottom: '0.8rem' }}>Key Clinical Distinctions</h3>
      <div style={{ marginBottom: '1.5rem' }}>
        {(data.key_clinical_distinctions || []).map((d, i) => <Alert key={i} text={d} />)}
      </div>

      <h3 style={{ color: '#e2e8f0', marginBottom: '0.8rem' }}>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 12 }}>
        {(data.gene_summary || []).map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>{g.pathognomonic}</div>
            <div style={{ fontSize: 11, color: '#64748b' }}>{g.n_patients} patients</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── TAB: Gene Table ────────────────────────────────────────────────────────── */
function GeneTableTab() {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-cholestasis-atlas/breakdown`)
      .then(r => r.json()).then(setData).catch(e => setErr(e.message));
  }, []);

  if (err)  return <ErrorBox msg={err} />;
  if (!data) return <Loading />;

  const genes = data.genes || [];
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#0f172a' }}>
            {['Gene','Locus','Size','Inh.','Disease/Syndrome','GGT','Pathognomonic Finding','Treatment'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {genes.map((g, i) => {
            const ggtHigh = g.gene === 'ABCB4' || g.gene === 'JAG1' || g.gene === 'NOTCH2';
            return (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#0f172a' }}>
                <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                <td style={{ padding: '7px 10px', color: '#e2e8f0', maxWidth: 200 }}>{GENE_DISEASE[g.gene] || ''}</td>
                <td style={{ padding: '7px 10px', fontWeight: 700, color: ggtHigh ? '#fcd34d' : '#86efac', whiteSpace: 'nowrap' }}>
                  {ggtHigh ? '↑ HIGH' : '↓ LOW'}
                </td>
                <td style={{ padding: '7px 10px', color: '#fca5a5', maxWidth: 220, fontSize: 11 }}>{g.pathognomonic}</td>
                <td style={{ padding: '7px 10px', color: '#93c5fd', maxWidth: 200, fontSize: 11 }}>{g.treatment}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ─── TAB: Clinical Atlas ────────────────────────────────────────────────────── */
function ClinicalAtlasTab() {
  const [data, setData]       = useState(null);
  const [err, setErr]         = useState(null);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-cholestasis-atlas/breakdown`)
      .then(r => r.json()).then(d => { setData(d); setSelected(d.genes?.[0]?.gene); })
      .catch(e => setErr(e.message));
  }, []);

  if (err)  return <ErrorBox msg={err} />;
  if (!data) return <Loading />;

  const genes = data.genes || [];
  const gene  = genes.find(g => g.gene === selected) || genes[0];

  return (
    <div>
      <div style={{ marginBottom: '1rem', display: 'flex', flexWrap: 'wrap' }}>
        {genes.map(g => (
          <GeneChip key={g.gene} gene={g.gene} selected={g.gene === selected} onClick={() => setSelected(g.gene)} />
        ))}
      </div>

      {gene && (
        <div style={{ background: '#1e293b', borderRadius: 12, padding: '1.2rem', borderTop: `4px solid ${GENE_COLORS[gene.gene] || '#6366f1'}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8, marginBottom: '1rem' }}>
            <div>
              <span style={{ fontSize: 22, fontWeight: 800, color: GENE_COLORS[gene.gene] || '#a5b4fc' }}>{gene.gene}</span>
              <span style={{ fontSize: 13, color: '#64748b', marginLeft: 12 }}>{gene.locus} · {gene.protein_size} · {gene.inheritance}</span>
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>n = {gene.n_patients} patients</div>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>KEY BIOMARKER</div>
            <div style={{ color: '#fcd34d', fontSize: 13 }}>{gene.key_biomarker}</div>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>PATHOGNOMONIC</div>
            <div style={{ color: '#fca5a5', fontSize: 13 }}>{gene.pathognomonic}</div>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>TREATMENT</div>
            <div style={{ color: '#86efac', fontSize: 13 }}>{gene.treatment}</div>
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>CRITICAL FLAGS</div>
            {(gene.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>SEVERITY DISTRIBUTION</div>
            <div style={{ display: 'flex', gap: 8 }}>
              {Object.entries(gene.severity_distribution || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#0f172a', borderRadius: 6, padding: '4px 12px', fontSize: 12, color: '#94a3b8' }}>
                  {k}: <strong style={{ color: '#e2e8f0' }}>{v}</strong>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>SAMPLE PATIENTS (first 5)</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ fontSize: 11, width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    {['ID', 'Age (yr)', 'Sex', 'Severity', 'Locus'].map(h => (
                      <th key={h} style={{ padding: '4px 8px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #334155' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(gene.patients || []).map(p => (
                    <tr key={p.patient_id}>
                      <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                      <td style={{ padding: '4px 8px', color: '#e2e8f0' }}>{p.age_at_diagnosis_yr}</td>
                      <td style={{ padding: '4px 8px', color: '#e2e8f0' }}>{p.sex}</td>
                      <td style={{ padding: '4px 8px', color: p.severity === 'severe' ? '#fca5a5' : p.severity === 'moderate' ? '#fcd34d' : '#86efac' }}>{p.severity}</td>
                      <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{p.locus}</td>
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

/* ─── TAB: Definitions ───────────────────────────────────────────────────────── */
function DefinitionsTab() {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch(`${API}/api/hereditary-cholestasis-atlas/definitions`)
      .then(r => r.json()).then(setData).catch(e => setErr(e.message));
  }, []);

  if (err)  return <ErrorBox msg={err} />;
  if (!data) return <Loading />;

  const q = search.toLowerCase();
  const glossaryEntries = Object.entries(data.glossary || {}).filter(
    ([k, v]) => !q || k.toLowerCase().includes(q) || v.toLowerCase().includes(q)
  );
  const geneEntries = (data.genes || []).filter(
    g => !q || g.gene.toLowerCase().includes(q) || (g.definition || '').toLowerCase().includes(q)
  );

  return (
    <div>
      <input
        placeholder="Search definitions…"
        value={search}
        onChange={e => setSearch(e.target.value)}
        style={{
          width: '100%', boxSizing: 'border-box', padding: '0.6rem 1rem',
          background: '#1e293b', border: '1px solid #334155', borderRadius: 8,
          color: '#e2e8f0', fontSize: 13, marginBottom: '1.2rem',
        }}
      />
      <h3 style={{ color: '#e2e8f0', marginBottom: '0.8rem' }}>Glossary</h3>
      {glossaryEntries.map(([k, v]) => (
        <div key={k} style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem 1rem', marginBottom: 8 }}>
          <div style={{ fontWeight: 700, color: '#a5b4fc', marginBottom: 4 }}>{k}</div>
          <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{v}</div>
        </div>
      ))}
      <h3 style={{ color: '#e2e8f0', marginBottom: '0.8rem', marginTop: '1.5rem' }}>Gene Definitions</h3>
      {geneEntries.map(g => (
        <div key={g.gene} style={{
          background: '#1e293b', borderRadius: 8, padding: '0.8rem 1rem', marginBottom: 8,
          borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
        }}>
          <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', marginBottom: 4 }}>
            {g.gene} — {g.locus} · {g.protein_size} · {g.inheritance} · onset: {g.age_of_onset}
          </div>
          <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.6, marginBottom: 6 }}>{g.definition}</div>
          <div>
            {(g.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ─── Page ──────────────────────────────────────────────────────────────────── */
export default function HereditoryCholestasisAtlasPage() {
  const [tab, setTab] = useState(0);

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'sans-serif', padding: '1.5rem' }}>
      <h1 style={{ fontSize: 22, fontWeight: 800, color: '#a5b4fc', marginBottom: 4 }}>
        Hereditary Cholestasis Atlas
      </h1>
      <p style={{ color: '#64748b', fontSize: 12, marginBottom: '1.2rem' }}>
        Complete 8-Gene Hereditary Cholestatic Liver Disease Atlas —
        ATP8B1 (PFIC1) · ABCB11 (PFIC2) · ABCB4 (PFIC3/LPAC) · TJP2 (PFIC4) ·
        NR1H4 (PFIC5) · MYO5B (PFIC6/MVID) · JAG1 (Alagille-1) · NOTCH2 (Alagille-2) |
        320 patients (8×40, seeds 1894–1901)
      </p>
      <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '6px 18px', borderRadius: 20, border: 'none', cursor: 'pointer',
            background: tab === i ? '#6366f1' : '#1e293b',
            color: tab === i ? '#fff' : '#94a3b8', fontWeight: tab === i ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>
      {tab === 0 && <OverviewTab />}
      {tab === 1 && <GeneTableTab />}
      {tab === 2 && <ClinicalAtlasTab />}
      {tab === 3 && <DefinitionsTab />}
    </div>
  );
}
