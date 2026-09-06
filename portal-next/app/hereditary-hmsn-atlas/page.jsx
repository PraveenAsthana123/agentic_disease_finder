'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PMP22:  '#01579b',  // deep blue — CMT1A most common
  MPZ:    '#1b5e20',  // deep green — CMT1B/CMT2I/J myelin P0
  GJB1:   '#4a148c',  // deep purple — CMTX1 connexin 32
  MFN2:   '#bf360c',  // deep burnt orange — CMT2A2 axonal
  GDAP1:  '#37474f',  // dark slate — CMT4A vocal cord
  SH3TC2: '#880e4f',  // deep magenta — CMT4C scoliosis
  NEFL:   '#006064',  // deep teal — CMT2E giant axon
  EGR2:   '#33691e',  // deep olive — CMT1D/CMT4E hypomyelination
};

const GENE_DISEASE = {
  PMP22:  'CMT1A — 17p12 Duplication (70-80% CMT1) · HNPP Deletion · CMT1E Point Mutation — MLPA MANDATORY',
  MPZ:    'CMT1B AD Severe Demyelinating NCV <20 m/s · CMT2I/J Adult Axonal — Audiogram + Pupil Check',
  GJB1:   'CMTX1 XLD — Most Common X-linked CMT 10-15% — Intermediate NCV 30-40 m/s Males — CNS WM Lesions',
  MFN2:   'CMT2A2 AD — Most Common Axonal CMT 20% CMT2 — Upper Limb > Lower — Optic Atrophy 15% Annual',
  GDAP1:  'CMT4A AR Severe — Vocal Cord Paresis 30% PATHOGNOMONIC — Diaphragm — CMT2K Milder',
  SH3TC2: 'CMT4C AR — Most Common AR CMT — R954W Romani Founder — Scoliosis 60% — Cranial Nerve Palsies',
  NEFL:   'CMT2E AD Axonal Giant Axon — CMT1F Demyelinating Intermediate — Serum NF-L Biomarker — Facial Possible',
  EGR2:   'CMT1D AD Hypomyelination R359W/R381H — CMT4E AR CHN Congenital — NCV <10 m/s — NICU Alert',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|PATHOGNOMONIC|ALERT|NICU/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|MANDATORY|STAT/i.test(text);
  const bg = isCI ? '#b71c1c' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, genes } = data;
  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
      <div style={{ flex: '1 1 340px' }}>
        <h3 style={{ color: '#01579b', marginBottom: 12 }}>Aggregate — 320 Patients (8×40, seeds 1462–1469)</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <tbody>
            {[
              ['Pes Cavus (any gene)', s.pes_cavus_any_gene, '%'],
              ['Distal Lower Limb Weakness', s.distal_weakness_lower_any_gene, '%'],
              ['Demyelinating NCV (any gene)', s.demyelinating_ncv_any_gene, '%'],
              ['Axonal NCV (CMT2/MFN2/NEFL)', s.axonal_ncv_any_gene, '%'],
              ['Intermediate NCV 30-40 m/s (CMTX1)', s.intermediate_ncv_cmtx1, '%'],
              ['Optic Atrophy (MFN2 CMT2A)', s.optic_atrophy_mfn2, '%'],
              ['Vocal Cord Paresis (GDAP1 CMT4A)', s.vocal_cord_paresis_gdap1, '%'],
              ['Scoliosis (any gene)', s.scoliosis_any_gene, '%'],
              ['Cranial Nerve Involvement (SH3TC2)', s.cranial_nerve_involvement, '%'],
              ['Congenital/Infantile Onset (EGR2/GDAP1)', s.congenital_onset_egr2_gdap1, '%'],
              ['Wheelchair Risk (combined)', s.wheelchair_risk_combined, '%'],
            ].map(([label, val, unit]) => (
              <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '6px 8px', color: '#555' }}>{label}</td>
                <td style={{ padding: '6px 8px', fontWeight: 700, color: '#01579b' }}>
                  {val !== undefined && val !== null ? `${val}${unit}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3 style={{ color: '#01579b', marginTop: 20, marginBottom: 10 }}>8 Genes — CMT/HMSN Subtypes</h3>
        {genes && genes.map(g => (
          <div key={g} style={{
            background: GENE_COLORS[g] || '#333',
            color: '#fff', borderRadius: 6, padding: '6px 12px',
            marginBottom: 6, fontSize: 13,
          }}>
            <strong>{g}</strong> — {GENE_DISEASE[g] || g}
          </div>
        ))}
      </div>

      <div style={{ flex: '1 1 340px' }}>
        <h3 style={{ color: '#b71c1c', marginBottom: 10 }}>Critical Clinical Alerts</h3>
        {top_alerts && top_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#01579b', color: '#fff' }}>
            {['Gene', 'Protein', 'Locus', 'aa', 'Inheritance', 'OMIM Gene', 'OMIM Disease', 'N'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {genes.map((g, i) => (
            <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff', borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
              <td style={{ padding: '7px 10px', maxWidth: 220, fontSize: 12 }}>{g.protein}</td>
              <td style={{ padding: '7px 10px', fontFamily: 'monospace' }}>{g.locus}</td>
              <td style={{ padding: '7px 10px' }}>{g.aa}</td>
              <td style={{ padding: '7px 10px', fontSize: 12, maxWidth: 160 }}>{g.inheritance}</td>
              <td style={{ padding: '7px 10px' }}>{g.omim_gene}</td>
              <td style={{ padding: '7px 10px' }}>{g.omim_disease}</td>
              <td style={{ padding: '7px 10px', fontWeight: 700 }}>{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(Object.keys(data)[0]);
  const g = data[selected];

  return (
    <div style={{ display: 'flex', gap: 20 }}>
      <div style={{ minWidth: 140 }}>
        {Object.keys(data).map(gene => (
          <button key={gene} onClick={() => setSelected(gene)} style={{
            display: 'block', width: '100%', textAlign: 'left',
            padding: '8px 12px', marginBottom: 4, borderRadius: 6, cursor: 'pointer',
            border: 'none',
            background: selected === gene ? (GENE_COLORS[gene] || '#01579b') : '#eee',
            color: selected === gene ? '#fff' : '#333',
            fontWeight: selected === gene ? 700 : 400, fontSize: 13,
          }}>{gene}</button>
        ))}
      </div>

      <div style={{ flex: 1 }}>
        <h3 style={{ color: GENE_COLORS[selected] || '#01579b', marginBottom: 4 }}>{g.gene} — {g.protein}</h3>
        <p style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
          {g.locus} · {g.aa} · {g.inheritance}
        </p>
        <p style={{ fontSize: 13, color: '#333', marginBottom: 12, lineHeight: 1.6 }}>{g.gene_class}</p>

        <h4 style={{ color: '#b71c1c', marginBottom: 6 }}>Critical Clinical Alerts</h4>
        {g.critical_alerts && g.critical_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}

        <h4 style={{ color: '#1565c0', marginTop: 14, marginBottom: 6 }}>DDx Rules</h4>
        {g.key_ddx_rules && g.key_ddx_rules.map((r, i) => (
          <div key={i} style={{
            background: '#e3f2fd', borderLeft: '4px solid #1565c0',
            padding: '6px 10px', marginBottom: 6, fontSize: 13, borderRadius: '0 4px 4px 0',
          }}>{r}</div>
        ))}

        <h4 style={{ color: '#555', marginTop: 14, marginBottom: 6 }}>Phenotype Rates (%)</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <tbody>
            {g.phenotype_rates && Object.entries(g.phenotype_rates).map(([k, v]) => (
              <tr key={k} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '4px 8px', color: '#555', textTransform: 'capitalize' }}>
                  {k.replace(/_/g, ' ')}
                </td>
                <td style={{ padding: '4px 8px', fontWeight: 700, color: '#01579b' }}>{v}%</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h4 style={{ color: '#555', marginTop: 14, marginBottom: 6 }}>Etiology Distribution</h4>
        {g.etiologies && g.etiologies.map((e, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
            <div style={{
              width: `${e.pct * 2.5}px`, minWidth: 4, height: 18,
              background: GENE_COLORS[selected] || '#01579b', borderRadius: 3, marginRight: 8,
            }} />
            <span style={{ fontSize: 12 }}>{e.label} — <strong>{e.pct}%</strong></span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || [];
  return (
    <div>
      {defs.map((d, i) => (
        <div key={i} style={{
          borderLeft: '4px solid #01579b', padding: '12px 16px',
          marginBottom: 16, background: '#f8f9fa', borderRadius: '0 6px 6px 0',
        }}>
          <h4 style={{ color: '#01579b', marginBottom: 6, fontSize: 15 }}>{d.term}</h4>
          <p style={{ fontSize: 13, color: '#333', lineHeight: 1.6, margin: 0 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryHMSNAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-hmsn-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
    fetch(`${API}/api/hereditary-hmsn-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/hereditary-hmsn-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto', padding: '24px 16px' }}>
      <h1 style={{ color: '#01579b', marginBottom: 4, fontSize: 22 }}>
        🧬 Hereditary HMSN Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 20, fontSize: 14 }}>
        Complete 8-Gene Hereditary Motor and Sensory Neuropathy (CMT/HMSN) Atlas ·
        PMP22-160aa-17p12-AD-CMT1A-Duplication-MLPA-Mandatory-HNPP-Deletion ·
        MPZ-248aa-1q23.3-AD-CMT1B-NCV-Less-20ms-CMT2IJ-Adult-Axonal ·
        GJB1-283aa-Xq13.1-XLD-CMTX1-Intermediate-NCV-CNS-White-Matter ·
        MFN2-741aa-1p36.22-AD-CMT2A2-Upper-Limb-Optic-Atrophy-15pct ·
        GDAP1-358aa-8q21.11-AR-CMT4A-Vocal-Cord-Paresis-Pathognomonic ·
        SH3TC2-1288aa-5q32-AR-CMT4C-Scoliosis-60pct-Cranial-Nerve ·
        NEFL-543aa-8p21.2-AD-CMT2E-Giant-Axon-NF-L-Biomarker ·
        EGR2-472aa-10q21.2-AD-CMT1D-Hypomyelination-R359W-NICU ·
        320 patients (8×40, seeds 1462–1469)
      </p>

      {error && (
        <div style={{ background: '#ffebee', border: '1px solid #ef9a9a', borderRadius: 6, padding: 12, marginBottom: 16, color: '#b71c1c' }}>
          Backend error: {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 6, cursor: 'pointer', border: 'none',
            background: tab === i ? '#01579b' : '#eee',
            color: tab === i ? '#fff' : '#333',
            fontWeight: tab === i ? 700 : 400, fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <GeneTableTab data={breakdown} />}
      {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
