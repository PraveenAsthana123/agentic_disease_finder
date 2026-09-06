'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  NOTCH3: '#0d47a1',  // deep blue — CADASIL most common
  HTRA1:  '#1b5e20',  // deep green — CARASIL/HTRA1-AD
  COL4A1: '#bf360c',  // deep burnt orange — COL4A1 angiopathy
  COL4A2: '#e65100',  // deep orange — COL4A2 angiopathy
  TREX1:  '#4a148c',  // deep purple — RVCL-S
  GLA:    '#006064',  // deep teal — Fabry treatable
  ADA2:   '#880e4f',  // deep magenta — DADA2
  CST3:   '#37474f',  // dark slate — HCCAA
};

const GENE_DISEASE = {
  NOTCH3: 'CADASIL AD — Most Common Hereditary SVD — EGF-Domain Cysteine Rule — GOM Skin Biopsy — NO tPA — NO OCP',
  HTRA1:  'CARASIL AR — Alopecia + Spondylosis PATHOGNOMONIC + Young Stroke No-RF · HTRA1-AD Heterozygous Milder',
  COL4A1: 'COL4A1 Angiopathy AD — Porencephaly = COL4A1/COL4A2 Until Proven — NO Anticoagulants — Test COL4A2 Simultaneously',
  COL4A2: 'COL4A2 Angiopathy AD — Same 13q34 Locus as COL4A1 — Test BOTH Simultaneously — ICH + Porencephaly',
  TREX1:  'RVCL-S AD — Retinal Vasculopathy ALWAYS First — Punctate Gd-Enhancing Mass Lesions PATHOGNOMONIC — Not Tumour',
  GLA:    'Fabry XLD — MOST COMMON TREATABLE Hereditary SVD — Posterior Lacunar Stroke — Pulvinar Sign — ERT + Migalastat',
  ADA2:   'DADA2 AR — Livedoid Rash + Mixed Strokes (Ischaemic + Haemorrhagic) — ADA2 Enzyme Assay — TNF Blockade CURATIVE',
  CST3:   'HCCAA AD — L68Q Icelandic — Plasma Cystatin C LOW (Paradoxical) — ICH — NO Anticoagulants',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|PATHOGNOMONIC|ALERT|CURATIVE|NO-tPA|NO-OCP|NO-ANTICOAG/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|MANDATORY|STAT|SIMULTANEOUSLY|FIRST/i.test(text);
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
        <h3 style={{ color: '#0d47a1', marginBottom: 12 }}>Aggregate — 320 Patients (8×40, seeds 1470–1477)</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <tbody>
            {[
              ['NOTCH3 — GOM Confirmed (skin biopsy)', s.notch3_gom_confirmed_pct, '%'],
              ['NOTCH3 — Migraine with Aura', s.notch3_migraine_aura_pct, '%'],
              ['NOTCH3 — Anterior Temporal WML', s.notch3_anterior_temporal_wml_pct, '%'],
              ['NOTCH3 — tPA Contraindicated (ALL)', s.notch3_tpa_contraindicated_pct, '%'],
              ['HTRA1 — Premature Alopecia', s.htra1_alopecia_pct, '%'],
              ['HTRA1 — Lumbar Spondylosis', s.htra1_spondylosis_pct, '%'],
              ['HTRA1 — GOM Absent (distinguishes CARASIL)', s.htra1_gom_absent_pct, '%'],
              ['COL4A1 — Porencephaly', s.col4a1_porencephaly_pct, '%'],
              ['COL4A1 — Retinal Arterial Tortuosity', s.col4a1_retinal_tortuosity_pct, '%'],
              ['COL4A1/COL4A2 — No Anticoagulants (ALL)', s.col4a1_anticoag_contraindicated_pct, '%'],
              ['TREX1 — Retinal FA Confirmed', s.trex1_retinal_fa_confirmed_pct, '%'],
              ['TREX1 — Gadolinium Mass Lesions', s.trex1_gadolinium_mass_lesions_pct, '%'],
              ['GLA — Posterior Circulation Stroke', s.gla_posterior_stroke_pct, '%'],
              ['GLA — ERT Started', s.gla_ert_started_pct, '%'],
              ['GLA — Lyso-Gb3 Elevated', s.gla_lyso_gb3_elevated_pct, '%'],
              ['ADA2 — Livedo Racemosa', s.ada2_livedo_racemosa_pct, '%'],
              ['ADA2 — Mixed (Ischaemic + Haemorrhagic) Strokes', s.ada2_mixed_strokes_pct, '%'],
              ['ADA2 — TNF Blockade Started', s.ada2_tnf_blockade_pct, '%'],
              ['CST3 — ICH Event', s.cst3_ich_pct, '%'],
              ['CST3 — Plasma Cystatin C Low (Paradoxical)', s.cst3_low_cystatin_c_pct, '%'],
              ['Any Gene — Anticoagulants Contraindicated', s.anticoag_contraindicated_any_pct, '%'],
            ].map(([label, val, unit]) => (
              <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '6px 8px', color: '#555' }}>{label}</td>
                <td style={{ padding: '6px 8px', fontWeight: 700, color: '#0d47a1' }}>
                  {val !== undefined && val !== null ? `${val}${unit}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3 style={{ color: '#0d47a1', marginTop: 20, marginBottom: 10 }}>8 Genes — Hereditary Cerebral SVD</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {(genes || []).map(g => (
            <div key={g.gene} style={{
              background: GENE_COLORS[g.gene] || '#333',
              color: '#fff', borderRadius: 8, padding: '8px 14px',
              fontSize: 13, minWidth: 140,
            }}>
              <strong>{g.gene}</strong><br />
              <span style={{ opacity: 0.85 }}>{g.locus} · {g.inheritance?.split('—')[0]?.trim()}</span><br />
              <span style={{ opacity: 0.75, fontSize: 11 }}>{g.aa} · n={g.n_patients}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ flex: '1 1 320px' }}>
        <h3 style={{ color: '#b71c1c', marginBottom: 10 }}>Clinical Alerts</h3>
        {(top_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;
  const genes = Object.values(breakdown || {});
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#0d47a1', color: '#fff' }}>
            {['Gene', 'Protein', 'Locus', 'aa', 'Inheritance', 'Disease', 'N', 'Top Alert'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {genes.map((g, idx) => (
            <tr key={g.gene} style={{ background: idx % 2 === 0 ? '#f8f9fa' : '#fff', verticalAlign: 'top' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
              <td style={{ padding: '7px 10px', maxWidth: 180 }}>{g.protein?.split('—')[0]?.trim()}</td>
              <td style={{ padding: '7px 10px', whiteSpace: 'nowrap' }}>{g.locus}</td>
              <td style={{ padding: '7px 10px', whiteSpace: 'nowrap' }}>{g.aa}</td>
              <td style={{ padding: '7px 10px', maxWidth: 160 }}>{g.inheritance}</td>
              <td style={{ padding: '7px 10px', maxWidth: 220, fontSize: 12 }}>{GENE_DISEASE[g.gene]}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{g.n_patients}</td>
              <td style={{ padding: '7px 10px', maxWidth: 260, fontSize: 12, color: '#b71c1c' }}>
                {(g.key_alerts || [])[0] || ''}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;
  const genes = Object.values(breakdown || {});
  return (
    <div>
      {genes.map(g => (
        <div key={g.gene} style={{
          marginBottom: 28, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#333'}`,
          paddingLeft: 16,
        }}>
          <h3 style={{ color: GENE_COLORS[g.gene] || '#333', marginBottom: 4 }}>
            {g.gene} — {g.locus} — {g.inheritance}
          </h3>
          <p style={{ fontSize: 13, color: '#444', marginBottom: 8 }}>{g.gene_class}</p>
          <h4 style={{ marginBottom: 6, fontSize: 14 }}>Clinical Alerts</h4>
          {(g.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
          <h4 style={{ marginTop: 10, marginBottom: 6, fontSize: 14 }}>Etiology Distribution (n={g.n_patients})</h4>
          <table style={{ borderCollapse: 'collapse', fontSize: 13, width: '100%', maxWidth: 600 }}>
            <tbody>
              {Object.entries(g.etiologies || {}).map(([etio, count]) => (
                <tr key={etio} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '4px 8px', color: '#555' }}>{etio}</td>
                  <td style={{ padding: '4px 8px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { definitions } = data;
  return (
    <div>
      {(definitions || []).map((def, i) => (
        <div key={i} style={{
          marginBottom: 18, padding: '12px 16px',
          background: i % 2 === 0 ? '#e8eaf6' : '#f3f4f6',
          borderRadius: 8,
        }}>
          <strong style={{ color: '#0d47a1', fontSize: 15 }}>{def.term}</strong>
          <p style={{ margin: '6px 0 0', fontSize: 13, color: '#333', lineHeight: 1.6 }}>{def.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditoryCerebralSVDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-cerebral-svd-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(console.error);
    fetch(`${API}/api/hereditary-cerebral-svd-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(console.error);
    fetch(`${API}/api/hereditary-cerebral-svd-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(console.error);
  }, []);

  return (
    <div style={{ padding: '1.5rem', maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ color: '#0d47a1', fontSize: 22, marginBottom: 4 }}>
        Hereditary-Cerebral-SVD-Atlas — Complete 8-Gene Hereditary Cerebral Small Vessel Disease Atlas
      </h1>
      <p style={{ color: '#555', fontSize: 13, marginBottom: 20 }}>
        NOTCH3 (CADASIL) · HTRA1 (CARASIL/HTRA1-AD) · COL4A1 · COL4A2 · TREX1 (RVCL-S) · GLA (Fabry) ·
        ADA2 (DADA2) · CST3 (HCCAA) — 320 patients (8×40, seeds 1470–1477)
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#0d47a1' : '#e3e8f0',
            color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400,
            fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Overview'      && <OverviewTab      data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab     data={breakdown} />}
      {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
