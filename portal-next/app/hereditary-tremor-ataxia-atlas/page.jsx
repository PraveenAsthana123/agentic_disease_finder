'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-tremor-ataxia-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FMR1:       '#4a148c',  // deep purple    — FXTAS MCP sign
  CACNA1A:    '#1a237e',  // deep indigo    — EA2/SCA6 acetazolamide
  FGF14:      '#006064',  // deep teal      — SCA27B 4-AP
  RFC1:       '#1b5e20',  // deep green     — CANVAS triad
  NOTCH2NLC:  '#37474f',  // dark slate     — NIID skin biopsy
  PRKCG:      '#e65100',  // deep orange    — SCA14 action tremor
  ITPR1:      '#880e4f',  // deep pink      — SCA15 pure cerebellar
  ELOVL5:     '#bf360c',  // deep brown-red — SCA38 DHA deficiency
};

const GENE_INFO = {
  FMR1:      { full: 'FMR1 / FMRP CGG-repeat Xq27.3',     locus: 'Xq27.3',   size: 'CGG repeat', inh: 'XL',  disease: 'FXTAS — Premutation 55-200 CGG / MCP Hyperintensity PATHOGNOMONIC / Late-Onset Males' },
  CACNA1A:   { full: 'CACNA1A / Cav2.1 2505aa',            locus: '19p13.13', size: '2505 aa',   inh: 'AD',  disease: 'EA2 (Episodic Ataxia) + SCA6 (CAG≥20) — Interictal Nystagmus PATHOGNOMONIC / Acetazolamide Level B' },
  FGF14:     { full: 'FGF14 / iFGF14 252aa',               locus: '13q33.1',  size: '252 aa',    inh: 'AD',  disease: 'SCA27B — GAA-TTC >250 / Most Common Late-Onset Ataxia / Standard Panels Miss / 4-AP Level B' },
  RFC1:      { full: 'RFC1 / RFC1 1148aa',                  locus: '4p14',     size: '1148 aa',   inh: 'AR',  disease: 'CANVAS — AAGGG Biallelic / Sensory Neuropathy + Vestibular Areflexia TRIAD / Chronic Cough 70%' },
  NOTCH2NLC: { full: 'NOTCH2NLC / GGC-repeat 1q22',        locus: '1q22',     size: 'GGC repeat', inh: 'AD',  disease: 'NIID — Skin Biopsy p62+ Intranuclear Inclusions PATHOGNOMONIC / DWI Cortico-Medullary Hyperintensity' },
  PRKCG:     { full: 'PRKCG / PKCγ 697aa',                 locus: '19q13.42', size: '697 aa',    inh: 'AD',  disease: 'SCA14 — Action Tremor PROMINENT / Slow Progression / Cognitive Preserved / Onset 8-42yr' },
  ITPR1:     { full: 'ITPR1 / IP3R1 2695aa',               locus: '3p26.1',   size: '2695 aa',   inh: 'AD',  disease: 'SCA15 — Pure Cerebellar / Deletions 35-45% MLPA Mandatory / Very Slow Progression' },
  ELOVL5:    { full: 'ELOVL5 / Elongase-5 299aa',          locus: '6p12.3',   size: '299 aa',    inh: 'AD',  disease: 'SCA38 — DHA Deficiency PATHOGNOMONIC Metabolic / Dietary DHA 1g/day Treatment / Pes Cavus' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('LEVEL-B') || flag.includes('LEVEL-A') ? '#1565c0'
    : flag.includes('MISS') || flag.includes('MISSES') ? '#880e4f'
    : flag.includes('CONTRAINDICATED') || flag.includes('CI') || flag.includes('AVOID') ? '#880e4f'
    : flag.includes('EMERGENCY') || flag.includes('FATAL') ? '#e65100'
    : flag.includes('TREATMENT') || flag.includes('RESPONDER') || flag.includes('BENEFIT') ? '#2e7d32'
    : flag.includes('TRIAD') || flag.includes('SPECIFIC') ? '#00695c'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryTremorAtaxiaAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, def]) => { setOverview(ov); setBreakdown(br); setDefinitions(def); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ color: '#4a148c', marginBottom: 4 }}>
        🌀 Hereditary Tremor-Ataxia Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Hereditary Tremor &amp; Episodic/Progressive Ataxia Atlas — FMR1 (FXTAS) · CACNA1A (EA2/SCA6) · FGF14 (SCA27B) · RFC1 (CANVAS) · NOTCH2NLC (NIID) · PRKCG (SCA14) · ITPR1 (SCA15) · ELOVL5 (SCA38)
        &nbsp;|&nbsp; 320 patients · seeds 2094-2101
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: activeTab === t ? '#4a148c' : '#ede7f6',
            color: activeTab === t ? '#fff' : '#4a148c', fontWeight: activeTab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading…</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* OVERVIEW TAB */}
      {activeTab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 14, marginBottom: 28 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#4a148c' },
              { label: 'FMR1 MCP Sign (FXTAS)', value: overview.fmr1_mcp_hyperintensity_patients, color: '#4a148c' },
              { label: 'CACNA1A Acetazolamide Resp.', value: overview.cacna1a_acetazolamide_responders, color: '#1a237e' },
              { label: 'FGF14 4-AP Responders', value: overview.fgf14_four_ap_responders, color: '#006064' },
              { label: 'RFC1 Chronic Cough (CANVAS)', value: overview.rfc1_canvas_chronic_cough_patients, color: '#1b5e20' },
              { label: 'RFC1 Vestibular Areflexia', value: overview.rfc1_vestibular_areflexia_patients, color: '#2e7d32' },
              { label: 'NOTCH2NLC Skin Biopsy +', value: overview.notch2nlc_skin_biopsy_inclusions, color: '#37474f' },
              { label: 'PRKCG Action Tremor', value: overview.prkcg_action_tremor_patients, color: '#e65100' },
              { label: 'ITPR1 Deletion Alleles', value: overview.itpr1_deletion_patients, color: '#880e4f' },
              { label: 'ELOVL5 Low Serum DHA', value: overview.elovl5_low_serum_dha_patients, color: '#bf360c' },
              { label: 'Standard Panels Missed', value: overview.standard_panels_missed_diagnosis_patients, color: '#b71c1c' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                background: '#fff', border: `2px solid ${color}`, borderRadius: 10,
                padding: '14px 16px', textAlign: 'center',
              }}>
                <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
                <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Key Clinical Alerts */}
          <div style={{ background: '#fce4ec', border: '2px solid #c62828', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#c62828', margin: '0 0 10px' }}>⚠️ Key Diagnostic Alerts (Standard Panels Miss These)</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>FGF14/SCA27B:</strong> Most common late-onset ataxia — GAA-TTC repeat missed by standard exome/panels — request long-read PCR or repeat-primed PCR explicitly</li>
              <li><strong>RFC1/CANVAS:</strong> AAGGG pentanucleotide repeat missed by standard sequencing — request repeat-primed PCR; don't miss the chronic cough clue (&gt;70%)</li>
              <li><strong>NOTCH2NLC/NIID:</strong> GGC repeat missed by standard sequencing — skin biopsy (p62+) is pathognomonic and ante-mortem accessible</li>
              <li><strong>ITPR1/SCA15:</strong> 35-45% alleles are deletions — MLPA mandatory alongside sequencing; standard panels miss deletions</li>
              <li><strong>FMR1/FXTAS:</strong> Premutation 55-200 CGG — request specific FMR1 repeat assay not standard array; confirm sex (males &gt;50yr predominantly affected)</li>
            </ul>
          </div>

          {/* Treatment Highlights */}
          <div style={{ background: '#e8f5e9', border: '2px solid #2e7d32', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#2e7d32', margin: '0 0 10px' }}>💊 Treatment-Actionable Diagnoses</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>CACNA1A/EA2:</strong> Acetazolamide Level B — 75-90% attack reduction; 4-AP alternative</li>
              <li><strong>FGF14/SCA27B:</strong> 4-Aminopyridine (4-AP) specific treatment — Level B; monitor QTc</li>
              <li><strong>ELOVL5/SCA38:</strong> Dietary DHA supplementation 1 g/day — clinical trial evidence; check serum DHA at 3 months</li>
              <li><strong>NOTCH2NLC/NIID:</strong> Levodopa trial for parkinsonism component (partial response)</li>
              <li><strong>RFC1/CANVAS:</strong> Vestibular rehabilitation; cough management (ACE-I cessation)</li>
            </ul>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {activeTab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#4a148c', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inh.', 'Disease / Syndrome', 'Patients', 'Key Biomarker'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, gd], idx) => {
                const gi = GENE_INFO[gene] || {};
                return (
                  <tr key={gene} style={{ background: idx % 2 === 0 ? '#f3e5f5' : '#fff' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                    <td style={{ padding: '10px 12px' }}>{gd.locus}</td>
                    <td style={{ padding: '10px 12px', whiteSpace: 'nowrap' }}>{gd.protein_size}</td>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{gi.inh || '?'}</td>
                    <td style={{ padding: '10px 12px', maxWidth: 280 }}><small>{gi.disease || '—'}</small></td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gd.patient_count}</td>
                    <td style={{ padding: '10px 12px', maxWidth: 220 }}><small style={{ color: '#555' }}>{gd.key_biomarker}</small></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {activeTab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(520px, 1fr))', gap: 20 }}>
          {Object.entries(breakdown).map(([gene, gd]) => (
            <div key={gene} style={{
              border: `2px solid ${GENE_COLORS[gene] || '#888'}`, borderRadius: 10, padding: 18,
              background: '#fafafa',
            }}>
              <h3 style={{ color: GENE_COLORS[gene] || '#333', margin: '0 0 4px' }}>{gene}</h3>
              <div style={{ fontSize: 12, color: '#777', marginBottom: 8 }}>
                {gd.locus} · {gd.protein_size} · {gd.inheritance?.split(';')[0]}
              </div>
              <div style={{ marginBottom: 8 }}>
                <strong style={{ fontSize: 12, color: '#555' }}>Pathognomonic:</strong>
                <p style={{ margin: '4px 0', fontSize: 13 }}>{gd.pathognomonic?.split(';')[0]}</p>
              </div>
              <div style={{ marginBottom: 8 }}>
                <strong style={{ fontSize: 12, color: '#555' }}>Treatment:</strong>
                <p style={{ margin: '4px 0', fontSize: 13 }}>{gd.treatment?.split(';')[0]}</p>
              </div>
              <div style={{ marginBottom: 8 }}>
                <strong style={{ fontSize: 12, color: '#555' }}>Onset:</strong>
                <span style={{ fontSize: 13, marginLeft: 6 }}>{gd.age_of_onset}</span>
              </div>
              <div style={{ marginTop: 10, borderTop: '1px solid #ddd', paddingTop: 8 }}>
                {(gd.critical_flags || []).map(f => <FLAG_BADGE key={f} flag={f} />)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <h2 style={{ color: '#4a148c', marginBottom: 12 }}>Gene Definitions</h2>
          {Object.entries(definitions.genes || {}).map(([gene, def]) => (
            <details key={gene} style={{ marginBottom: 10, border: `1px solid ${GENE_COLORS[gene] || '#ccc'}`, borderRadius: 6 }}>
              <summary style={{
                padding: '10px 14px', cursor: 'pointer', fontWeight: 700,
                color: GENE_COLORS[gene] || '#333', background: '#f5f5f5',
              }}>{gene}</summary>
              <div style={{ padding: '12px 14px', fontSize: 13, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{def}</div>
            </details>
          ))}

          <h2 style={{ color: '#4a148c', margin: '24px 0 12px' }}>Clinical Glossary</h2>
          {Object.entries(definitions.glossary || {}).map(([term, explanation]) => (
            <details key={term} style={{ marginBottom: 8, border: '1px solid #ce93d8', borderRadius: 6 }}>
              <summary style={{ padding: '10px 14px', cursor: 'pointer', fontWeight: 600, background: '#f3e5f5', color: '#4a148c' }}>{term}</summary>
              <div style={{ padding: '12px 14px', fontSize: 13, lineHeight: 1.7 }}>{explanation}</div>
            </details>
          ))}

          <h2 style={{ color: '#4a148c', margin: '24px 0 12px' }}>Surveillance Protocols</h2>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, protocol]) => (
            <details key={gene} style={{ marginBottom: 8, border: `1px solid ${GENE_COLORS[gene.split(' ')[0]] || '#ccc'}`, borderRadius: 6 }}>
              <summary style={{
                padding: '10px 14px', cursor: 'pointer', fontWeight: 600,
                background: '#fce4ec', color: '#c62828',
              }}>{gene}</summary>
              <div style={{ padding: '12px 14px', fontSize: 13, lineHeight: 1.7 }}>{protocol}</div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
