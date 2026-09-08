'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-polyglutamine-sca-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ATXN1:    '#4a148c',  // deep purple    — SCA1 hyperreflexia fastest progression
  ATXN2:    '#1a237e',  // deep indigo    — SCA2 slow saccades ALS modifier
  ATXN3:    '#006064',  // deep teal      — SCA3/MJD most common exophthalmos
  ATXN7:    '#880e4f',  // deep rose      — SCA7 retinal degeneration
  ATXN10:   '#1b5e20',  // deep green     — SCA10 seizures ATTCT repeat
  ATXN8OS:  '#37474f',  // dark slate     — SCA8 incomplete penetrance tremor
  PPP2R2B:  '#e65100',  // deep orange    — SCA12 tremor-dominant Indian
  KCNC3:    '#bf360c',  // deep brown-red — SCA13 childhood onset hypoplasia
};

const GENE_INFO = {
  ATXN1:   { full: 'ATXN1 / Ataxin-1 816aa',          locus: '6p22.3',   size: '816 aa',          inh: 'AD',  disease: 'SCA1 — CAG≥39 / Hyperreflexia PATHOGNOMONIC / Fastest Progression / Brainstem Atrophy' },
  ATXN2:   { full: 'ATXN2 / Ataxin-2 1312aa',         locus: '12q24.12', size: '1312 aa',          inh: 'AD',  disease: 'SCA2 — Slow Saccades PATHOGNOMONIC / ALS Modifier 27-33 CAG / Hyporeflexia / Cuban Founder' },
  ATXN3:   { full: 'ATXN3 / Ataxin-3 361aa',          locus: '14q32.12', size: '361 aa',           inh: 'AD',  disease: 'SCA3/MJD — Most Common SCA Worldwide 28% / CAG≥60 / Exophthalmos PATHOGNOMONIC / Azorean' },
  ATXN7:   { full: 'ATXN7 / Ataxin-7 892aa',          locus: '3p14.1',   size: '892 aa',           inh: 'AD',  disease: 'SCA7 — Retinal Degeneration PATHOGNOMONIC / Annual Fundoscopy / Extreme Anticipation' },
  ATXN10:  { full: 'ATXN10 / Ataxin-10 475aa',        locus: '22q13.31', size: '475 aa',           inh: 'AD',  disease: 'SCA10 — ATTCT Pentanucleotide / Seizures 50% / Standard Panels Miss / Mexican-Brazilian' },
  ATXN8OS: { full: 'ATXN8OS / ATXN8 CTG/CAG',         locus: '13q21.33', size: 'CTG/CAG repeat',   inh: 'AD',  disease: 'SCA8 — Incomplete Penetrance 30% / Tremor Prominent / Bidirectional Assay Required' },
  PPP2R2B: { full: "PPP2R2B / PP2A-Bβ 443aa",         locus: '5q32',     size: '443 aa',           inh: 'AD',  disease: "SCA12 — Tremor-Dominant Misdiagnosed ET / Indian Founder / 5'-UTR CAG / Standard Panels Miss" },
  KCNC3:   { full: 'KCNC3 / Kv3.3 735aa',            locus: '19q13.33', size: '735 aa',           inh: 'AD',  disease: 'SCA13 — R420H Childhood+ID / F448L Adult / Cerebellar Hypoplasia NOT Atrophy / Two Phenotypes' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('LEVEL-B') || flag.includes('LEVEL-A') ? '#1565c0'
    : flag.includes('MISS') || flag.includes('MISSES') ? '#880e4f'
    : flag.includes('CONTRAINDICATED') || flag.includes('CI') || flag.includes('AVOID') ? '#880e4f'
    : flag.includes('EMERGENCY') || flag.includes('FATAL') ? '#e65100'
    : flag.includes('TREATMENT') || flag.includes('RESPONDER') || flag.includes('BENEFIT') ? '#2e7d32'
    : flag.includes('FOUNDER') || flag.includes('ANCESTRY') ? '#00695c'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryPolyglutamineSCAAtlasPage() {
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
        🧬 Hereditary Polyglutamine SCA Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Polyglutamine &amp; Repeat-Expansion Spinocerebellar Ataxia Atlas —
        ATXN1 (SCA1) · ATXN2 (SCA2/ALS) · ATXN3 (SCA3/MJD) · ATXN7 (SCA7/Retinal) · ATXN10 (SCA10/Seizures) · ATXN8OS (SCA8) · PPP2R2B (SCA12) · KCNC3 (SCA13)
        &nbsp;|&nbsp; 320 patients · seeds 2102-2109
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
              { label: 'ATXN1 Hyperreflexia (SCA1)', value: overview.atxn1_hyperreflexia_patients, color: '#4a148c' },
              { label: 'ATXN2 Slow Saccades (SCA2)', value: overview.atxn2_slow_saccades_patients, color: '#1a237e' },
              { label: 'ATXN3 Exophthalmos (SCA3)', value: overview.atxn3_exophthalmos_patients, color: '#006064' },
              { label: 'ATXN7 Retinal Degen. (SCA7)', value: overview.atxn7_retinal_degeneration_patients, color: '#880e4f' },
              { label: 'ATXN10 Seizures (SCA10)', value: overview.atxn10_seizure_patients, color: '#1b5e20' },
              { label: 'SCA8 Incomplete Penetrance', value: overview.sca8_incomplete_penetrance_patients, color: '#37474f' },
              { label: 'PPP2R2B Action Tremor (SCA12)', value: overview.ppp2r2b_action_tremor_patients, color: '#e65100' },
              { label: 'KCNC3 Childhood Onset (SCA13)', value: overview.kcnc3_childhood_onset_patients, color: '#bf360c' },
              { label: 'Standard Panels Missed', value: overview.standard_panels_missed_diagnosis_patients, color: '#b71c1c' },
              { label: 'PPP2R2B Misdiagnosed as ET', value: overview.ppp2r2b_misdiagnosed_et_patients, color: '#e65100' },
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
            <h3 style={{ color: '#c62828', margin: '0 0 10px' }}>⚠️ Key Diagnostic Alerts — Panels Miss / Misdiagnosis Traps</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>ATXN10/SCA10:</strong> ATTCT pentanucleotide repeat missed by ALL standard panels/WES — request ATTCT-specific PCR explicitly; seizures + ataxia + Amerindian ancestry = test SCA10</li>
              <li><strong>PPP2R2B/SCA12:</strong> 5'-UTR non-coding CAG expansion missed by standard exome — request PPP2R2B 5'-UTR assay; tremor in Indian patient misdiagnosed as ET for 10-20yr before ataxia</li>
              <li><strong>ATXN8OS/SCA8:</strong> Incomplete penetrance (~30%) — positive test does NOT confirm diagnosis alone; bidirectional CTG/CAG assay required</li>
              <li><strong>ATXN2/SCA2:</strong> Intermediate alleles 27-33 CAG = ALS modifier (not SCA2) — family members with ALS should be tested for ATXN2 intermediate repeats</li>
              <li><strong>ATXN3/SCA3:</strong> Most common SCA worldwide (28%) — if exophthalmos + ataxia in any ancestry, test ATXN3 first</li>
            </ul>
          </div>

          {/* Pathognomonic Finder */}
          <div style={{ background: '#e8f5e9', border: '2px solid #2e7d32', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#2e7d32', margin: '0 0 10px' }}>🎯 Pathognomonic Bedside Signs — Gene Finder</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>Hyperreflexia brisk early:</strong> SCA1 (ATXN1) — only common SCA with early hyperreflexia</li>
              <li><strong>Slow/hypometric saccades:</strong> SCA2 (ATXN2) — oculomotor exam mandatory</li>
              <li><strong>Exophthalmos + facial fasciculations:</strong> SCA3/MJD (ATXN3) — most common SCA worldwide</li>
              <li><strong>Retinal degeneration + colour vision loss:</strong> SCA7 (ATXN7) — annual fundoscopy mandatory</li>
              <li><strong>Ataxia + seizures (Amerindian ancestry):</strong> SCA10 (ATXN10) — ATTCT repeat PCR</li>
              <li><strong>Action + head tremor decades before ataxia (Indian):</strong> SCA12 (PPP2R2B) — ET misdiagnosis</li>
              <li><strong>Childhood ataxia + ID + cerebellar hypoplasia:</strong> SCA13 R420H (KCNC3)</li>
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
