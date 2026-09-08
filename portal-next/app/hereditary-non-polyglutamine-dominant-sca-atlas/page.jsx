'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-non-polyglutamine-dominant-sca-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SPTBN2:  '#1a237e',  // deep indigo   — SCA5 Lincoln family pure cerebellar
  CACNA1G: '#00695c',  // deep teal     — SCA42 French Cav3.1 T-type
  KCND3:   '#4a148c',  // deep purple   — SCA19/22 cognitive+myoclonus+tremor
  TMEM240: '#880e4f',  // deep rose     — SCA21 childhood cognitive PATHOGNOMONIC
  STUB1:   '#b71c1c',  // deep red      — SCA48 CHIP cognitive>motor FTD-mimic
  GRM1:    '#e65100',  // deep orange   — SCA44 action tremor before ataxia
  FAT2:    '#2e7d32',  // deep green    — SCA45 protocadherin pure cerebellar
  PUM1:    '#37474f',  // dark slate    — SCA47 childhood ID + cerebellar hypoplasia
};

const GENE_INFO = {
  SPTBN2:  { full: 'SPTBN2 / Spectrin-βIII 2390aa', locus: '11q13.2', size: '2390 aa', inh: 'AD', disease: 'SCA5 — Lincoln Family / Pure Cerebellar / Very Slow Progression / EAAT4 Stabilisation' },
  CACNA1G: { full: 'CACNA1G / Cav3.1 2107aa',       locus: '17q21.33', size: '2107 aa', inh: 'AD', disease: 'SCA42 — French Founder p.Arg1715His / Cav3.1 GOF / Flunarizine Trial / Pure Cerebellar Adult' },
  KCND3:   { full: 'KCND3 / Kv4.3 655aa',           locus: '1p13.2',  size: '655 aa',  inh: 'AD', disease: 'SCA19/22 — Cognitive+Psychiatric EARLY PATHOGNOMONIC / Myoclonus 25% / Tremor 40% / Dutch-Japanese' },
  TMEM240: { full: 'TMEM240 / TMEM240 232aa',        locus: '1p36.33', size: '232 aa',  inh: 'AD', disease: 'SCA21 — Cognitive Childhood PATHOGNOMONIC (ADHD/Dyslexia) / Ataxia Delayed / Extrapyramidal 30%' },
  STUB1:   { full: 'STUB1 / CHIP 303aa',             locus: '16p13.3', size: '303 aa',  inh: 'AD', disease: 'SCA48 — Cognitive DOMINATES / FTD-Mimic / Parkinsonism 30% / Het=SCA48 / Biallelic=SCAR16' },
  GRM1:    { full: 'GRM1 / mGluR1 1194aa',           locus: '6q24.3',  size: '1194 aa', inh: 'AD', disease: 'SCA44 — Action Tremor PROMINENT Before Ataxia / ET Misdiagnosis / mGluR1 Gq-PKC' },
  FAT2:    { full: 'FAT2 / FAT2 4589aa',             locus: '5q33.1',  size: '4589 aa', inh: 'AD', disease: 'SCA45 — Pure Cerebellar Adult / Protocadherin / Brazilian Founder / Slow Progression' },
  PUM1:    { full: 'PUM1 / Pumilio-1 1186aa',        locus: '1p35.2',  size: '1186 aa', inh: 'AD', disease: 'SCA47 — Developmental Delay/ID PATHOGNOMONIC / Childhood Onset / Seizures 30% / Cerebellar Hypoplasia' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('LEVEL-B') || flag.includes('LEVEL-A') ? '#1565c0'
    : flag.includes('MISS') || flag.includes('MISSES') || flag.includes('MISDIAGNOSIS') || flag.includes('MIMIC') ? '#880e4f'
    : flag.includes('CONTRAINDICATED') || flag.includes('CI') || flag.includes('AVOID') ? '#880e4f'
    : flag.includes('EMERGENCY') || flag.includes('FATAL') ? '#e65100'
    : flag.includes('TREATMENT') || flag.includes('TRIAL') || flag.includes('VIGABATRIN') || flag.includes('PROPRANOLOL') || flag.includes('LEVODOPA') || flag.includes('DONEPEZIL') ? '#2e7d32'
    : flag.includes('FOUNDER') || flag.includes('ANCESTRY') || flag.includes('LINCOLN') ? '#00695c'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryNonPolyQDominantSCAAtlasPage() {
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
      <h1 style={{ color: '#1a237e', marginBottom: 4 }}>
        🧬 Hereditary Non-Polyglutamine Dominant SCA Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Non-Polyglutamine Autosomal-Dominant Spinocerebellar Ataxia Atlas —
        SPTBN2 (SCA5/Lincoln) · CACNA1G (SCA42/French) · KCND3 (SCA19/Cognitive+Myoclonus) · TMEM240 (SCA21/Childhood-Cognitive) · STUB1 (SCA48/FTD-Mimic) · GRM1 (SCA44/Action-Tremor) · FAT2 (SCA45/Protocadherin) · PUM1 (SCA47/ID+Hypoplasia)
        &nbsp;|&nbsp; 320 patients · seeds 2110-2117
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: activeTab === t ? '#1a237e' : '#e8eaf6',
            color: activeTab === t ? '#fff' : '#1a237e', fontWeight: activeTab === t ? 700 : 400,
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
              { label: 'Total Patients', value: overview.total_patients, color: '#1a237e' },
              { label: 'STUB1 Cognitive / FTD-Mimic (SCA48)', value: overview.stub1_cognitive_ftd_mimic_patients, color: '#b71c1c' },
              { label: 'KCND3 Cognitive+Psychiatric (SCA19)', value: overview.kcnd3_cognitive_psychiatric_patients, color: '#4a148c' },
              { label: 'TMEM240 Childhood Cognitive (SCA21)', value: overview.tmem240_childhood_cognitive_patients, color: '#880e4f' },
              { label: 'KCND3 Myoclonus', value: overview.kcnd3_myoclonus_patients, color: '#4a148c' },
              { label: 'PUM1 Seizures (SCA47)', value: overview.pum1_seizure_patients, color: '#37474f' },
              { label: 'PUM1 Childhood ID (SCA47)', value: overview.pum1_childhood_id_patients, color: '#37474f' },
              { label: 'STUB1 Parkinsonism (SCA48)', value: overview.stub1_parkinsonism_patients, color: '#b71c1c' },
              { label: 'GRM1 Tremor Prominent (SCA44)', value: overview.grm1_tremor_prominent_patients, color: '#e65100' },
              { label: 'Any Gene Cognitive Impairment', value: overview.cognitive_any_gene_patients, color: '#880e4f' },
              { label: 'ET Misdiagnosis Risk (GRM1+KCND3)', value: overview.et_misdiagnosis_risk_patients, color: '#e65100' },
              { label: 'FTD Misdiagnosis Risk (STUB1)', value: overview.ftd_misdiagnosis_risk_patients, color: '#b71c1c' },
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

          {/* Misdiagnosis Alert */}
          <div style={{ background: '#fce4ec', border: '2px solid #c62828', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#c62828', margin: '0 0 10px' }}>⚠️ Misdiagnosis Traps — Non-Polyglutamine Dominant SCAs</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>STUB1/SCA48:</strong> Cognitive impairment DOMINATES — misdiagnosed as FTD or early-onset dementia for years before ataxia; cerebellar signs appear AFTER cognitive decline</li>
              <li><strong>KCND3/SCA19:</strong> Cognitive, psychiatric features, and tremor BEFORE ataxia — misdiagnosed as Essential Tremor or psychiatric disorder</li>
              <li><strong>GRM1/SCA44:</strong> Action tremor prominent BEFORE cerebellar ataxia — misdiagnosed as Essential Tremor; mGluR1 not in ET workup</li>
              <li><strong>TMEM240/SCA21:</strong> ADHD/dyslexia/cognitive issues in childhood — ataxia delayed by decades; childhood cognitive → later ataxia = test TMEM240</li>
              <li><strong>PUM1/SCA47:</strong> Childhood ataxia + ID + cerebellar HYPOPLASIA (not atrophy) — may be misclassified as non-genetic developmental delay</li>
              <li><strong>STUB1 zygosity:</strong> Heterozygous = AD SCA48; Biallelic = AR SCAR16 — always report zygosity explicitly</li>
            </ul>
          </div>

          {/* Pathognomonic Finder */}
          <div style={{ background: '#e8f5e9', border: '2px solid #2e7d32', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#2e7d32', margin: '0 0 10px' }}>🎯 Pathognomonic / Prominent Findings — Non-PolyQ SCA Gene Finder</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>Cognitive impairment DOMINATES (FTD-like) + late-onset ataxia:</strong> STUB1/SCA48 — check STUB1 zygosity</li>
              <li><strong>Cognitive + psychiatric + myoclonus + tremor before ataxia (Dutch/Japanese):</strong> KCND3/SCA19/22 — LEV for myoclonus</li>
              <li><strong>Childhood ADHD/dyslexia + adult-onset ataxia (French):</strong> TMEM240/SCA21 — cognitive precedes ataxia</li>
              <li><strong>Parkinsonism (DaT-SPECT reduced) + late cerebellar ataxia:</strong> STUB1/SCA48 — CHIP E3 ligase</li>
              <li><strong>Action tremor (ET misdiagnosis) → later ataxia:</strong> GRM1/SCA44 — propranolol/DBS VIM</li>
              <li><strong>Childhood ataxia + ID + cerebellar hypoplasia (not atrophy) ± seizures:</strong> PUM1/SCA47 — vigabatrin/LEV</li>
              <li><strong>Pure cerebellar + very slow (drives 20yr) + North American Lincoln descent:</strong> SPTBN2/SCA5</li>
              <li><strong>Pure cerebellar adult + French/French-Canadian + T-type Ca GOF:</strong> CACNA1G/SCA42 — flunarizine trial</li>
            </ul>
          </div>

          {/* Treatment Priority Box */}
          <div style={{ background: '#e3f2fd', border: '2px solid #1565c0', borderRadius: 8, padding: 16 }}>
            <h3 style={{ color: '#1565c0', margin: '0 0 10px' }}>💊 Treatment Priority — Non-PolyQ Dominant SCAs</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>CACNA1G/SCA42:</strong> FLUNARIZINE 10 mg/day — T-type Ca blocker, GOF mechanism rationale; trial in French-Canadian kindreds</li>
              <li><strong>KCND3/SCA19:</strong> LEVETIRACETAM (myoclonus) + PROPRANOLOL (tremor) + SSRI (psychiatric component)</li>
              <li><strong>STUB1/SCA48:</strong> LEVODOPA trial (Parkinsonism) + Donepezil/Rivastigmine (cognitive) + SSRI (psychiatric)</li>
              <li><strong>GRM1/SCA44:</strong> PROPRANOLOL 40-120 mg/day → DBS VIM if refractory; mGluR1 NAM trial eligibility</li>
              <li><strong>PUM1/SCA47:</strong> VIGABATRIN (infantile spasms) + LEVETIRACETAM (focal) + educational support</li>
              <li><strong>All SCAs:</strong> RILUZOLE 50 mg BD off-label (Level B neuroprotection); physiotherapy + SLT</li>
            </ul>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {activeTab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inh.', 'Disease / Syndrome', 'Patients', 'Key Feature'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, gd], idx) => {
                const gi = GENE_INFO[gene] || {};
                return (
                  <tr key={gene} style={{ background: idx % 2 === 0 ? '#e8eaf6' : '#fff' }}>
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

          {/* Stats table */}
          <h3 style={{ color: '#1a237e', marginTop: 24, marginBottom: 12 }}>Clinical Feature Frequencies by Gene</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#3949ab', color: '#fff' }}>
                {['Gene', 'Mean Onset (yr)', 'Mean SARA', 'Cognitive %', 'Myoclonus %', 'Parkinsonism %', 'Tremor %', 'Seizures %', 'Childhood ID %'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, gd], idx) => (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#e8eaf6' : '#fff' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                  <td style={{ padding: '8px 10px' }}>{gd.mean_onset_age}</td>
                  <td style={{ padding: '8px 10px' }}>{gd.mean_sara}</td>
                  <td style={{ padding: '8px 10px', color: gd.cognitive_pct > 50 ? '#b71c1c' : '#333', fontWeight: gd.cognitive_pct > 50 ? 700 : 400 }}>{gd.cognitive_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.myoclonus_pct > 15 ? '#880e4f' : '#333', fontWeight: gd.myoclonus_pct > 15 ? 700 : 400 }}>{gd.myoclonus_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.parkinsonism_pct > 20 ? '#b71c1c' : '#333', fontWeight: gd.parkinsonism_pct > 20 ? 700 : 400 }}>{gd.parkinsonism_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.tremor_prominent_pct > 30 ? '#e65100' : '#333', fontWeight: gd.tremor_prominent_pct > 30 ? 700 : 400 }}>{gd.tremor_prominent_pct}%</td>
                  <td style={{ padding: '8px 10px' }}>{gd.seizures_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.childhood_id_pct > 50 ? '#37474f' : '#333', fontWeight: gd.childhood_id_pct > 50 ? 700 : 400 }}>{gd.childhood_id_pct}%</td>
                </tr>
              ))}
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
          <h2 style={{ color: '#1a237e', marginBottom: 12 }}>Gene Definitions</h2>
          {Object.entries(definitions.genes || {}).map(([gene, def]) => (
            <details key={gene} style={{ marginBottom: 10, border: `1px solid ${GENE_COLORS[gene] || '#ccc'}`, borderRadius: 6 }}>
              <summary style={{
                padding: '10px 14px', cursor: 'pointer', fontWeight: 700,
                color: GENE_COLORS[gene] || '#333', background: '#f5f5f5',
              }}>{gene}</summary>
              <div style={{ padding: '12px 14px', fontSize: 13, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{def}</div>
            </details>
          ))}

          <h2 style={{ color: '#1a237e', margin: '24px 0 12px' }}>Clinical Glossary</h2>
          {Object.entries(definitions.glossary || {}).map(([term, explanation]) => (
            <details key={term} style={{ marginBottom: 8, border: '1px solid #9fa8da', borderRadius: 6 }}>
              <summary style={{ padding: '10px 14px', cursor: 'pointer', fontWeight: 600, background: '#e8eaf6', color: '#1a237e' }}>{term}</summary>
              <div style={{ padding: '12px 14px', fontSize: 13, lineHeight: 1.7 }}>{explanation}</div>
            </details>
          ))}

          <h2 style={{ color: '#1a237e', margin: '24px 0 12px' }}>Surveillance Protocols</h2>
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
