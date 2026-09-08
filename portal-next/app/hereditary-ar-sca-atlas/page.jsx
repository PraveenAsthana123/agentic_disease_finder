'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ar-sca-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FXN:   '#4a148c',  // deep purple  — FRDA most common hereditary ataxia
  SACS:  '#1b5e20',  // deep green   — ARSACS spasticity + retinal stripes
  APTX:  '#e65100',  // deep orange  — AOA1 hypoalbuminaemia + oculomotor apraxia
  SETX:  '#0d47a1',  // deep blue    — AOA2 elevated AFP
  TTPA:  '#f9a825',  // amber        — AVED vitamin E deficiency TREATABLE
  POLG:  '#b71c1c',  // deep red     — POLG valproate ABSOLUTE CI epilepsy
  ADCK3: '#006064',  // deep cyan    — ARCA2 CoQ10 deficiency
  SYNE1: '#37474f',  // dark slate   — ARCA1 pure cerebellar French-Canadian
};

const GENE_INFO = {
  FXN:   { full: 'FXN / Frataxin 210aa',      locus: '9q21.11',  size: '210 aa',   inh: 'AR', disease: 'Friedreich Ataxia — GAA Repeat >66 / Absent Lower Limb Reflexes PATHOGNOMONIC / Cardiomyopathy / Omaveloxolone FDA-2023' },
  SACS:  { full: 'SACS / Sacsin 4579aa',      locus: '13q12.12', size: '4579 aa',  inh: 'AR', disease: 'ARSACS — Spasticity PATHOGNOMONIC / Retinal Hypermyelination PATHOGNOMONIC / French-Canadian Founder / Enlarged Pons MRI' },
  APTX:  { full: 'APTX / Aprataxin 342aa',    locus: '9p21.1',   size: '342 aa',   inh: 'AR', disease: 'AOA1 — Oculomotor Apraxia PATHOGNOMONIC / Hypoalbuminaemia PATHOGNOMONIC / Hypercholesterolaemia / Japan-Portugal' },
  SETX:  { full: 'SETX / Senataxin 2667aa',   locus: '9q34.13',  size: '2667 aa',  inh: 'AR', disease: 'AOA2 — Elevated AFP PATHOGNOMONIC / Normal Albumin (DDx AOA1) / Most Common AR Ataxia Southern Europe' },
  TTPA:  { full: 'TTPA / α-TTP 278aa',        locus: '8q12.3',   size: '278 aa',   inh: 'AR', disease: 'AVED — Very Low Vitamin E PATHOGNOMONIC / TREATABLE Vitamin E 800-1200mg/day / North African-Mediterranean Founder' },
  POLG:  { full: 'POLG / Pol-γ 1239aa',       locus: '15q26.1',  size: '1239 aa',  inh: 'AR', disease: 'POLG-Ataxia — Epilepsy PROMINENT / VALPROATE ABSOLUTE CI (Fatal Hepatotoxicity) / LEV+LTG Safe / MIRAS-MEMSA-SANDO' },
  ADCK3: { full: 'ADCK3 / CABC1 454aa',       locus: '1q42.13',  size: '454 aa',   inh: 'AR', disease: 'ARCA2 — CoQ10 Deficiency / Elevated Lactate / CoQ10 Supplementation May Improve / Childhood Onset' },
  SYNE1: { full: 'SYNE1 / Nesprin-1 8797aa',  locus: '6q25.2',   size: '8797 aa',  inh: 'AR', disease: 'ARCA1 — Pure Cerebellar / No Extracerebellar Features / French-Canadian Quebec Founder / Slow Progression' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('ABSOLUTE-CI') || flag.includes('CONTRAINDICATED') || flag.includes('VALPROATE') ? '#880e4f'
    : flag.includes('MANDATORY') || flag.includes('LEVEL-A') || flag.includes('LEVEL-B') || flag.includes('FDA') ? '#1565c0'
    : flag.includes('TREATABLE') || flag.includes('SUPPLEMENTATION') || flag.includes('VITAMIN-E') || flag.includes('CoQ10') ? '#2e7d32'
    : flag.includes('MISS') || flag.includes('MISSES') || flag.includes('MISSED') ? '#880e4f'
    : flag.includes('FOUNDER') || flag.includes('FRENCH-CANADIAN') || flag.includes('JAPAN') || flag.includes('NORTH-AFRICAN') ? '#00695c'
    : flag.includes('FATAL') || flag.includes('EMERGENCY') ? '#e65100'
    : flag.includes('DDx') || flag.includes('DISTINGUISH') ? '#4a148c'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryARSCAAtlasPage() {
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
        🧬 Hereditary AR-SCA Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Autosomal-Recessive Cerebellar Ataxia Atlas —
        FXN (FRDA/Most-Common) · SACS (ARSACS/Spasticity+Retina) · APTX (AOA1/Albumin↓) · SETX (AOA2/AFP↑) · TTPA (AVED/Vit-E-Treatable) · POLG (VPA-ABSOLUTE-CI) · ADCK3 (CoQ10/ARCA2) · SYNE1 (ARCA1/Pure-Cerebellar)
        &nbsp;|&nbsp; 320 patients · seeds 2118-2125
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
              { label: 'FRDA Omaveloxolone Eligible (FXN)', value: overview.frda_omaveloxolone_eligible_patients, color: '#4a148c' },
              { label: 'FRDA Absent Reflexes (FXN)', value: overview.frda_absent_reflexes_patients, color: '#4a148c' },
              { label: 'FRDA Cardiomyopathy (FXN)', value: overview.frda_cardiomyopathy_patients, color: '#4a148c' },
              { label: 'ARSACS Spasticity (SACS)', value: overview.sacs_spasticity_patients, color: '#1b5e20' },
              { label: 'AOA1 Low Albumin (APTX)', value: overview.aoa1_low_albumin_patients, color: '#e65100' },
              { label: 'AOA2 Elevated AFP (SETX)', value: overview.aoa2_elevated_afp_patients, color: '#0d47a1' },
              { label: 'AVED Low Vitamin E (TTPA)', value: overview.aved_low_vitamin_e_patients, color: '#f9a825' },
              { label: 'POLG Epilepsy (VPA CI)', value: overview.polg_epilepsy_patients, color: '#b71c1c' },
              { label: 'ARCA2 CoQ10 Deficient (ADCK3)', value: overview.adck3_coq10_deficient_patients, color: '#006064' },
              { label: 'ARCA1 Pure Cerebellar (SYNE1)', value: overview.syne1_pure_cerebellar_patients, color: '#37474f' },
              { label: 'Treatable Patients (FRDA+AVED+ARCA2)', value: overview.treatable_patients, color: '#2e7d32' },
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

          {/* Valproate Danger Alert */}
          <div style={{ background: '#ffebee', border: '3px solid #b71c1c', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#b71c1c', margin: '0 0 10px' }}>🚨 VALPROATE ABSOLUTE CONTRAINDICATION — POLG-Related Ataxia</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>POLG (POLG-ataxia):</strong> Valproate triggers FATAL HEPATIC FAILURE (Alpers-type) — mitochondrial toxicity → hepatocellular necrosis → liver failure</li>
              <li><strong>Rule:</strong> In ANY patient with ataxia + epilepsy → EXCLUDE POLG BEFORE prescribing valproate</li>
              <li><strong>Safe AEDs:</strong> Levetiracetam (first-line), Lamotrigine, Lacosamide — no mitochondrial toxicity</li>
              <li><strong>Also avoid:</strong> Phenobarbitone (ETC complex I toxicity), Linezolid (ETC toxicity)</li>
              <li><strong>MIRAS alleles (Scandinavian):</strong> p.Ala467Thr + p.Trp748Ser — test these first in Norwegian/Finnish patients with ataxia+epilepsy</li>
            </ul>
          </div>

          {/* Treatable Conditions Alert */}
          <div style={{ background: '#e8f5e9', border: '2px solid #2e7d32', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#2e7d32', margin: '0 0 10px' }}>✅ Treatable AR-SCAs — Must Not Miss</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>AVED (TTPA):</strong> Serum vitamin E &lt;3 mg/L → Vitamin E 800-1200 mg/day → halts/reverses progression; LIFELONG, NEVER STOP; test siblings presymptomatically</li>
              <li><strong>ARCA2 (ADCK3):</strong> Muscle CoQ10 assay → CoQ10 300-2400 mg/day → 30-40% improve; ubiquinol preferred; elevated lactate as biomarker</li>
              <li><strong>FRDA (FXN):</strong> Omaveloxolone (Skyclarys) 150 mg/day — FDA-approved Feb 2023; Nrf2 activator; slows SARA decline; age ≥16 yr, ambulant</li>
              <li><strong>Check in ALL ataxia workup:</strong> Serum vitamin E (AVED), muscle CoQ10 (ARCA2), FXN GAA repeat (FRDA — WES MISSES this)</li>
            </ul>
          </div>

          {/* Diagnostic DDx Panel */}
          <div style={{ background: '#e8eaf6', border: '2px solid #3949ab', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#3949ab', margin: '0 0 10px' }}>🔬 AR-SCA Diagnostic Distinguishers — Biochemical Panel</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>Absent lower limb reflexes + cardiomyopathy:</strong> FXN (FRDA) — GAA repeat-primed PCR; WES MISSES repeat expansion</li>
              <li><strong>Spasticity + retinal hypermyelination (yellow stripes on OCT):</strong> SACS (ARSACS) — French-Canadian or Turkish/Italian/North African</li>
              <li><strong>Oculomotor apraxia + LOW albumin + HIGH cholesterol, normal AFP:</strong> APTX (AOA1) — Japan/Portugal</li>
              <li><strong>Oculomotor apraxia + ELEVATED AFP, normal albumin:</strong> SETX (AOA2) — Southern Europe; AFP distinguishes from AOA1</li>
              <li><strong>Ataxia + very low vitamin E (&lt;3 mg/L):</strong> TTPA (AVED) — North African/Mediterranean; treat immediately</li>
              <li><strong>Ataxia + epilepsy + elevated lactate:</strong> POLG — EXCLUDE before valproate; muscle mtDNA analysis</li>
              <li><strong>Childhood ataxia + elevated lactate + CoQ10 deficient (muscle):</strong> ADCK3 (ARCA2) — CoQ10 supplementation trial</li>
              <li><strong>Pure cerebellar + normal reflexes + normal biochemistry + slow:</strong> SYNE1 (ARCA1) — French-Canadian; diagnosis of exclusion</li>
            </ul>
          </div>

          {/* Repeat Expansion Warning */}
          <div style={{ background: '#fff8e1', border: '2px solid #f57f17', borderRadius: 8, padding: 16 }}>
            <h3 style={{ color: '#f57f17', margin: '0 0 10px' }}>⚠️ FXN GAA Repeat — Standard WES/NGS MISSES Friedreich Ataxia</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>GAA triplet repeat</strong> in FXN intron 1 is NOT detected by standard whole-exome or whole-genome sequencing</li>
              <li><strong>Triplet-primed PCR</strong> (repeat-primed PCR) is MANDATORY — request specifically</li>
              <li>If NGS/exome is negative in classic FRDA phenotype (absent reflexes + ataxia ± cardiomyopathy), order <em>FXN GAA repeat analysis separately</em></li>
              <li>Most common cause of missed FRDA diagnosis on next-generation sequencing panels</li>
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
                {['Gene', 'Locus', 'Size', 'Inh.', 'Disease / Syndrome', 'Patients', 'Key Feature'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, gd], idx) => {
                const gi = GENE_INFO[gene] || {};
                return (
                  <tr key={gene} style={{ background: idx % 2 === 0 ? '#ede7f6' : '#fff' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                    <td style={{ padding: '10px 12px' }}>{gd.locus}</td>
                    <td style={{ padding: '10px 12px', whiteSpace: 'nowrap' }}>{gd.protein_size}</td>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{gi.inh || 'AR'}</td>
                    <td style={{ padding: '10px 12px', maxWidth: 280 }}><small>{gi.disease || '—'}</small></td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gd.patient_count}</td>
                    <td style={{ padding: '10px 12px', maxWidth: 220 }}><small style={{ color: '#555' }}>{gd.key_biomarker?.split(';')[0]}</small></td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Clinical feature frequency table */}
          <h3 style={{ color: '#4a148c', marginTop: 24, marginBottom: 12 }}>Clinical Feature Frequencies by Gene</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#7b1fa2', color: '#fff' }}>
                {['Gene', 'Mean Onset (yr)', 'Mean SARA', 'Absent Reflexes %', 'Cardiomyopathy %', 'Spasticity %', 'Oculomotor Apraxia %', 'Epilepsy %', 'Elevated Lactate %'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, gd], idx) => (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#ede7f6' : '#fff' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                  <td style={{ padding: '8px 10px' }}>{gd.mean_onset_age}</td>
                  <td style={{ padding: '8px 10px' }}>{gd.mean_sara}</td>
                  <td style={{ padding: '8px 10px', color: gd.absent_reflexes_pct > 50 ? '#b71c1c' : '#333', fontWeight: gd.absent_reflexes_pct > 50 ? 700 : 400 }}>{gd.absent_reflexes_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.cardiomyopathy_pct > 20 ? '#880e4f' : '#333', fontWeight: gd.cardiomyopathy_pct > 20 ? 700 : 400 }}>{gd.cardiomyopathy_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.spasticity_pct > 50 ? '#1b5e20' : '#333', fontWeight: gd.spasticity_pct > 50 ? 700 : 400 }}>{gd.spasticity_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.oculomotor_apraxia_pct > 50 ? '#e65100' : '#333', fontWeight: gd.oculomotor_apraxia_pct > 50 ? 700 : 400 }}>{gd.oculomotor_apraxia_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.epilepsy_pct > 50 ? '#b71c1c' : '#333', fontWeight: gd.epilepsy_pct > 50 ? 700 : 400 }}>{gd.epilepsy_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.elevated_lactate_pct > 50 ? '#006064' : '#333', fontWeight: gd.elevated_lactate_pct > 50 ? 700 : 400 }}>{gd.elevated_lactate_pct}%</td>
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
              <summary style={{ padding: '10px 14px', cursor: 'pointer', fontWeight: 600, background: '#ede7f6', color: '#4a148c' }}>{term}</summary>
              <div style={{ padding: '12px 14px', fontSize: 13, lineHeight: 1.7 }}>{explanation}</div>
            </details>
          ))}

          <h2 style={{ color: '#4a148c', margin: '24px 0 12px' }}>Surveillance Protocols</h2>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, protocol]) => {
            const gKey = gene.split(' ')[0].replace(/[()]/g, '');
            return (
              <details key={gene} style={{ marginBottom: 8, border: `1px solid ${GENE_COLORS[gKey] || '#ccc'}`, borderRadius: 6 }}>
                <summary style={{
                  padding: '10px 14px', cursor: 'pointer', fontWeight: 600,
                  background: '#fce4ec', color: '#c62828',
                }}>{gene}</summary>
                <div style={{ padding: '12px 14px', fontSize: 13, lineHeight: 1.7 }}>{protocol}</div>
              </details>
            );
          })}
        </div>
      )}
    </div>
  );
}
