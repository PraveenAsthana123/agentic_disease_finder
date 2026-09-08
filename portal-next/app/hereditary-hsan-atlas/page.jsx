'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hsan-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SPTLC1:  '#4a148c',  // deep purple  — HSAN1A shooting pains, L-serine treatable
  IKBKAP:  '#1b5e20',  // deep green   — HSAN3/FD autonomic crises + Ashkenazi
  NTRK1:   '#b71c1c',  // deep red     — HSAN4/CIPA self-mutilation + hyperthermia
  SCN9A:   '#0d47a1',  // deep blue    — PARADOX: LOF=CIP / GOF=erythromelalgia
  RETREG1: '#e65100',  // deep orange  — HSAN2B corneal ulcers + mutilation
  WNK1:    '#006064',  // deep cyan    — HSAN2A severe mutilation HSN2-exon
  DNMT1:   '#880e4f',  // deep pink    — HSAN1E narcolepsy+dementia+hearing
  ATL1:    '#37474f',  // dark slate   — HSAN1D cough+GERD+hearing+neuropathy
};

const GENE_INFO = {
  SPTLC1:  { full: 'SPTLC1 / Ser-Palm-Transf-LCB1 479aa', locus: '9q22.31',  size: '479 aa',   inh: 'AD', disease: 'HSAN1A — Shooting/Lancinating Pains PATHOGNOMONIC / Foot Ulcers / L-Serine 400mg/kg/day TREATABLE / dSL metabolites' },
  IKBKAP:  { full: 'IKBKAP / ELP1 1332aa',                 locus: '9q31.3',   size: '1332 aa',  inh: 'AR', disease: 'HSAN3/Riley-Day/FD — Absent Fungiform Papillae PATHOGNOMONIC / Autonomic Crises PATHOGNOMONIC / Ashkenazi IVS20+6T>C 99.5%' },
  NTRK1:   { full: 'NTRK1 / TrkA 796aa',                   locus: '1q23.1',   size: '796 aa',   inh: 'AR', disease: 'HSAN4/CIPA — Pain Insensitivity + Anhidrosis + Self-Mutilation PATHOGNOMONIC / Hyperthermia Deaths / AVOID AMPUTATION' },
  SCN9A:   { full: 'SCN9A / Nav1.7 1988aa',                locus: '2q24.3',   size: '1988 aa',  inh: 'AR/AD', disease: 'PARADOX — LOF(AR)=CIP+Anosmia / GOF(AD)=Erythromelalgia Burning Feet PATHOGNOMONIC / Carbamazepine GOF pain' },
  RETREG1: { full: 'RETREG1 / FAM134B 460aa',              locus: '5p15.1',   size: '460 aa',   inh: 'AR', disease: 'HSAN2B — Corneal Anaesthesia → Neurotrophic Ulceration PATHOGNOMONIC / Pan-Sensory Loss / Mutilating Arthropathy' },
  WNK1:    { full: 'WNK1 / HSN2-isoform 2382aa',           locus: '12p13.33', size: '2382 aa',  inh: 'AR', disease: 'HSAN2A — Severe Mutilation Extremities PATHOGNOMONIC / HSN2 Exon Standard-WES-MISSES / Sudanese-Nova-Scotian Founder' },
  DNMT1:   { full: 'DNMT1 / DNA-Methyltransf-1 1616aa',    locus: '19p13.2',  size: '1616 aa',  inh: 'AD', disease: 'HSAN1E/ADCA-DN — Narcolepsy+Hearing Loss+Dementia+Neuropathy = 4-FEATURE PATHOGNOMONIC / REMD domain' },
  ATL1:    { full: 'ATL1 / Atlastin-1 558aa',              locus: '14q22.1',  size: '558 aa',   inh: 'AD', disease: 'HSAN1D — Chronic Cough+GERD PATHOGNOMONIC / Sensorineural Hearing Loss / Late-Onset 40-60yr / ATL1≠SPG3A' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('ABSOLUTE-CI') || flag.includes('AVOID-AMPUTATION') || flag.includes('MANDATORY') ? '#880e4f'
    : flag.includes('TREATABLE') || flag.includes('L-SERINE') || flag.includes('CARBAMAZEPINE') ? '#2e7d32'
    : flag.includes('MISS') || flag.includes('MISSES') || flag.includes('MISSED') ? '#880e4f'
    : flag.includes('FOUNDER') || flag.includes('ASHKENAZI') || flag.includes('SUDANESE') ? '#00695c'
    : flag.includes('FATAL') || flag.includes('HYPERTHERMIA') || flag.includes('SELF-MUTILATION') ? '#e65100'
    : flag.includes('PARADOX') || flag.includes('DDx') ? '#4a148c'
    : flag.includes('CONGENITAL') ? '#0d47a1'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryHSANAtlasPage() {
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
        🧬 Hereditary HSAN Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Hereditary Sensory and Autonomic Neuropathy Atlas —
        SPTLC1 (HSAN1A/L-Serine-Treatable) · IKBKAP/ELP1 (HSAN3/FD/Ashkenazi) · NTRK1 (HSAN4/CIPA/Hyperthermia) · SCN9A (PARADOX:CIP+IEM) · RETREG1 (HSAN2B/Cornea) · WNK1-HSN2 (HSAN2A/Mutilation) · DNMT1 (HSAN1E/4-Feature) · ATL1 (HSAN1D/Cough+GERD)
        &nbsp;|&nbsp; 320 patients · seeds 2126-2133
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
              { label: 'SPTLC1 Shooting Pains (HSAN1A)', value: overview.sptlc1_shooting_pains_patients, color: '#4a148c' },
              { label: 'IKBKAP Autonomic Crises (FD)', value: overview.ikbkap_autonomic_crises_patients, color: '#1b5e20' },
              { label: 'NTRK1 Pain Insensitivity (CIPA)', value: overview.ntrk1_pain_insensitivity_patients, color: '#b71c1c' },
              { label: 'NTRK1 Self-Mutilation (CIPA)', value: overview.ntrk1_self_mutilation_patients, color: '#b71c1c' },
              { label: 'SCN9A CIP — LOF (no pain)', value: overview.scn9a_cip_lof_patients, color: '#0d47a1' },
              { label: 'SCN9A Erythromelalgia — GOF', value: overview.scn9a_iem_gof_patients, color: '#0d47a1' },
              { label: 'RETREG1 Corneal Ulcers (HSAN2B)', value: overview.retreg1_corneal_ulcers_patients, color: '#e65100' },
              { label: 'WNK1 Self-Mutilation (HSAN2A)', value: overview.wnk1_self_mutilation_patients, color: '#006064' },
              { label: 'DNMT1 Narcolepsy (HSAN1E)', value: overview.dnmt1_narcolepsy_patients, color: '#880e4f' },
              { label: 'DNMT1 Dementia (HSAN1E)', value: overview.dnmt1_dementia_patients, color: '#880e4f' },
              { label: 'Treatable (SPTLC1 L-Serine)', value: overview.treatable_patients, color: '#2e7d32' },
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

          {/* CIPA Hyperthermia Alert */}
          <div style={{ background: '#ffebee', border: '3px solid #b71c1c', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#b71c1c', margin: '0 0 10px' }}>🚨 CIPA (NTRK1) — Hyperthermia Is a Life-Threatening Emergency</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>CIPA (HSAN4):</strong> Anhidrosis → no sweating → heat cannot be dissipated → anhidrotic hyperthermia → death</li>
              <li><strong>Most deaths in early childhood</strong> from heat stroke — NOT from injury or infection</li>
              <li><strong>Fever protocol MANDATORY:</strong> cool environment always; fan + tepid sponging + paracetamol for ANY temp &gt;38°C</li>
              <li><strong>Avoid:</strong> vigorous exercise, hot weather, hot baths, heated rooms, direct sun</li>
              <li><strong>AVOID AMPUTATION:</strong> fractures and osteomyelitis heal with conservative management — amputation should be last resort</li>
            </ul>
          </div>

          {/* Treatable Alert */}
          <div style={{ background: '#e8f5e9', border: '2px solid #2e7d32', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#2e7d32', margin: '0 0 10px' }}>✅ Treatable HSAN — SPTLC1 (HSAN1A): L-Serine Supplementation</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>SPTLC1 GOF:</strong> mutant enzyme uses L-alanine → deoxy-sphingolipids (dSL) accumulate → sensory neuron toxicity</li>
              <li><strong>L-SERINE 400 mg/kg/day:</strong> competitive substrate supplementation → reduces plasma dSL levels → slows neuropathy progression</li>
              <li><strong>Phase 2 trial evidence:</strong> SNAP amplitude stabilisation; plasma dSL monitoring on treatment</li>
              <li><strong>Presymptomatic carriers:</strong> cascade family testing → early L-serine before ulcers develop</li>
              <li><strong>SCN9A GOF IEM:</strong> carbamazepine — variant-specific Nav1.7 channel blocker; check genotype before prescribing</li>
            </ul>
          </div>

          {/* SCN9A Paradox Alert */}
          <div style={{ background: '#e3f2fd', border: '2px solid #0d47a1', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#0d47a1', margin: '0 0 10px' }}>🔄 SCN9A Nav1.7 — Paradox Gene: Same Channel, Opposite Pain Phenotypes</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>LOF (AR biallelic):</strong> complete pain insensitivity from birth + anosmia 50% + normal intellect (HSAN2D/CIP)</li>
              <li><strong>GOF (AD heterozygous):</strong> Inherited Erythromelalgia — bilateral burning feet + redness + heat triggers (PATHOGNOMONIC)</li>
              <li><strong>DDx from NTRK1/CIPA (LOF):</strong> SCN9A CIP = normal sweating + no ID vs NTRK1/CIPA = anhidrosis + ID 50%</li>
              <li><strong>GOF treatment:</strong> carbamazepine (Na-channel blocker) — effective in channel-blocking-sensitive alleles (p.Arg185His, p.Gly616Arg)</li>
              <li><strong>PEPD variant:</strong> also GOF SCN9A — rectal/periorbital/submandibular pain episodes; carbamazepine first-line</li>
            </ul>
          </div>

          {/* Diagnostic Pearls Panel */}
          <div style={{ background: '#e8eaf6', border: '2px solid #3949ab', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#3949ab', margin: '0 0 10px' }}>🔬 HSAN Diagnostic Pearls — Clinic-First Signs</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>Inspect tongue dorsum:</strong> absent fungiform papillae (smooth) = FD/IKBKAP (HSAN3) — bedside sign</li>
              <li><strong>Shooting/lancinating pains + foot ulcers, adult onset:</strong> SPTLC1 (HSAN1A) → plasma dSL; L-serine</li>
              <li><strong>Congenital pain insensitivity + self-mutilation + anhidrosis:</strong> NTRK1/CIPA — fever protocol STAT</li>
              <li><strong>Congenital pain insensitivity + anosmia + normal sweating/intellect:</strong> SCN9A LOF/CIP</li>
              <li><strong>Bilateral burning feet, warm triggers, redness:</strong> SCN9A GOF/Erythromelalgia → carbamazepine</li>
              <li><strong>Corneal anaesthesia + congenital sensory loss + mutilation:</strong> RETREG1/HSAN2B (ophthalmology urgent)</li>
              <li><strong>Severe mutilation + congenital + no anhidrosis:</strong> WNK1/HSAN2A — request HSN2 exon sequencing</li>
              <li><strong>Narcolepsy + hearing loss + dementia + neuropathy (any 2 of 4):</strong> DNMT1/HSAN1E → REMD domain</li>
              <li><strong>Chronic dry cough + GERD + hearing loss + neuropathy:</strong> ATL1/HSAN1D → confirm HSAN1D (not SPG3A)</li>
            </ul>
          </div>

          {/* WNK1 WES Warning */}
          <div style={{ background: '#fff8e1', border: '2px solid #f57f17', borderRadius: 8, padding: 16 }}>
            <h3 style={{ color: '#f57f17', margin: '0 0 10px' }}>⚠️ WNK1 HSN2 Exon — Standard WES May Miss HSAN2A</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>WNK1 HSAN2A mutations</strong> occur exclusively in the <em>neuronal-specific HSN2 exon</em> — not the ubiquitous kinase</li>
              <li><strong>Standard WES exome capture</strong> may not include or annotate this exon → false-negative result</li>
              <li>If clinical picture = severe congenital HSAN2 + Sudanese/Nova Scotian/French-Canadian ancestry and WES negative → request <em>WNK1 HSN2 exon-specific Sanger</em></li>
              <li>WNK1 ubiquitous kinase mutations → a separate condition (pseudohypoaldosteronism/hypertension) — do NOT conflate</li>
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
                {['Gene', 'Locus', 'Size', 'Inh.', 'HSAN Subtype / Disease', 'Patients', 'Key Feature'].map(h => (
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
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{gi.inh || '—'}</td>
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
                {['Gene', 'Mean Onset (yr)', 'Pain Insensitivity %', 'Burning Pain %', 'Shooting Pains %', 'Autonomic Crises %', 'Self-Mutilation %', 'Anhidrosis %', 'Corneal Ulcers %', 'Narcolepsy %', 'Dementia %', 'Hearing Loss %'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, gd], idx) => (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#ede7f6' : '#fff' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                  <td style={{ padding: '8px 10px' }}>{gd.mean_onset_age}</td>
                  <td style={{ padding: '8px 10px', color: gd.pain_insensitivity_pct > 50 ? '#b71c1c' : '#333', fontWeight: gd.pain_insensitivity_pct > 50 ? 700 : 400 }}>{gd.pain_insensitivity_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.burning_pain_pct > 30 ? '#0d47a1' : '#333', fontWeight: gd.burning_pain_pct > 30 ? 700 : 400 }}>{gd.burning_pain_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.shooting_pains_pct > 50 ? '#4a148c' : '#333', fontWeight: gd.shooting_pains_pct > 50 ? 700 : 400 }}>{gd.shooting_pains_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.autonomic_crises_pct > 50 ? '#1b5e20' : '#333', fontWeight: gd.autonomic_crises_pct > 50 ? 700 : 400 }}>{gd.autonomic_crises_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.self_mutilation_pct > 40 ? '#e65100' : '#333', fontWeight: gd.self_mutilation_pct > 40 ? 700 : 400 }}>{gd.self_mutilation_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.anhidrosis_pct > 50 ? '#880e4f' : '#333', fontWeight: gd.anhidrosis_pct > 50 ? 700 : 400 }}>{gd.anhidrosis_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.corneal_ulcers_pct > 40 ? '#e65100' : '#333', fontWeight: gd.corneal_ulcers_pct > 40 ? 700 : 400 }}>{gd.corneal_ulcers_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.narcolepsy_pct > 50 ? '#880e4f' : '#333', fontWeight: gd.narcolepsy_pct > 50 ? 700 : 400 }}>{gd.narcolepsy_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.dementia_pct > 50 ? '#880e4f' : '#333', fontWeight: gd.dementia_pct > 50 ? 700 : 400 }}>{gd.dementia_pct}%</td>
                  <td style={{ padding: '8px 10px', color: gd.hearing_loss_pct > 50 ? '#880e4f' : '#333', fontWeight: gd.hearing_loss_pct > 50 ? 700 : 400 }}>{gd.hearing_loss_pct}%</td>
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
            const gKey = gene.split(' ')[0].replace(/[()]/g, '').split('/')[0];
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
