'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-peripheral-neuropathy-cmt-atlas';
const TABS = ['Overview', 'Gene Table', 'NCV Analysis', 'Definitions'];

const GENE_COLORS = {
  'PMP22':  '#1565c0',  // deep blue       -- CMT1A most common 60-70%; pes cavus triad PATHOGNOMONIC
  'MPZ':    '#4a148c',  // deep purple      -- CMT1B; mutation class determines severity; onion bulb
  'GJB1':   '#880e4f',  // deep crimson     -- CMT1X; intermediate NCV PATHOGNOMONIC; males severe
  'MFN2':   '#1b5e20',  // dark green       -- CMT2A most common axonal; optic atrophy PATHOGNOMONIC
  'GDAP1':  '#b71c1c',  // dark red         -- CMT4A; vocal cord + phrenic PATHOGNOMONIC
  'SH3TC2': '#e65100',  // burnt orange     -- CMT4C; scoliosis + cranial nerve PATHOGNOMONIC; R954W
  'PRX':    '#37474f',  // dark blue-grey   -- CMT4F; onion bulb severe NCV <10 PATHOGNOMONIC
  'NEFL':   '#004d40',  // dark teal        -- CMT2E/CMT1F same gene; neurofilament accumulation
};

const GENE_INFO = {
  'PMP22':  { full: 'Peripheral-myelin-protein-22 / CMT1A-duplication-60-70pct / HNPP-deletion-episodic / pes-cavus-hammertoe-hyporeflexia-TRIAD-PATHOGNOMONIC / no-curative-therapy', locus: '17p12', size: '160 aa / 17.8 kDa', inh: 'AD' },
  'MPZ':    { full: 'Myelin-protein-zero-P0 / CMT1B-2nd-common-demyelinating / mutation-class-determines-severity / Thr124Met-late-mild / onion-bulb-severe-PATHOGNOMONIC', locus: '1q23.3', size: '248 aa / 28 kDa', inh: 'AD' },
  'GJB1':   { full: 'Connexin-32-Cx32 / CMT1X-most-common-X-linked / intermediate-NCV-25-45ms-PATHOGNOMONIC / males-severe-females-mild-lyonization / central-white-matter-lesions', locus: 'Xq13.1', size: '283 aa / 31 kDa', inh: 'XLD' },
  'MFN2':   { full: 'Mitofusin-2 / CMT2A-most-common-axonal / optic-atrophy-20-30pct-PATHOGNOMONIC / axonal-NCV-reduced-CMAP / lower-limb-predominant', locus: '1p36.22', size: '757 aa / 84 kDa', inh: 'AD' },
  'GDAP1':  { full: 'GDAP1-mitochondrial-fission / CMT4A-AR / vocal-cord-phrenic-PATHOGNOMONIC / early-onset-rapid-progression / wheelchair-2nd-3rd-decade', locus: '8q21.11', size: '358 aa / 41 kDa', inh: 'AR' },
  'SH3TC2': { full: 'SH3TC2-Schwann-cell-scaffold / CMT4C-most-common-AR-CMT-Europeans / R954W-founder / scoliosis-cranial-nerve-PATHOGNOMONIC / hearing-loss', locus: '5q32', size: '1288 aa / 142 kDa', inh: 'AR' },
  'PRX':    { full: 'Periaxin-abaxonal-scaffold / CMT4F-severe-hypomyelinating / onion-bulb-severe-PATHOGNOMONIC / NCV-lt10ms / early-onset-sensory-predominant / Dejerine-Sottas', locus: '19q13.2', size: '1461 aa / 163 kDa', inh: 'AR' },
  'NEFL':   { full: 'Neurofilament-light-chain / CMT2E-axonal-AND-CMT1F-demyelinating-SAME-GENE / neurofilament-accumulation-PATHOGNOMONIC / CSF-protein-elevated-CMT1F', locus: '8p21.2', size: '543 aa / 61 kDa', inh: 'AD' },
};

// NCV classification data
const NCV_TABLE = [
  { gene: 'PMP22', subtype: 'CMT1A', type: 'Demyelinating', ncv_range: '<38 m/s (~15-30)', pattern: 'Uniform slow', pathognomonic: 'Pes cavus triad' },
  { gene: 'MPZ',   subtype: 'CMT1B', type: 'Demyelinating', ncv_range: '<38 m/s (variable)', pattern: 'Non-uniform slow', pathognomonic: 'Onion bulb (severe)' },
  { gene: 'GJB1',  subtype: 'CMT1X', type: 'INTERMEDIATE',  ncv_range: '25-45 m/s', pattern: 'Intermediate — not pure demyel/axonal', pathognomonic: 'INTERMEDIATE NCV PATHOGNOMONIC' },
  { gene: 'MFN2',  subtype: 'CMT2A', type: 'Axonal',        ncv_range: '>38 m/s (≥40)', pattern: 'Normal NCV, ↓ CMAP amplitude', pathognomonic: 'Optic atrophy + CMT2A' },
  { gene: 'GDAP1', subtype: 'CMT4A', type: 'Axonal/Demyel', ncv_range: 'Variable 12-28', pattern: 'Mutation-dependent subtype', pathognomonic: 'Vocal cord + phrenic nerve' },
  { gene: 'SH3TC2',subtype: 'CMT4C', type: 'Demyelinating', ncv_range: '<38 m/s (~10-22)', pattern: 'Slow demyelinating', pathognomonic: 'Scoliosis + cranial nerve' },
  { gene: 'PRX',   subtype: 'CMT4F', type: 'Severe Demyel', ncv_range: '<10 m/s', pattern: 'Severely hypomyelinated', pathognomonic: 'Onion bulb + NCV <10 m/s' },
  { gene: 'NEFL',  subtype: 'CMT2E/1F', type: 'Axonal OR Demyel', ncv_range: '22-42 m/s', pattern: 'Same gene → axonal or demyel', pathognomonic: 'Neurofilament accumulation' },
];

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, breakdown }) {
  const bg = color;
  const cohort = breakdown?.gene_data?.[gene] || {};
  const geneAtlas = breakdown?.atlas_genes?.find(g => g.gene === gene) || {};
  return (
    <div style={{ border: `2px solid ${bg}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: bg, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{geneAtlas.syndrome?.split('/')[0] || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge text={`Age Dx: ${cohort.mean_age_dx} yr`} color={bg} />
        <Badge text={`Pes cavus: ${cohort.pes_cavus_n}/40`} color='#1565c0' />
        <Badge text={`Wheelchair: ${cohort.wheelchair_n}/40`} color='#6a1b9a' />
        <Badge text={`Mean NCV: ${cohort.mean_ncv} m/s`} color='#004d40' />
      </div>

      {/* Gene-specific pathognomonic highlights */}
      {gene === 'PMP22' && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#1565c0' }}>
          👣 <b>PES CAVUS TRIAD PATHOGNOMONIC:</b> Pes cavus + hammertoe + hyporeflexia = CMT. Pes cavus: {cohort.pes_cavus_n}/40. CMT1A (duplication) most common CMT — 60-70% of CMT1. NO curative therapy. Avoid neurotoxic drugs.
        </div>
      )}
      {gene === 'MPZ' && (
        <div style={{ fontSize: 12, background: '#ede7f6', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#4a148c' }}>
          🔬 <b>MUTATION CLASS DETERMINES SEVERITY:</b> Thr124Met = late-onset mild. Ile107Met = severe early-onset. Onion bulbs on biopsy: {cohort.onion_bulb_n}/40 PATHOGNOMONIC. Genotype MANDATORY before prognosis.
        </div>
      )}
      {gene === 'GJB1' && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#880e4f' }}>
          ⚡ <b>INTERMEDIATE NCV 25-45 m/s PATHOGNOMONIC:</b> Does NOT fit pure demyelinating or axonal. Intermediate NCV: {cohort.intermediate_ncv_n}/40. Males severely affected; females mild (lyonization). Central white matter lesions in severe cases.
        </div>
      )}
      {gene === 'MFN2' && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#1b5e20' }}>
          👁️ <b>OPTIC ATROPHY PATHOGNOMONIC:</b> CMT2A with optic atrophy = MFN2. Optic atrophy: {cohort.optic_atrophy_n}/40 (target ~10). Most common axonal CMT2. Mean NCV: {cohort.mean_ncv} m/s (axonal-preserved). Annual ophthalmology mandatory.
        </div>
      )}
      {gene === 'GDAP1' && (
        <div style={{ fontSize: 12, background: '#ffebee', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#b71c1c' }}>
          🗣️ <b>VOCAL CORD + PHRENIC PATHOGNOMONIC:</b> Vocal cord paresis: {cohort.vocal_cord_n}/40. ENT + pulmonary monitoring mandatory. Wheelchair {cohort.wheelchair_n}/40. Early rapid progression — AR-null mutations reach wheelchair by 2nd-3rd decade.
        </div>
      )}
      {gene === 'SH3TC2' && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#e65100' }}>
          🦴 <b>SCOLIOSIS + CRANIAL NERVE PATHOGNOMONIC:</b> Scoliosis: {cohort.scoliosis_n}/40. Cranial nerve: {cohort.cranial_nerve_n}/40. Most common AR-CMT in Europeans; R954W founder. Screen R954W first in European patients.
        </div>
      )}
      {gene === 'PRX' && (
        <div style={{ fontSize: 12, background: '#eceff1', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#37474f' }}>
          🔬 <b>ONION BULB SEVERE + NCV &lt;10 m/s PATHOGNOMONIC:</b> Onion bulbs: {cohort.onion_bulb_n}/40 — most severe of all CMT. Mean NCV: {cohort.mean_ncv} m/s. Sensory-predominant. Foot ulcers prevention essential. Dejerine-Sottas in truncating mutations.
        </div>
      )}
      {gene === 'NEFL' && (
        <div style={{ fontSize: 12, background: '#e0f2f1', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#004d40' }}>
          🧬 <b>SAME GENE TWO PHENOTYPES:</b> CMT2E (axonal) OR CMT1F (demyelinating). Neurofilament accumulation on biopsy PATHOGNOMONIC. CSF protein elevated in CMT1F. Mean NCV: {cohort.mean_ncv} m/s. Serum NfL biomarker elevated.
        </div>
      )}
      <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>
        <b>Key therapy:</b> {geneAtlas.key_therapy || ''}
      </div>
    </div>
  );
}

export default function HereditaryCMTAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#888' }}>Loading Hereditary CMT Atlas…</div>;
  if (err)     return <div style={{ padding: 40, color: 'red' }}>Error: {err}</div>;

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        🧬 Hereditary Peripheral Neuropathy (CMT Disease) Atlas
      </h1>
      <div style={{ fontSize: 13, color: '#555', marginBottom: 16 }}>
        Complete 8-Gene CMT Reference · PMP22 · MPZ · GJB1 · MFN2 · GDAP1 · SH3TC2 · PRX · NEFL ·
        Seeds {overview?.seeds} · {overview?.total_patients} patients
      </div>

      {/* NCV Classification alert */}
      <div style={{ background: '#e8f5e9', border: '2px solid #1b5e20', borderRadius: 8, padding: '10px 16px', marginBottom: 10, fontSize: 13 }}>
        <b>NCV CLASSIFICATION:</b> Demyelinating CMT1 &lt;38 m/s (PMP22, MPZ, SH3TC2, PRX) ·
        Intermediate 25-45 m/s (GJB1 PATHOGNOMONIC) · Axonal CMT2 ≥38 m/s reduced CMAP (MFN2, GDAP1-axonal) ·
        NEFL unique: same gene → axonal (CMT2E) OR demyelinating (CMT1F)
      </div>

      {/* Key pathognomonic alert */}
      <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 8, padding: '10px 16px', marginBottom: 16, fontSize: 13 }}>
        <b>⚠️ KEY PATHOGNOMONIC FEATURES:</b> GJB1 → intermediate NCV (25-45 m/s) ·
        MFN2 → CMT2A with optic atrophy · GDAP1 → vocal cord + phrenic nerve ·
        SH3TC2 → scoliosis + cranial nerve (R954W founder) · PRX → onion bulb severe + NCV &lt;10 ·
        PMP22 → pes cavus + hammertoe + hyporeflexia TRIAD
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ padding: '6px 16px', borderRadius: 20, border: 'none', cursor: 'pointer',
              background: tab === t ? '#1565c0' : '#eee', color: tab === t ? '#fff' : '#333',
              fontWeight: tab === t ? 700 : 400, fontSize: 13 }}>
            {t}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Atlas Overview — {overview.genes?.join(' · ')}</h2>
          <p style={{ fontSize: 13, color: '#444', marginBottom: 16, lineHeight: 1.6 }}>{overview.description}</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {overview.gene_summaries?.map(g => (
              <div key={g.gene} style={{
                border: `2px solid ${GENE_COLORS[g.gene] || '#999'}`,
                borderRadius: 8, padding: 12
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <div style={{ background: GENE_COLORS[g.gene] || '#999', color: '#fff',
                    borderRadius: 5, padding: '3px 10px', fontWeight: 700, fontSize: 15 }}>{g.gene}</div>
                  <div style={{ fontSize: 11, color: '#666' }}>{g.locus} · {g.inheritance}</div>
                </div>
                <div style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>{g.syndrome?.split('/').slice(0,2).join(' / ')}</div>
                <div style={{ fontSize: 11, color: '#666' }}>{g.pathognomonic?.substring(0, 100)}…</div>
                <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{g.protein_size?.split('/').slice(0,2).join(' / ')}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Cohort Summary — 320 Patients (8 × 40)</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1565c0', color: '#fff' }}>
                {['Gene', 'Locus', 'Inh', 'CMT Type', 'Age Dx (yr)', 'Pes Cavus', 'Onion Bulb', 'Wheelchair', 'Mean NCV', 'Pathognomonic'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.atlas_genes?.map((g, i) => {
                const c = breakdown.gene_data?.[g.gene] || {};
                const ncvClass = c.mean_ncv < 10 ? '#b71c1c' : c.mean_ncv < 38 ? '#e65100' : c.mean_ncv < 45 ? '#880e4f' : '#1b5e20';
                return (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
                    <td style={{ padding: '8px 10px' }}>{g.locus}</td>
                    <td style={{ padding: '8px 10px' }}>{g.inheritance}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11 }}>{g.syndrome?.split('/')[0]}</td>
                    <td style={{ padding: '8px 10px' }}>{c.mean_age_dx}</td>
                    <td style={{ padding: '8px 10px' }}>{c.pes_cavus_n}/40</td>
                    <td style={{ padding: '8px 10px', fontWeight: c.onion_bulb_n > 15 ? 700 : 400, color: c.onion_bulb_n > 15 ? '#4a148c' : '#333' }}>{c.onion_bulb_n}/40</td>
                    <td style={{ padding: '8px 10px' }}>{c.wheelchair_n}/40</td>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: ncvClass }}>{c.mean_ncv} m/s</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#555' }}>{g.pathognomonic?.substring(0, 60)}…</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* NCV Analysis Tab */}
      {tab === 'NCV Analysis' && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>NCV Classification & Diagnostic Approach</h2>
          <div style={{ background: '#e3f2fd', border: '1px solid #1565c0', borderRadius: 8, padding: 12, marginBottom: 16, fontSize: 13 }}>
            <b>Key Rule:</b> Measure motor NCV first → classify demyelinating (&lt;38 m/s) / intermediate (25-45 m/s) / axonal (≥38 m/s with ↓ CMAP).
            Intermediate NCV in male → test GJB1 FIRST (CMT1X). Demyelinating → MLPA PMP22 first.
            Axonal with optic atrophy → MFN2 PATHOGNOMONIC. Vocal cord + neuropathy → GDAP1 FIRST.
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginBottom: 20 }}>
            <thead>
              <tr style={{ background: '#1565c0', color: '#fff' }}>
                {['Gene', 'CMT Subtype', 'NCV Type', 'NCV Range', 'NCV Pattern', 'Pathognomonic Feature'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {NCV_TABLE.map((row, i) => {
                const bg = row.type === 'INTERMEDIATE' ? '#fce4ec' :
                           row.type === 'Axonal' ? '#e8f5e9' :
                           row.type === 'Severe Demyel' ? '#ffebee' :
                           row.type === 'Axonal OR Demyel' ? '#e0f2f1' : '#f5f5f5';
                return (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? bg : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px' }}>{row.subtype}</td>
                    <td style={{ padding: '8px 10px', fontWeight: row.type === 'INTERMEDIATE' || row.type === 'Severe Demyel' ? 700 : 400,
                      color: row.type === 'INTERMEDIATE' ? '#880e4f' : row.type === 'Severe Demyel' ? '#b71c1c' : '#333' }}>{row.type}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: row.ncv_range.includes('<10') ? '#b71c1c' : row.ncv_range.includes('25-45') ? '#880e4f' : '#333' }}>{row.ncv_range}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.pattern}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, fontWeight: row.pathognomonic.includes('PATHOGNOMONIC') ? 700 : 400,
                      color: row.pathognomonic.includes('PATHOGNOMONIC') ? '#1b5e20' : '#555' }}>{row.pathognomonic}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
            <div style={{ background: '#e3f2fd', border: '2px solid #1565c0', borderRadius: 8, padding: 12 }}>
              <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 6 }}>Demyelinating CMT1 (&lt;38 m/s)</div>
              <div style={{ fontSize: 12 }}>First test: MLPA PMP22 duplication (60-70% yield)</div>
              <div style={{ fontSize: 12 }}>If negative: sequence MPZ, GJB1, SH3TC2, PRX, NEFL</div>
              <div style={{ fontSize: 12 }}>NCV &lt;10 m/s: consider PRX (CMT4F)</div>
            </div>
            <div style={{ background: '#fce4ec', border: '2px solid #880e4f', borderRadius: 8, padding: 12 }}>
              <div style={{ fontWeight: 700, color: '#880e4f', marginBottom: 6 }}>Intermediate CMT (25-45 m/s)</div>
              <div style={{ fontSize: 12 }}>Male + intermediate NCV: test GJB1 FIRST</div>
              <div style={{ fontSize: 12 }}>Intermediate NCV = PATHOGNOMONIC for GJB1/CMT1X</div>
              <div style={{ fontSize: 12 }}>Females may have NCV in normal range</div>
            </div>
            <div style={{ background: '#e8f5e9', border: '2px solid #1b5e20', borderRadius: 8, padding: 12 }}>
              <div style={{ fontWeight: 700, color: '#1b5e20', marginBottom: 6 }}>Axonal CMT2 (≥38 m/s ↓CMAP)</div>
              <div style={{ fontSize: 12 }}>First test: MFN2 (most common CMT2)</div>
              <div style={{ fontSize: 12 }}>CMT2A + optic atrophy: MFN2 PATHOGNOMONIC</div>
              <div style={{ fontSize: 12 }}>Vocal cord + axonal: test GDAP1</div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>CMT Definitions & Critical Rules</h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
            {Object.entries(definitions.definitions || {}).map(([term, def]) => (
              <div key={term} style={{ border: '1px solid #ddd', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: '#1565c0' }}>{term}</div>
                <div style={{ fontSize: 12, color: '#555', lineHeight: 1.5 }}>{def}</div>
              </div>
            ))}
          </div>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>Critical Clinical Rules ({Object.keys(definitions.critical_clinical_rules || {}).length})</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {Object.keys(definitions.critical_clinical_rules || {}).map(rule => (
              <span key={rule} style={{
                background: rule.includes('PATHOGNOMONIC') ? '#e8f5e9' :
                            rule.includes('AVOID') ? '#ffebee' : '#e3f2fd',
                color: rule.includes('PATHOGNOMONIC') ? '#1b5e20' :
                       rule.includes('AVOID') ? '#b71c1c' : '#1565c0',
                border: `1px solid ${rule.includes('PATHOGNOMONIC') ? '#a5d6a7' :
                         rule.includes('AVOID') ? '#ef9a9a' : '#90caf9'}`,
                borderRadius: 4, padding: '2px 8px', fontSize: 11
              }}>{rule}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
