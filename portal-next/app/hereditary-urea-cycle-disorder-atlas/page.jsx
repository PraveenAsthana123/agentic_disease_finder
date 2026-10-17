'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-urea-cycle-disorder-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'OTC':      '#4a148c',  // deep purple    -- X-linked most common; citrulline-low + orotic-high PATHOGNOMONIC
  'ASS1':     '#1a237e',  // deep navy      -- CTLN1; citrulline >1000 μmol/L PATHOGNOMONIC
  'ASL':      '#880e4f',  // deep crimson   -- argininosuccinic acid PATHOGNOMONIC; liver-brain dissociation
  'ARG1':     '#b71c1c',  // dark red       -- hyperargininemia; spastic diplegia PATHOGNOMONIC; mild NH3
  'CPS1':     '#1b5e20',  // dark green     -- most severe neonatal; carglumic acid NOT effective
  'NAGS':     '#e65100',  // burnt orange   -- ONLY UCD carglumic acid Carbaglu CURATIVE
  'SLC25A13': '#004d40',  // dark teal      -- citrin CTLN2; alcohol+carb TRIGGER PATHOGNOMONIC; Japanese founder
  'SLC25A15': '#37474f',  // dark blue-grey -- HHH syndrome; homocitrulline PATHOGNOMONIC; spastic paraparesis
};

const GENE_INFO = {
  'OTC':      { full: 'Ornithine-transcarbamylase / X-linked-most-common-UCD / citrulline-LOW+orotic-HIGH-PATHOGNOMONIC / RAVICTI-nitrogen-scavenger / fasting-ABSOLUTELY-AVOID / LTx-curative', locus: 'Xp11.4', size: '354 aa / 40 kDa', inh: 'X-linked' },
  'ASS1':     { full: 'Argininosuccinate-synthetase-1 / CTLN1-citrullinemia-type-1 / citrulline-gt1000-micromolL-PATHOGNOMONIC / arginine-MANDATORY / LTx-curative', locus: '9q34.11', size: '412 aa / 47 kDa', inh: 'AR' },
  'ASL':      { full: 'Argininosuccinate-lyase / argininosuccinic-aciduria / argininosuccinic-acid-PATHOGNOMONIC / liver-brain-dissociation / hypertension-75pct / arginine-CRITICAL', locus: '7q11.21', size: '464 aa / 52 kDa', inh: 'AR' },
  'ARG1':     { full: 'Arginase-1 / hyperargininemia / MILD-NH3-KEY-DISTINCTION / spastic-diplegia+ID-PATHOGNOMONIC / arginine-gt200-diagnostic / low-arginine-diet', locus: '6q23.2', size: '322 aa / 35 kDa', inh: 'AR' },
  'CPS1':     { full: 'CPS1-deficiency / most-severe-neonatal / citrulline-low+orotic-NORMAL / carglumic-acid-NOT-effective-DISTINGUISH-NAGS / LTx-curative', locus: '2q34', size: '1500 aa / 165 kDa', inh: 'AR' },
  'NAGS':     { full: 'N-acetylglutamate-synthase / ONLY-UCD-carglumic-acid-Carbaglu-FDA2010-CURATIVE / citrulline-low+orotic-normal-same-CPS1 / carglumic-response-PATHOGNOMONIC', locus: '17q21.31', size: '534 aa / 58 kDa', inh: 'AR' },
  'SLC25A13': { full: 'Citrin-AGC2 / CTLN2-citrin-deficiency / Japanese-founder-exon16-ins3kb / adult-onset / alcohol+high-carb-ABSOLUTELY-TRIGGER / MCT-diet / LTx-curative', locus: '7q21.3', size: '675 aa / 74 kDa', inh: 'AR' },
  'SLC25A15': { full: 'Ornithine-carrier-ORNT1 / HHH-syndrome / homocitrulline-urine-PATHOGNOMONIC / spastic-paraparesis / ornithine-restriction+citrulline-supplement', locus: '13q14.11', size: '301 aa / 33 kDa', inh: 'AR' },
};

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
        <Badge text={`Hyperammonemia: ${cohort.hyperammonemia_crisis_n}/40`} color='#6a1b9a' />
        <Badge text={`LTx: ${cohort.liver_transplant_n}/40`} color='#1b5e20' />
        <Badge text={`Neuro: ${cohort.neurological_n}/40`} color='#37474f' />
      </div>
      {gene === 'OTC' && (
        <div style={{ fontSize: 12, background: '#ede7f6', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700 }}>
          🧬 <b>X-linked:</b> Citrulline LOW + orotic acid HIGH = PATHOGNOMONIC OTC pattern. Female carriers {cohort.female_symptomatic_n}/40 symptomatic.
        </div>
      )}
      {gene === 'ASS1' && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700 }}>
          🔬 <b>CTLN1 PATHOGNOMONIC:</b> Plasma citrulline &gt;1000 μmol/L. Urine orotic acid NORMAL (vs OTC high).
        </div>
      )}
      {gene === 'ASL' && (cohort.hypertension_n || 0) > 0 && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#880e4f' }}>
          🩺 <b>Hypertension (NO deficiency):</b> {cohort.hypertension_n}/40 — Liver-brain dissociation: LTx corrects NH3 NOT neuro.
        </div>
      )}
      {gene === 'ARG1' && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#b71c1c' }}>
          🦵 <b>Spastic diplegia + ID:</b> {cohort.spastic_paraparesis_n}/40 — MILD NH3 (KEY DISTINCTION). Low-arginine diet prevents progression.
        </div>
      )}
      {gene === 'CPS1' && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#1b5e20' }}>
          ⚠️ <b>Carglumic acid NOT effective:</b> {cohort.pathognomonic_marker_n}/40 confirmed CPS1. Citrulline LOW + orotic NORMAL. DISTINGUISH from NAGS.
        </div>
      )}
      {gene === 'NAGS' && cohort.carglumic_acid_n > 0 && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#e65100' }}>
          💊 <b>Carglumic acid CURATIVE:</b> {cohort.carglumic_acid_n}/40 — ONLY UCD with pharmacological cure. Response = PATHOGNOMONIC for NAGS.
        </div>
      )}
      {gene === 'SLC25A13' && cohort.alcohol_trigger_n > 0 && (
        <div style={{ fontSize: 12, background: '#e0f2f1', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#004d40' }}>
          🍶 <b>Alcohol trigger PATHOGNOMONIC:</b> {cohort.alcohol_trigger_n}/40 — Japanese founder. IV dextrose also triggers — ALERT ICU/anesthesia.
        </div>
      )}
      {gene === 'SLC25A15' && (
        <div style={{ fontSize: 12, background: '#eceff1', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#37474f' }}>
          🧪 <b>Homocitrulline urine PATHOGNOMONIC:</b> {cohort.spastic_paraparesis_n}/40 spastic paraparesis. HHH triad.
        </div>
      )}
      <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>
        <b>Key therapy:</b> {geneAtlas.key_therapy || ''}
      </div>
      <div style={{ fontSize: 12, color: '#333', marginTop: 4, background: '#f5f5f5', padding: '4px 8px', borderRadius: 4 }}>
        <b>Pathognomonic:</b> {geneAtlas.pathognomonic || ''}
      </div>
    </div>
  );
}

export default function HUCDAtlasPage() {
  const [tab, setTab] = useState('Overview');
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
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const genes = overview?.genes || Object.keys(GENE_INFO);

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        🧬 Hereditary Urea Cycle Disorder Atlas
      </h1>
      <p style={{ fontSize: 13, color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Reference: OTC-ASS1-ASL-ARG1-CPS1-NAGS-SLC25A13-SLC25A15 ·
        320 patients (8×40) · Seeds 3806-3813
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #ddd', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#4a148c' : '#f0f0f0',
            color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <p>Loading atlas data…</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ background: '#f3e5f5', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <b>Atlas:</b> {overview.atlas} · <b>Patients:</b> {overview.total_patients} ·
            <b> Seeds:</b> {overview.seeds}
            <p style={{ marginTop: 8, fontSize: 13 }}>{overview.description}</p>
          </div>
          <h2 style={{ fontSize: 16, marginBottom: 12 }}>Gene Overview — 8 Urea Cycle Genes</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
            {genes.map(gene => {
              const s = overview.gene_summaries?.find(g => g.gene === gene) || {};
              const col = GENE_COLORS[gene] || '#555';
              return (
                <div key={gene} style={{ border: `2px solid ${col}`, borderRadius: 8, padding: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    <div style={{ background: col, color: '#fff', borderRadius: 4, padding: '4px 10px', fontWeight: 700 }}>{gene}</div>
                    <span style={{ fontSize: 11, color: '#666' }}>{s.locus} · {s.protein_size?.split('/')[0]?.trim()}</span>
                  </div>
                  <div style={{ fontSize: 12 }}>{s.syndrome?.split('/').slice(0,3).join(' / ')}</div>
                  <div style={{ fontSize: 11, color: '#555', marginTop: 4 }}>
                    <b>Rx:</b> {s.key_therapy?.split('—')[0]?.trim()}
                  </div>
                  <div style={{ fontSize: 11, color: '#333', marginTop: 4, background: '#fafafa', padding: '3px 6px', borderRadius: 4 }}>
                    ⚡ {s.pathognomonic?.split(';')[0]}
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ marginTop: 20, background: '#fff3e0', borderRadius: 8, padding: 14 }}>
            <b>⚠️ Cohort Summary (320 patients)</b>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8, fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#4a148c', color: '#fff' }}>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Gene</th>
                  <th style={{ padding: '4px 8px' }}>Mean Age Dx</th>
                  <th style={{ padding: '4px 8px' }}>Hyperammonemia</th>
                  <th style={{ padding: '4px 8px' }}>LTx</th>
                  <th style={{ padding: '4px 8px' }}>Neuro</th>
                </tr>
              </thead>
              <tbody>
                {genes.map((gene, i) => {
                  const c = overview.cohort_summary?.[gene] || {};
                  return (
                    <tr key={gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff' }}>
                      <td style={{ padding: '4px 8px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'center' }}>{c.mean_age_dx_yr} yr</td>
                      <td style={{ padding: '4px 8px', textAlign: 'center' }}>{c.hyperammonemia_crisis_n}/40</td>
                      <td style={{ padding: '4px 8px', textAlign: 'center' }}>{c.liver_transplant_n}/40</td>
                      <td style={{ padding: '4px 8px', textAlign: 'center' }}>{c.neurological_n}/40</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <h2 style={{ fontSize: 16, marginBottom: 12 }}>Gene Reference Table</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#4a148c', color: '#fff' }}>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>Gene</th>
                  <th style={{ padding: '6px 8px' }}>Locus</th>
                  <th style={{ padding: '6px 8px' }}>Size</th>
                  <th style={{ padding: '6px 8px' }}>Inheritance</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>Syndrome / Pathognomonic</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>Key Therapy</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.atlas_genes || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f3e5f5' : '#fff' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'center' }}>{g.locus}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'center' }}>{GENE_INFO[g.gene]?.size}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'center' }}>{g.inheritance?.split('(')[0]?.trim()}</td>
                    <td style={{ padding: '6px 8px' }}>
                      <div style={{ fontWeight: 600 }}>{g.syndrome?.split('/')[0]}</div>
                      <div style={{ fontSize: 11, color: '#555', marginTop: 2 }}>{g.pathognomonic?.split(';')[0]}</div>
                    </td>
                    <td style={{ padding: '6px 8px', fontSize: 11 }}>{g.key_therapy?.split('—')[0]?.trim()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <h2 style={{ fontSize: 16, marginBottom: 12 }}>Clinical Atlas — 8 Genes × 40 Patients</h2>
          {genes.map(gene => (
            <GeneCard
              key={gene}
              gene={gene}
              color={GENE_COLORS[gene] || '#555'}
              info={GENE_INFO[gene]}
              breakdown={breakdown}
            />
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h2 style={{ fontSize: 16, marginBottom: 12 }}>Definitions &amp; Critical Clinical Rules</h2>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Definitions</h3>
          {Object.entries(definitions.definitions || {}).map(([term, def]) => (
            <div key={term} style={{ marginBottom: 12, padding: 12, background: '#f3e5f5', borderRadius: 8 }}>
              <div style={{ fontWeight: 700, marginBottom: 4, color: '#4a148c' }}>{term}</div>
              <div style={{ fontSize: 13 }}>{def}</div>
            </div>
          ))}
          <h3 style={{ fontSize: 14, margin: '20px 0 8px' }}>Critical Clinical Rules</h3>
          {Object.entries(definitions.critical_clinical_rules || {}).map(([rule, text]) => (
            <div key={rule} style={{ marginBottom: 12, padding: 12, background: '#fff3e0', borderRadius: 8, borderLeft: '4px solid #e65100' }}>
              <div style={{ fontWeight: 700, marginBottom: 4, color: '#e65100', fontSize: 12 }}>{rule}</div>
              <div style={{ fontSize: 13 }}>{text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
