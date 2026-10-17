'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cerebrovascular-disease-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'NOTCH3': '#4a148c',  // deep purple    -- CADASIL; GOM-PATHOGNOMONIC; lacunar 40-50yr
  'COL4A1': '#1565c0',  // deep blue      -- Porencephaly-PATHOGNOMONIC; ICH young; ANTICOAG-CI
  'COL4A2': '#0277bd',  // medium blue    -- ICH-dominant; panel-together-COL4A1
  'ACTA2':  '#b71c1c',  // deep red       -- Moyamoya-TRIAD-PATHOGNOMONIC; iris flocculi
  'HTRA1':  '#1b5e20',  // dark green     -- CARASIL; alopecia-PATHOGNOMONIC; spondylosis
  'TREX1':  '#e65100',  // burnt orange   -- AGS1; IFN-signature-PATHOGNOMONIC; baricitinib
  'CST3':   '#880e4f',  // deep crimson   -- HCHWA-I; ICH 20-30yr; ANTICOAG-ABSOLUTELY-CI
  'APP':    '#4e342e',  // dark brown     -- HCHWA-D; Dutch/Iowa; CAA; ANTICOAG-ABSOLUTELY-CI
};

const GENE_INFO = {
  'NOTCH3': { full: 'EGF-repeat-cysteine / CADASIL / GOM-PATHOGNOMONIC / lacunar-40-50yr / migraine-aura-30yr / NO-DMT', locus: '19p13.2', size: '2321 aa / 280 kDa', inh: 'AD GOF' },
  'COL4A1': { full: 'Type-IV-collagen-alpha1 / Porencephaly-PATHOGNOMONIC / ICH-young / renal-cysts / cataracts / ANTICOAG-CI', locus: '13q34', size: '1454 aa / 160 kDa', inh: 'AD LOF dominant-negative' },
  'COL4A2': { full: 'Type-IV-collagen-alpha2 / ICH-dominant / less-porencephaly / panel-COL4A1-COL4A2-together / ANTICOAG-CI', locus: '13q34', size: '1712 aa / 167 kDa', inh: 'AD LOF' },
  'ACTA2':  { full: 'alpha-SMA / Moyamoya-TRIAD-PATHOGNOMONIC / iris-flocculi / livedo-reticularis / cerebral-MRA-MANDATORY / TAAD-overlap', locus: '10q23.31', size: '377 aa / 42 kDa', inh: 'AD GOF' },
  'HTRA1':  { full: 'HtrA-serine-peptidase / CARASIL / alopecia-PATHOGNOMONIC / spondylosis-before-stroke / Japanese-founder / AD-milder', locus: '10q26.13', size: '480 aa / 51 kDa', inh: 'AR biallelic LOF (CARASIL) / AD LOF (milder)' },
  'TREX1':  { full: '3-prime-exonuclease / AGS1 / IFN-signature-PATHOGNOMONIC / baricitinib-FDA2023 / RVCL-AD', locus: '3p21.31', size: '314 aa / 33 kDa', inh: 'AR/AD LOF' },
  'CST3':   { full: 'Cystatin-C / HCHWA-I-Icelandic / L68Q-founder / CAA / ICH-20-30yr / ANTICOAGULATION-ABSOLUTELY-CI', locus: '20p11.21', size: '146 aa / 13 kDa', inh: 'AD GOF amyloidogenic' },
  'APP':    { full: 'Amyloid-precursor-protein / HCHWA-D-Dutch-Iowa / Abeta-CAA / ICH-40-60yr / early-Alzheimer / ANTICOAGULATION-ABSOLUTELY-CI', locus: '21q21.3', size: '770 aa / 87 kDa', inh: 'AD GOF amyloidogenic' },
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
        <Badge text={`Stroke: ${cohort.stroke_n}/40`} color={bg} />
        <Badge text={`Dementia: ${cohort.dementia_n}/40`} color='#6a1b9a' />
        <Badge text={`MRI+: ${cohort.mri_lesion_n}/40`} color='#555' />
      </div>
      {cohort.anticoag_contraindicated && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#b71c1c' }}>
          🚫 ANTICOAGULATION CONTRAINDICATED — hemorrhagic cerebrovascular phenotype
        </div>
      )}
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      {geneAtlas.key_drugs_ci?.length > 0 && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          💊 <b>Key Rules:</b> {geneAtlas.key_drugs_ci.join(' · ')}
        </div>
      )}
      <div style={{ fontSize: 12, color: '#444' }}><b>Inheritance:</b> {info?.inh}</div>
      <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>{info?.full}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div>Loading...</div>;
  const cats = data.clinical_categories || {};
  return (
    <div>
      <h2 style={{ color: '#4a148c' }}>{data.atlas}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle}</p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {[
          { label: 'Total Patients', val: data.total_patients, color: '#4a148c' },
          { label: 'Genes', val: data.gene_count, color: '#1565c0' },
          { label: 'Seeds', val: data.seeds, color: '#1b5e20' },
          { label: 'SVD Genes', val: cats.hereditary_svd_cadasil_carasil?.length || 0, color: '#880e4f' },
          { label: 'CAA Genes', val: cats.cerebral_amyloid_angiopathy_hemorrhagic?.length || 0, color: '#b71c1c' },
          { label: 'AntiCoag CI', val: cats.anticoagulation_absolutely_contraindicated?.length || 0, color: '#c62828' },
        ].map(k => (
          <div key={k.label} style={{ background: k.color, color: '#fff', borderRadius: 8, padding: '10px 18px', minWidth: 120, textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 700 }}>{k.val}</div>
            <div style={{ fontSize: 12 }}>{k.label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
        {Object.entries(cats).map(([cat, genes]) => (
          <div key={cat} style={{ border: '1px solid #e0e0e0', borderRadius: 8, padding: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, color: '#333', textTransform: 'capitalize' }}>
              {cat.replace(/_/g, ' ')}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {genes.map(g => (
                <span key={g} style={{ background: GENE_COLORS[g] || '#666', color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12 }}>{g}</span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div style={{ background: '#fff3e0', borderRadius: 8, padding: 16, marginBottom: 16 }}>
        <h3 style={{ margin: '0 0 12px', color: '#e65100' }}>🚨 Key Clinical Rules</h3>
        {(data.key_clinical_rules || []).map((rule, i) => (
          <div key={i} style={{ borderBottom: '1px solid #ffe0b2', padding: '6px 0', fontSize: 13 }}>
            <span style={{ fontWeight: 700, color: '#4a148c' }}>{rule.split(':')[0]}:</span>
            {rule.includes(':') ? rule.slice(rule.indexOf(':') + 1) : ''}
          </div>
        ))}
      </div>

      <div style={{ background: '#e8eaf6', borderRadius: 8, padding: 12 }}>
        <h3 style={{ margin: '0 0 8px', color: '#4a148c' }}>📊 Cohort Summary (8 × 40 patients)</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
          {(data.cohort_stats || []).map(cs => (
            <div key={cs.gene} style={{ background: GENE_COLORS[cs.gene] || '#555', color: '#fff', borderRadius: 6, padding: '8px 12px' }}>
              <div style={{ fontWeight: 700 }}>{cs.gene}</div>
              <div style={{ fontSize: 11 }}>{cs.syndrome?.substring(0, 25)}</div>
              <div style={{ fontSize: 11 }}>Age Dx: {cs.mean_age} yr</div>
              <div style={{ fontSize: 11 }}>♀ {cs.pct_female}%</div>
              {cs.anticoag_ci && <div style={{ fontSize: 10, fontWeight: 700 }}>🚫 ANTICOAG CI</div>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function GeneTableTab({ breakdown }) {
  if (!breakdown) return <div>Loading...</div>;
  const genes = Object.keys(breakdown.gene_data || {});
  return (
    <div>
      <h2 style={{ color: '#4a148c' }}>Gene Reference Table</h2>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#4a148c', color: '#fff' }}>
              {['Gene', 'Locus', 'Size', 'Inheritance', 'Syndrome', 'Age Dx', '♀%', 'Stroke/40', 'Dementia/40', 'AntiCoag CI', 'Pathognomonic'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes.map((g, idx) => {
              const d = breakdown.gene_data[g] || {};
              const info = GENE_INFO[g] || {};
              return (
                <tr key={g} style={{ background: idx % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={{ background: GENE_COLORS[g] || '#666', color: '#fff', borderRadius: 4, padding: '2px 7px', fontWeight: 700 }}>{g}</span>
                  </td>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{info.locus || '–'}</td>
                  <td style={{ padding: '6px 10px' }}>{info.size || '–'}</td>
                  <td style={{ padding: '6px 10px' }}>{d.inheritance || '–'}</td>
                  <td style={{ padding: '6px 10px' }}>{d.syndrome?.split(' / ')[0] || '–'}</td>
                  <td style={{ padding: '6px 10px' }}>{d.mean_age_dx} yr</td>
                  <td style={{ padding: '6px 10px' }}>{d.pct_female}%</td>
                  <td style={{ padding: '6px 10px' }}>{d.stroke_n}</td>
                  <td style={{ padding: '6px 10px' }}>{d.dementia_n}</td>
                  <td style={{ padding: '6px 10px', color: d.anticoag_contraindicated ? '#c62828' : '#2e7d32', fontWeight: 700 }}>
                    {d.anticoag_contraindicated ? '🚫 YES' : '✓ No'}
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11 }}>{d.pathognomonic?.substring(0, 55) || '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 20, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {Object.entries(breakdown.summary || {}).map(([k, v]) => (
          <div key={k} style={{ border: '1px solid #e0e0e0', borderRadius: 6, padding: '8px 12px' }}>
            <span style={{ fontWeight: 600, color: '#4a148c', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}: </span>
            <span style={{ fontSize: 12 }}>{Array.isArray(v) ? v.join(', ') : String(v)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ breakdown }) {
  if (!breakdown) return <div>Loading...</div>;
  return (
    <div>
      <h2 style={{ color: '#4a148c' }}>Clinical Atlas — 8-Gene Reference</h2>
      {(breakdown.atlas_genes || []).map(g => (
        <GeneCard
          key={g.gene}
          gene={g.gene}
          color={GENE_COLORS[g.gene] || '#555'}
          info={GENE_INFO[g.gene]}
          breakdown={breakdown}
        />
      ))}
    </div>
  );
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div>Loading...</div>;
  return (
    <div>
      <h2 style={{ color: '#4a148c' }}>Definitions & Critical Clinical Rules</h2>
      <div style={{ marginBottom: 24 }}>
        <h3 style={{ color: '#1565c0' }}>Glossary</h3>
        {Object.entries(defs.definitions || {}).map(([term, def]) => (
          <div key={term} style={{ borderBottom: '1px solid #e0e0e0', padding: '8px 0' }}>
            <span style={{ fontWeight: 700, color: '#4a148c' }}>{term.replace(/_/g, ' ')}: </span>
            <span style={{ fontSize: 13 }}>{def}</span>
          </div>
        ))}
      </div>
      <div>
        <h3 style={{ color: '#b71c1c' }}>🚨 Critical Clinical Rules</h3>
        {Object.entries(defs.critical_clinical_rules || {}).map(([key, rule]) => (
          <div key={key} style={{ background: '#f3e5f5', borderRadius: 6, padding: '8px 12px', marginBottom: 8 }}>
            <div style={{ fontWeight: 700, color: '#4a148c', fontSize: 12 }}>{key.replace(/-/g, ' ')}</div>
            <div style={{ fontSize: 13 }}>{rule}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HereditoryCerebrovascularDiseaseAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div style={{ padding: 32, color: 'red' }}>Error: {err}</div>;

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ background: 'linear-gradient(135deg, #4a148c, #1565c0)', color: '#fff', borderRadius: 10, padding: '20px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>🧠 Hereditary Cerebrovascular Disease Atlas</h1>
        <p style={{ margin: '6px 0 0', opacity: 0.85, fontSize: 14 }}>
          Complete 8-Gene Reference — NOTCH3 · COL4A1 · COL4A2 · ACTA2 · HTRA1 · TREX1 · CST3 · APP | 320-patient aggregate (8×40) | seeds 3774–3781
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(i)}
            style={{
              padding: '8px 18px', border: 'none', cursor: 'pointer',
              background: tab === i ? '#4a148c' : 'transparent',
              color: tab === i ? '#fff' : '#4a148c',
              borderRadius: '6px 6px 0 0',
              fontWeight: tab === i ? 700 : 400,
              fontSize: 14,
            }}
          >{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <GeneTableTab breakdown={breakdown} />}
      {tab === 2 && <ClinicalAtlasTab breakdown={breakdown} />}
      {tab === 3 && <DefinitionsTab defs={defs} />}
    </div>
  );
}
