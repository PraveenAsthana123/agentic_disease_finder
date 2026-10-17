'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-taad-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'FBN1':   '#1565c0',  // deep blue    -- Marfan syndrome; aortic root; ectopia lentis; losartan
  'TGFBR1': '#880e4f',  // deep crimson -- LDS1A; bifid uvula; 4cm threshold; aggressive
  'TGFBR2': '#c62828',  // deep red     -- LDS1B; dissection <4cm PATHOGNOMONIC; branch vessel
  'ACTA2':  '#4a148c',  // deep purple  -- TAAD2; Moyamoya; livedo reticularis; iris flocculi
  'MYH11':  '#1b5e20',  // dark green   -- TAAD4; PDA PATHOGNOMONIC; BAV; reduced penetrance
  'SMAD3':  '#e65100',  // burnt orange -- LDS3/AOS; early OA PATHOGNOMONIC; 4.5cm threshold
  'COL3A1': '#b71c1c',  // darkest red  -- vEDS; spontaneous rupture ANY diameter; surgery CI
  'LOX':    '#4e342e',  // dark brown   -- FTAAD; BAV enriched; collagen crosslinking; new 2018
};

const GENE_INFO = {
  'FBN1':   { full: 'Fibrillin-1 / Marfan-MFS1 / aortic-root-ectopia-lentis-tall-stature-PATHOGNOMONIC / losartan-beta-blocker / surgery-4.5cm', locus: '15q21.1', size: '2871 aa / 320 kDa', inh: 'AD LOF' },
  'TGFBR1': { full: 'TGF-β-R1 / LDS1A / bifid-uvula-tortuosity-hypertelorism-PATHOGNOMONIC / MORE-aggressive-than-Marfan / surgery-4.0cm', locus: '9q22.33', size: '503 aa / 55 kDa', inh: 'AD LOF' },
  'TGFBR2': { full: 'TGF-β-R2 / LDS1B / dissection-at-lt4cm-PATHOGNOMONIC / branch-vessel-aneurysms / total-body-MRA-mandatory', locus: '3p24.1', size: '567 aa / 65 kDa', inh: 'AD LOF' },
  'ACTA2':  { full: 'α-SMA / TAAD2 / Moyamoya-cerebrovascular / livedo-reticularis-iris-flocculi-PATHOGNOMONIC / cerebral-MRA-mandatory', locus: '10q23.31', size: '377 aa / 42 kDa', inh: 'AD GOF' },
  'MYH11':  { full: 'SM-MHC / TAAD4 / PDA-TAAD-PATHOGNOMONIC-pair / BAV-associated / reduced-penetrance-50pct', locus: '16p13.11', size: '1972 aa / 227 kDa', inh: 'AD LOF' },
  'SMAD3':  { full: 'SMAD3 / LDS3-AOS / early-OA-PATHOGNOMONIC / aneurysm-osteoarthritis-syndrome / 4.5cm-threshold', locus: '15q22.33', size: '425 aa / 48 kDa', inh: 'AD LOF' },
  'COL3A1': { full: 'Collagen-IIIα1 / vEDS / spontaneous-rupture-ANY-diameter-PATHOGNOMONIC / surgery-ABSOLUTELY-CI-except-emergency / celiprolol-RCT', locus: '2q32.2', size: '1466 aa / 138 kDa', inh: 'AD LOF dominant-negative' },
  'LOX':    { full: 'Lysyl-oxidase / FTAAD / BAV-enriched / collagen-elastin-crosslinking / recently-described-2018', locus: '5q23.1', size: '417 aa / 48 kDa', inh: 'AD LOF' },
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
        <Badge text={`Dissection/Rupture: ${cohort.dissection_rupture_n}/40`} color='#c62828' />
        <Badge text={`Surgery: ${cohort.surgery_n}/40`} color={gene === 'COL3A1' ? '#b71c1c' : '#555'} />
        <Badge text={`Stroke: ${cohort.stroke_n}/40`} color={gene === 'ACTA2' ? '#4a148c' : '#777'} />
      </div>
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      {geneAtlas.key_drugs_ci?.length > 0 && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🚫 <b>CI / Watch:</b> {geneAtlas.key_drugs_ci.join(' · ')}
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
      <h2 style={{ color: '#1565c0' }}>{data.atlas}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle}</p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {[
          { label: 'Total Patients', val: data.total_patients, color: '#1565c0' },
          { label: 'Genes', val: data.gene_count, color: '#880e4f' },
          { label: 'Seeds', val: data.seeds, color: '#1b5e20' },
          { label: 'Marfan/LDS Genes', val: (cats.marfan_spectrum?.length || 0) + (cats.loeys_dietz_spectrum?.length || 0), color: '#c62828' },
          { label: 'Surgery CI (vEDS)', val: cats.surgery_ci_veds?.length || 0, color: '#b71c1c' },
          { label: 'Cerebrovascular Risk', val: cats.cerebrovascular_risk?.length || 0, color: '#4a148c' },
        ].map(k => (
          <div key={k.label} style={{ background: k.color, color: '#fff', borderRadius: 8, padding: '10px 18px', minWidth: 130, textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 700 }}>{k.val}</div>
            <div style={{ fontSize: 12 }}>{k.label}</div>
          </div>
        ))}
      </div>

      <div style={{ background: '#fff8e1', borderRadius: 8, padding: 14, marginBottom: 16, border: '2px solid #ff8f00' }}>
        <h3 style={{ margin: '0 0 8px', color: '#e65100' }}>🔑 Surgery Threshold Comparison</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, fontSize: 12 }}>
          {[
            { gene: 'FBN1', threshold: '4.5–5.0 cm', color: '#1565c0' },
            { gene: 'TGFBR1/2', threshold: '4.0 cm ⚡', color: '#c62828' },
            { gene: 'ACTA2', threshold: '4.5–5.0 cm', color: '#4a148c' },
            { gene: 'MYH11', threshold: '5.0 cm', color: '#1b5e20' },
            { gene: 'SMAD3', threshold: '4.5 cm', color: '#e65100' },
            { gene: 'COL3A1', threshold: '🚫 NO elective surgery', color: '#b71c1c' },
            { gene: 'LOX', threshold: '5.0 cm', color: '#4e342e' },
          ].map(t => (
            <div key={t.gene} style={{ background: t.color, color: '#fff', borderRadius: 6, padding: '6px 10px', textAlign: 'center' }}>
              <div style={{ fontWeight: 700, fontSize: 13 }}>{t.gene}</div>
              <div style={{ fontSize: 11 }}>{t.threshold}</div>
            </div>
          ))}
        </div>
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
            <span style={{ fontWeight: 700, color: '#b71c1c' }}>{rule.split(':')[0]}:</span>
            {rule.includes(':') ? rule.slice(rule.indexOf(':') + 1) : ''}
          </div>
        ))}
      </div>

      <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 12 }}>
        <h3 style={{ margin: '0 0 8px', color: '#1b5e20' }}>📊 Cohort Summary (8 × 40 patients)</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
          {(data.cohort_stats || []).map(cs => (
            <div key={cs.gene} style={{ background: GENE_COLORS[cs.gene] || '#555', color: '#fff', borderRadius: 6, padding: '8px 12px' }}>
              <div style={{ fontWeight: 700 }}>{cs.gene}</div>
              <div style={{ fontSize: 11 }}>{cs.syndrome?.substring(0, 22)}</div>
              <div style={{ fontSize: 11 }}>Age Dx: {cs.mean_age} yr</div>
              <div style={{ fontSize: 11 }}>♀ {cs.pct_female}%</div>
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
      <h2 style={{ color: '#1565c0' }}>Gene Reference Table</h2>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#1565c0', color: '#fff' }}>
              {['Gene', 'Locus', 'Size', 'Inheritance', 'Syndrome', 'Age Dx', '♀%', 'Dissect/Rupture/40', 'Surgery/40', 'Stroke/40', 'Pathognomonic'].map(h => (
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
                  <td style={{ padding: '6px 10px' }}>{d.syndrome?.substring(0, 30) || '–'}</td>
                  <td style={{ padding: '6px 10px' }}>{d.mean_age_dx} yr</td>
                  <td style={{ padding: '6px 10px' }}>{d.pct_female}%</td>
                  <td style={{ padding: '6px 10px', color: '#c62828', fontWeight: 600 }}>{d.dissection_rupture_n}</td>
                  <td style={{ padding: '6px 10px', color: g === 'COL3A1' ? '#b71c1c' : '#333', fontWeight: g === 'COL3A1' ? 700 : 400 }}>
                    {g === 'COL3A1' ? '0 (CI)' : d.surgery_n}
                  </td>
                  <td style={{ padding: '6px 10px', color: g === 'ACTA2' ? '#4a148c' : '#333', fontWeight: g === 'ACTA2' ? 700 : 400 }}>{d.stroke_n}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11 }}>{d.pathognomonic?.substring(0, 60) || '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 20, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {Object.entries(breakdown.summary || {}).map(([k, v]) => (
          <div key={k} style={{ border: '1px solid #e0e0e0', borderRadius: 6, padding: '8px 12px' }}>
            <span style={{ fontWeight: 600, color: '#1565c0', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}: </span>
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
      <h2 style={{ color: '#1565c0' }}>Clinical Atlas — 8-Gene Reference</h2>
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
      <h2 style={{ color: '#1565c0' }}>Definitions & Critical Clinical Rules</h2>
      <div style={{ marginBottom: 24 }}>
        <h3 style={{ color: '#1565c0' }}>Glossary</h3>
        {Object.entries(defs.definitions || {}).map(([term, def]) => (
          <div key={term} style={{ borderBottom: '1px solid #e0e0e0', padding: '8px 0' }}>
            <span style={{ fontWeight: 700, color: '#1565c0' }}>{term.replace(/_/g, ' ')}: </span>
            <span style={{ fontSize: 13 }}>{def}</span>
          </div>
        ))}
      </div>
      <div>
        <h3 style={{ color: '#b71c1c' }}>🚨 Critical Clinical Rules</h3>
        {Object.entries(defs.critical_clinical_rules || {}).map(([key, rule]) => (
          <div key={key} style={{ background: '#fff3e0', borderRadius: 6, padding: '8px 12px', marginBottom: 8 }}>
            <div style={{ fontWeight: 700, color: '#e65100', fontSize: 12 }}>{key.replace(/-/g, ' ')}</div>
            <div style={{ fontSize: 13 }}>{rule}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HereditaryTAADAtlasPage() {
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
      <div style={{ background: 'linear-gradient(135deg, #1565c0, #4a148c)', color: '#fff', borderRadius: 10, padding: '20px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>🩸 Hereditary TAAD Atlas</h1>
        <p style={{ margin: '6px 0 0', opacity: 0.85, fontSize: 14 }}>
          Complete 8-Gene Reference — FBN1 · TGFBR1 · TGFBR2 · ACTA2 · MYH11 · SMAD3 · COL3A1 · LOX | 320-patient aggregate (8×40) | seeds 3758–3765
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(i)}
            style={{
              padding: '8px 18px', border: 'none', cursor: 'pointer',
              background: tab === i ? '#1565c0' : 'transparent',
              color: tab === i ? '#fff' : '#1565c0',
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
