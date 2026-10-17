'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cardiomyopathy-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MYH7':   '#b71c1c',  // deep red          -- HCM 35-50%; R403Q PATHOGNOMONIC; mavacamten
  'MYBPC3': '#1565c0',  // deep blue          -- HCM 30-40%; late-onset 70yr; mavacamten FDA2022
  'LMNA':   '#1b5e20',  // dark green         -- DCM; AV block PATHOGNOMONIC; VT storm; Wahbi ICD
  'PLN':    '#4a148c',  // deep purple        -- R14del Dutch founder; DCM+ARVC; ICD MANDATORY
  'FLNC':   '#e65100',  // burnt orange       -- HCM+DCM+ARVC; truncating ICD preserved EF
  'RBM20':  '#006064',  // dark teal          -- R634W aggressive young DCM; highest transplant
  'MYH6':   '#880e4f',  // deep crimson       -- DCM+ASD+SSS; excellent CRT response
  'TTN':    '#4e342e',  // dark brown         -- DCM 25%; incomplete penetrance; PPCM enriched
};

const GENE_INFO = {
  'MYH7':   { full: 'beta-MHC / HCM-35-50pct / R403Q-PATHOGNOMONIC / mavacamten-disopyramide / LVOTO', locus: '14q11.2', size: '1935 aa / 223 kDa', inh: 'AD GOF' },
  'MYBPC3': { full: 'MyBP-C / HCM-30-40pct / late-onset-70yr / frameshift-haploinsufficiency / mavacamten-FDA2022', locus: '11p11.2', size: '1274 aa / 141 kDa', inh: 'AD LOF' },
  'LMNA':   { full: 'Lamin-A/C / DCM-AV-Block-PATHOGNOMONIC / VT-Storm / ICD-before-LVEF35 / Wahbi-score / EDMD-LGMD', locus: '1q22', size: '664 aa / 74 kDa', inh: 'AD LOF' },
  'PLN':    { full: 'Phospholamban-52aa / R14del-Dutch-Founder-PATHOGNOMONIC / DCM-ARVC-overlap / ICD-MANDATORY / SERCA2a', locus: '6q22.31', size: '52 aa / 6 kDa', inh: 'AD LOF dominant-negative' },
  'FLNC':   { full: 'Filamin-C / HCM-DCM-ARVC-spectrum / truncating-highest-SCD / ICD-preserved-EF / distal-myopathy', locus: '7q32.1', size: '2725 aa / 291 kDa', inh: 'AD LOF truncating' },
  'RBM20':  { full: 'RNA-binding-motif-20 / R634W-HOTSPOT / DCM-aggressive-young / biventricular-fibrosis / highest-transplant', locus: '10q25.2', size: '1227 aa / 145 kDa', inh: 'AD GOF' },
  'MYH6':   { full: 'alpha-MHC / DCM-ASD-SSS / excellent-CRT-response / atrial-involvement-prominent', locus: '14q11.2', size: '1939 aa / 224 kDa', inh: 'AD LOF' },
  'TTN':    { full: 'Titin-largest-protein / DCM-25pct / incomplete-penetrance / PPCM-enriched / HCM-modifier', locus: '2q31.2', size: '34350 aa / 3800 kDa', inh: 'AD LOF truncating' },
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
        <Badge text={`ICD: ${cohort.icd_implanted_n}/40`} color={bg} />
        <Badge text={`Transplant: ${cohort.transplant_n}/40`} color='#c62828' />
        <Badge text={`HF Admissions: ${cohort.hf_admission_n}/40`} color='#555' />
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
      <h2 style={{ color: '#b71c1c' }}>{data.atlas}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle}</p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {[
          { label: 'Total Patients', val: data.total_patients, color: '#b71c1c' },
          { label: 'Genes', val: data.gene_count, color: '#1565c0' },
          { label: 'Seeds', val: data.seeds, color: '#1b5e20' },
          { label: 'HCM Genes', val: cats.hcm_genes?.length || 0, color: '#880e4f' },
          { label: 'DCM Genes', val: cats.dcm_genes?.length || 0, color: '#4a148c' },
          { label: 'Early ICD Genes', val: cats.early_icd_non_standard?.length || 0, color: '#006064' },
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
              <div style={{ fontSize: 11 }}>{cs.syndrome?.substring(0, 25)}</div>
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
      <h2 style={{ color: '#b71c1c' }}>Gene Reference Table</h2>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#b71c1c', color: '#fff' }}>
              {['Gene', 'Locus', 'Size', 'Inheritance', 'Syndrome', 'Age Dx', '♀%', 'ICD/40', 'Transplant/40', 'HF Admit/40', 'Pathognomonic'].map(h => (
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
                  <td style={{ padding: '6px 10px' }}>{d.icd_implanted_n}</td>
                  <td style={{ padding: '6px 10px', color: '#c62828', fontWeight: 600 }}>{d.transplant_n}</td>
                  <td style={{ padding: '6px 10px' }}>{d.hf_admission_n}</td>
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
            <span style={{ fontWeight: 600, color: '#b71c1c', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}: </span>
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
      <h2 style={{ color: '#b71c1c' }}>Clinical Atlas — 8-Gene Reference</h2>
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
      <h2 style={{ color: '#b71c1c' }}>Definitions & Critical Clinical Rules</h2>
      <div style={{ marginBottom: 24 }}>
        <h3 style={{ color: '#1565c0' }}>Glossary</h3>
        {Object.entries(defs.definitions || {}).map(([term, def]) => (
          <div key={term} style={{ borderBottom: '1px solid #e0e0e0', padding: '8px 0' }}>
            <span style={{ fontWeight: 700, color: '#b71c1c' }}>{term.replace(/_/g, ' ')}: </span>
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

export default function HereditaryCardiomyopathyAtlasPage() {
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
      <div style={{ background: 'linear-gradient(135deg, #b71c1c, #880e4f)', color: '#fff', borderRadius: 10, padding: '20px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>❤️ Hereditary Cardiomyopathy Atlas</h1>
        <p style={{ margin: '6px 0 0', opacity: 0.85, fontSize: 14 }}>
          Complete 8-Gene Reference — MYH7 · MYBPC3 · LMNA · PLN · FLNC · RBM20 · MYH6 · TTN | 320-patient aggregate (8×40) | seeds 3750–3757
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(i)}
            style={{
              padding: '8px 18px', border: 'none', cursor: 'pointer',
              background: tab === i ? '#b71c1c' : 'transparent',
              color: tab === i ? '#fff' : '#b71c1c',
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
