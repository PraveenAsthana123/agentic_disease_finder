'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-inherited-arrhythmia-syndrome-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'KCNQ1':   '#1a237e',  // deep navy          -- IKs; LQT1-swim-trigger; JLNS-deafness
  'KCNH2':   '#4a148c',  // deep purple        -- IKr/hERG; LQT2-auditory-trigger; SQT1
  'SCN5A':   '#b71c1c',  // deep red           -- Nav1.5; LQT3/Brugada/SSS/PCCD; fever-CI
  'RYR2':    '#1b5e20',  // forest green       -- RyR2; CPVT1; bidirectional-VT-PATHOGNOMONIC
  'PKP2':    '#bf360c',  // deep orange-red    -- Plakophilin-2; ARVC1; epsilon-wave
  'DSP':     '#006064',  // dark teal          -- Desmoplakin; ARVC8-LV-dominant; Carvajal
  'CACNA1C': '#880e4f',  // deep crimson       -- Cav1.2; LQT8-Timothy; 2:1-AV-block-PATHOGNOMONIC
  'HCN4':    '#4e342e',  // dark brown         -- If-pacemaker; SSS2; ivermectin-ABSOLUTE-CI
};

const GENE_INFO = {
  'KCNQ1':   { full: 'Kv7.1 / IKs / LQT1-swim-exertion-trigger-PATHOGNOMONIC / JLNS-deafness-PATHOGNOMONIC / SQT2', locus: '11p15.5', size: '676 aa / 75 kDa', inh: 'AD/AR LOF' },
  'KCNH2':   { full: 'hERG / IKr / LQT2-auditory-arousal-trigger-PATHOGNOMONIC / SQT1 / hERG-block=drug-induced-QT', locus: '7q36.1', size: '1159 aa / 127 kDa', inh: 'AD LOF' },
  'SCN5A':   { full: 'Nav1.5 / LQT3-nocturnal / Brugada-fever-trigger-PATHOGNOMONIC / SSS / PCCD / coved-ST-PATHOGNOMONIC', locus: '3p22.2', size: '2016 aa / 220 kDa', inh: 'AD LOF/GOF' },
  'RYR2':    { full: 'RyR2 / CPVT1-bidirectional-VT-PATHOGNOMONIC / normal-resting-ECG-PATHOGNOMONIC / ARVC2 / nadolol-flecainide', locus: '1q43', size: '4967 aa / 560 kDa', inh: 'AD GOF' },
  'PKP2':    { full: 'Plakophilin-2 / ARVC1-most-common / epsilon-wave-PATHOGNOMONIC / RV-fat-fibrosis / sport-ABSOLUTE-CI', locus: '12p11.21', size: '897 aa / 97 kDa', inh: 'AD LOF' },
  'DSP':     { full: 'Desmoplakin / ARVC8-LV-dominant-PATHOGNOMONIC / Carvajal-wooly-hair-keratoderma-PATHOGNOMONIC / highest-SCD-risk', locus: '6p24.3', size: '2871 aa / 332 kDa', inh: 'AD LOF / AR (Carvajal)' },
  'CACNA1C': { full: 'Cav1.2 / LQT8-Timothy-EXTREMELY-RARE / 2:1-AV-block-syndactyly-PATHOGNOMONIC / multisystem-GOF', locus: '12p13.33', size: '2221 aa / 250 kDa', inh: 'AD GOF (de novo)' },
  'HCN4':    { full: 'If-pacemaker-current / SSS2-hereditary-bradycardia / pacemaker-MANDATORY / LVNC-overlap / ivermectin-ABSOLUTE-CI', locus: '15q24.1', size: '1208 aa / 130 kDa', inh: 'AD LOF' },
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
        <Badge text={`SCD/VF: ${cohort.scd_or_resuscitated_n}/40`} color='#c62828' />
        <Badge text={`Syncope: ${cohort.syncope_n}/40`} color='#555' />
      </div>
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      {geneAtlas.key_drugs_ci?.length > 0 && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🚫 <b>CI:</b> {geneAtlas.key_drugs_ci.join(' · ')}
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
      <h2 style={{ color: '#1a237e' }}>{data.atlas}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle}</p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {[
          { label: 'Total Patients', val: data.total_patients, color: '#1a237e' },
          { label: 'Genes', val: data.gene_count, color: '#4a148c' },
          { label: 'Seeds', val: data.seeds, color: '#b71c1c' },
          { label: 'LQT Genes', val: cats.long_qt_syndromes?.length || 0, color: '#880e4f' },
          { label: 'ARVC Genes', val: cats.arvc_desmosomal?.length || 0, color: '#006064' },
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
      <h2 style={{ color: '#1a237e' }}>Gene Reference Table</h2>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#1a237e', color: '#fff' }}>
              {['Gene', 'Locus', 'Size', 'Inheritance', 'Syndrome', 'Age Dx', '♀%', 'ICD/40', 'SCD/40', 'Syncope/40', 'Pathognomonic'].map(h => (
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
                  <td style={{ padding: '6px 10px', color: '#c62828', fontWeight: 600 }}>{d.scd_or_resuscitated_n}</td>
                  <td style={{ padding: '6px 10px' }}>{d.syncope_n}</td>
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
            <span style={{ fontWeight: 600, color: '#1a237e', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}: </span>
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
      <h2 style={{ color: '#1a237e' }}>Clinical Atlas — 8-Gene Reference</h2>
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
      <h2 style={{ color: '#1a237e' }}>Definitions & Critical Clinical Rules</h2>
      <div style={{ marginBottom: 24 }}>
        <h3 style={{ color: '#4a148c' }}>Glossary</h3>
        {Object.entries(defs.definitions || {}).map(([term, def]) => (
          <div key={term} style={{ borderBottom: '1px solid #e0e0e0', padding: '8px 0' }}>
            <span style={{ fontWeight: 700, color: '#1a237e' }}>{term.replace(/_/g, ' ')}: </span>
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

export default function HereditaryInheritedArrhythmiaSyndromeAtlasPage() {
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
      <div style={{ background: 'linear-gradient(135deg, #1a237e, #4a148c)', color: '#fff', borderRadius: 10, padding: '20px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>🫀 Hereditary Inherited Arrhythmia Syndrome Atlas</h1>
        <p style={{ margin: '6px 0 0', opacity: 0.85, fontSize: 14 }}>
          Complete 8-Gene Reference — KCNQ1 · KCNH2 · SCN5A · RYR2 · PKP2 · DSP · CACNA1C · HCN4 | 320-patient aggregate (8×40) | seeds 3742–3749
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(i)}
            style={{
              padding: '8px 18px', border: 'none', cursor: 'pointer',
              background: tab === i ? '#1a237e' : 'transparent',
              color: tab === i ? '#fff' : '#1a237e',
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
