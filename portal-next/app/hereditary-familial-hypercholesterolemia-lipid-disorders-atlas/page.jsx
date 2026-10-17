'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-familial-hypercholesterolemia-lipid-disorders-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'LDLR':    '#1565c0',  // deep blue     -- FH1; most common; corneal arcus; PCSK9i
  'APOB':    '#0277bd',  // medium blue   -- FDB FH2; R3527Q; milder; PCSK9i reduced
  'PCSK9':   '#880e4f',  // deep crimson  -- GOF=FH3 D374Y / LOF=protection R46L
  'LDLRAP1': '#c62828',  // deep red      -- ARH; AR biallelic; splenomegaly; apheresis
  'APOE':    '#e65100',  // burnt orange  -- FH3 Type III; ε2/ε2; palmar xanthoma; fibrates
  'LPL':     '#1b5e20',  // dark green    -- FCS; pancreatitis; volanesorsen; fat restrict
  'APOA5':   '#4e342e',  // dark brown    -- HTG type V; REDUCE-IT; fibrates+omega3
  'ABCG5':   '#4a148c',  // deep purple   -- sitosterolaemia; childhood xanthomas; ezetimibe
};

const GENE_INFO = {
  'LDLR':    { full: 'LDL-Receptor / FH1-most-common-60-80% / corneal-arcus-lt45yr-PATHOGNOMONIC / statin+ezetimibe+PCSK9i / evinacumab-HoFH', locus: '19p13.2', size: '1324 aa / 164 kDa', inh: 'AD LOF' },
  'APOB':    { full: 'ApoB-100 / FDB-FH2-R3527Q / PCSK9i-less-effective / inclisiran+lomitapide-effective / milder-than-LDLR', locus: '2p24.1', size: '4563 aa / 550 kDa', inh: 'AD LOF' },
  'PCSK9':   { full: 'PCSK9 / GOF-D374Y-FH3-PATHOGNOMONIC / LOF-R46L-natural-protection-47% / evolocumab-alirocumab-FDA2015 / inclisiran-siRNA', locus: '1p32.3', size: '692 aa / 75 kDa', inh: 'AD GOF/LOF' },
  'LDLRAP1': { full: 'LDL-Receptor-Adaptor / ARH-AR-biallelic / splenomegaly-PATHOGNOMONIC / evinacumab-ANGPTL3i / LDL-apheresis', locus: '1p36.11', size: '308 aa / 35 kDa', inh: 'AR LOF' },
  'APOE':    { full: 'ApoE-ε2/ε2 / FH-type-3-Fredrickson-III / palmar-crease-xanthoma-PATHOGNOMONIC / fibrates-FIRST-LINE / secondary-trigger-needed', locus: '19q13.32', size: '299 aa / 34 kDa', inh: 'AR/Codominant ε2/ε2' },
  'LPL':     { full: 'Lipoprotein-Lipase / FCS-AR-biallelic / recurrent-pancreatitis-PATHOGNOMONIC / volanesorsen-siRNA-FDA2019 / fibrates-omega3-INEFFECTIVE', locus: '8p21.3', size: '475 aa / 55 kDa', inh: 'AR LOF' },
  'APOA5':   { full: 'ApoA-V / HTG-type-V / pancreatitis-risk / REDUCE-IT-omega3-25%-MACE / fibrates+EPA-first-line', locus: '11q23.3', size: '366 aa / 40 kDa', inh: 'AD LOF' },
  'ABCG5':   { full: 'Sterolin-1/ABCG5 / sitosterolaemia / childhood-xanthomas-PATHOGNOMONIC / haemolytic-anaemia-PATHOGNOMONIC / ezetimibe-mechanism-matched / stanol-CATASTROPHIC-CI', locus: '2p21', size: '651 aa / 75 kDa', inh: 'AR LOF' },
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
        <Badge text={`Pancreatitis: ${cohort.pancreatitis_n}/40`} color={gene === 'LPL' || gene === 'APOA5' ? '#c62828' : '#777'} />
        <Badge text={`CV Events: ${cohort.cv_event_n}/40`} color='#c62828' />
        <Badge text={`Xanthomas: ${cohort.xanthoma_n}/40`} color={gene === 'ABCG5' ? '#4a148c' : '#555'} />
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
          { label: 'AD FH Genes', val: cats.ad_fh_genes?.length || 0, color: '#c62828' },
          { label: 'Pancreatitis Risk', val: cats.pancreatitis_risk?.length || 0, color: '#e65100' },
          { label: 'Childhood Onset', val: cats.childhood_onset?.length || 0, color: '#4a148c' },
        ].map(k => (
          <div key={k.label} style={{ background: k.color, color: '#fff', borderRadius: 8, padding: '10px 18px', minWidth: 130, textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 700 }}>{k.val}</div>
            <div style={{ fontSize: 12 }}>{k.label}</div>
          </div>
        ))}
      </div>

      <div style={{ background: '#fff8e1', borderRadius: 8, padding: 14, marginBottom: 16, border: '2px solid #ff8f00' }}>
        <h3 style={{ margin: '0 0 8px', color: '#e65100' }}>💊 Treatment Strategy by Gene</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, fontSize: 12 }}>
          {[
            { gene: 'LDLR', rx: 'Statin+Ezet+PCSK9i', color: '#1565c0' },
            { gene: 'APOB', rx: 'Statin+Inclisiran ⚡', color: '#0277bd' },
            { gene: 'PCSK9', rx: 'PCSK9i ↑↑ response', color: '#880e4f' },
            { gene: 'LDLRAP1', rx: 'Evinacumab+Apheresis', color: '#c62828' },
            { gene: 'APOE', rx: 'Fibrates FIRST 🥇', color: '#e65100' },
            { gene: 'LPL', rx: 'Volanesorsen siRNA', color: '#1b5e20' },
            { gene: 'APOA5', rx: 'Fibrates+EPA REDUCE-IT', color: '#4e342e' },
            { gene: 'ABCG5', rx: 'Ezetimibe 🥇 + Diet', color: '#4a148c' },
          ].map(t => (
            <div key={t.gene} style={{ background: t.color, color: '#fff', borderRadius: 6, padding: '6px 10px', textAlign: 'center' }}>
              <div style={{ fontWeight: 700, fontSize: 13 }}>{t.gene}</div>
              <div style={{ fontSize: 11 }}>{t.rx}</div>
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
              {['Gene', 'Locus', 'Size', 'Inheritance', 'Syndrome', 'Age Dx', '♀%', 'Pancreatitis/40', 'CV Events/40', 'Xanthomas/40', 'Pathognomonic'].map(h => (
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
                  <td style={{ padding: '6px 10px', color: (g === 'LPL' || g === 'APOA5') ? '#c62828' : '#333', fontWeight: (g === 'LPL' || g === 'APOA5') ? 700 : 400 }}>{d.pancreatitis_n}</td>
                  <td style={{ padding: '6px 10px', color: '#c62828', fontWeight: 600 }}>{d.cv_event_n}</td>
                  <td style={{ padding: '6px 10px', color: g === 'ABCG5' ? '#4a148c' : '#333', fontWeight: g === 'ABCG5' ? 700 : 400 }}>{d.xanthoma_n}</td>
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

export default function HereditaryFHLipidDisordersAtlasPage() {
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
        <h1 style={{ margin: 0, fontSize: 22 }}>🫀 Hereditary Familial Hypercholesterolaemia & Lipid Disorders Atlas</h1>
        <p style={{ margin: '6px 0 0', opacity: 0.85, fontSize: 14 }}>
          Complete 8-Gene Reference — LDLR · APOB · PCSK9 · LDLRAP1 · APOE · LPL · APOA5 · ABCG5 | 320-patient aggregate (8×40) | seeds 3766–3773
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
