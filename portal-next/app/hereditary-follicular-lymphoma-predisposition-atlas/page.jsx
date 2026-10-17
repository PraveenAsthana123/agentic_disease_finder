'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-follicular-lymphoma-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'CREBBP':   '#4a148c',  // deep purple     — RTS1; HAT-LOF; mocetinostat HDACi; FL 60-70%
  'EZH2':     '#1b5e20',  // deep green      — Weaver; GOF Y641; tazemetostat FDA2020; FL 80-90%
  'TNFRSF14': '#880e4f',  // deep magenta    — 1p36 LOF; BTLA checkpoint; obinutuzumab; poor prognosis
  'KMT2D':    '#e65100',  // deep orange     — Kabuki1; H3K4me1 loss; tazemetostat + epigenetic
  'RRAGC':    '#006064',  // dark teal       — mTOR GOF; everolimus; grade 3B transformation
  'MAP2K1':   '#bf360c',  // deep red-brown  — CFC; MEK1 GOF; cobimetinib; RAS-MAPK
  'TP53':     '#b71c1c',  // deep red        — LFS; AVOID RADIATION ABSOLUTELY; transform 40-50%
  'ATM':      '#1a237e',  // deep navy       — RADIOSENSITIVITY ABSOLUTE; del11q; ceralasertib ATRi
};

const GENE_INFO = {
  'CREBBP':   { full: 'RTS1-Rubinstein-Taybi / HAT-LOF / FL-60-70pct / Mocetinostat-HDACi / Tazemetostat-EZH2-synergy / broad-thumbs-ID',                      locus: '16p13.3', size: '2442 aa / 265 kDa', inh: 'AD LOF heterozygous' },
  'EZH2':     { full: 'Weaver-Syndrome / GOF-Y641F-Y641N / FL-80-90pct-MOST-COMMON / Tazemetostat-FDA2020-ORR-69pct / SYMPHONY-frontline / H3K27me3',            locus: '7q36.1',  size: '746 aa / 85 kDa',   inh: 'AD GOF heterozygous' },
  'TNFRSF14': { full: '1p36-LOF / HVEM-BTLA-Checkpoint-Loss / FL-40-68pct-HIGHEST / Obinutuzumab-preferred / Poor-prognosis / FLIPI2-elevated / PD-1-rationale', locus: '1p36.32', size: '283 aa / 31 kDa',   inh: 'AD LOF / 1p36 monosomy' },
  'KMT2D':    { full: 'Kabuki1 / H3K4me1-Enhancer-Loss / FL-60-80pct / Tazemetostat-active / Lenalidomide-R2-aza / Cardiac-30pct',                               locus: '12q13.12', size: '5537 aa / 500 kDa', inh: 'AD LOF heterozygous' },
  'RRAGC':    { full: 'mTORC1-GOF / FL-15-17pct-relapsed / Everolimus-active / Grade-3B-transform-risk / Idelalisib-mTOR-combo / lysosomal-mTOR',                locus: '1p34.3',  size: '424 aa / 46 kDa',   inh: 'AD GOF heterozygous' },
  'MAP2K1':   { full: 'CFC-Syndrome / MEK1-GOF / FL-10-15pct-relapsed / Cobimetinib-MEKi / Trametinib / p-ERK-biomarker / BRAF-co-test',                        locus: '15q22.31', size: '393 aa / 44 kDa',   inh: 'AD GOF heterozygous' },
  'TP53':     { full: 'LFS-Li-Fraumeni / AVOID-RADIATION-ABSOLUTELY / Transform-40-50pct / No-Watchful-Waiting / APR-246-venetoclax / WB-MRI-Toronto',           locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF heterozygous' },
  'ATM':      { full: 'RADIOSENSITIVITY-ABSOLUTE-no-RT / NHL-4x-FL-2-3x / del11q-35pct / Ceralasertib-ATRi-synthetic-lethality / RIC-HSCT-no-TBI',              locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF heterozygous' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}18`, color, border: `1px solid ${color}55`,
      borderRadius: 4, fontSize: 10, padding: '2px 7px', marginLeft: 6, fontWeight: 600,
    }}>{text}</span>
  );
}

function card(label, value, sub, color) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 10,
      padding: '14px 18px', minWidth: 140, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#aaa' }}>{sub}</div>
    </div>
  );
}

export default function HereditaryFLPredispositionAtlasPage() {
  const [tab, setTab]                 = useState('Overview');
  const [overview, setOverview]       = useState(null);
  const [breakdown, setBreakdown]     = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState('');
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true); setError('');
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '8px 18px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600,
    fontSize: 13, background: tab === t ? '#1b5e20' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary Follicular Lymphoma Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh', padding: '28px 24px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1b5e20 0%,#4a148c 50%,#1a237e 100%)',
        borderRadius: 14, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>
          🧬 Hereditary Follicular Lymphoma Predisposition Atlas
        </div>
        <div style={{ fontSize: 13, opacity: 0.88 }}>
          Complete 8-Gene Reference · CREBBP · EZH2 · TNFRSF14 · KMT2D · RRAGC · MAP2K1 · TP53 · ATM
        </div>
        <div style={{ fontSize: 11, opacity: 0.70, marginTop: 4 }}>
          320-Patient Aggregate · 8×40 · Seeds 3590–3597 · Follicular Lymphoma Germline Predisposition
        </div>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            {card('Total Patients', overview.total_patients, '8 genes × 40 each', '#1b5e20')}
            {card('CR (1st-line)', overview.cr_n, `${Math.round(overview.cr_n/overview.total_patients*100)}% complete response`, '#4a148c')}
            {card('EZH2 Mutated', overview.ezh2_mutated, 'somatic GOF across cohort', '#1b5e20')}
            {card('BM Involved', overview.bm_involved, 'bone marrow trephine positive', '#880e4f')}
            {card('Transformed', overview.transformed, 'FL→DLBCL histological', '#b71c1c')}
            {card('5-yr Alive', overview.five_yr_alive, `${Math.round(overview.five_yr_alive/overview.total_patients*100)}% overall survival`, '#1b5e20')}
            {card('Grade 3B', overview.grade3b, 'aggressive grade; DLBCL-like', '#e65100')}
            {card('FLIPI High', overview.flipi_high, 'FLIPI ≥3 poor prognosis', '#1a237e')}
          </div>

          {overview.clinical_pearl && (
            <div style={{ background: '#e8f5e9', border: '1px solid #a5d6a7', borderRadius: 10,
              padding: '14px 18px', marginBottom: 20, fontSize: 12, color: '#1b5e20' }}>
              <strong>⚡ Clinical Pearl:</strong> {overview.clinical_pearl}
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
            {(overview.gene_summary || []).map(gs => {
              const color = GENE_COLORS[gs.gene] || '#555';
              const info  = GENE_INFO[gs.gene] || {};
              const expanded = expandedGene === gs.gene;
              return (
                <div key={gs.gene} style={{ background: '#fff', border: `2px solid ${color}33`,
                  borderRadius: 12, padding: '16px 18px', cursor: 'pointer' }}
                  onClick={() => setExpandedGene(expanded ? null : gs.gene)}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontSize: 17, fontWeight: 800, color }}>{gs.gene}</span>
                    <Badge text={gs.inheritance.split(' ')[0]} color={color} />
                    <Badge text={info.locus || gs.locus} color={color} />
                    <span style={{ marginLeft: 'auto', fontSize: 11, color: '#aaa' }}>{info.size}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>{gs.syndrome}</div>
                  <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>
                    Mean age Dx: <strong style={{ color }}>{gs.mean_age_dx} yr</strong> · n={gs.n}
                  </div>
                  <div style={{ fontSize: 11, background: `${color}0d`, borderRadius: 6,
                    padding: '6px 10px', color, marginBottom: 6 }}>
                    <strong>FL Risk:</strong> {gs.fl_risk}
                  </div>
                  {expanded && (
                    <div style={{ marginTop: 10 }}>
                      <div style={{ fontSize: 11, color: '#555', marginBottom: 6 }}>
                        <strong>Full info:</strong> {info.full}
                      </div>
                      <div style={{ fontSize: 11, background: '#fff3e0', borderRadius: 6,
                        padding: '6px 10px', color: '#bf360c', marginBottom: 6 }}>
                        <strong>KEY AVOID:</strong> {gs.key_avoid}
                      </div>
                      <div style={{ fontSize: 11, background: '#e8f5e9', borderRadius: 6,
                        padding: '6px 10px', color: '#1b5e20', marginBottom: 6 }}>
                        <strong>KEY RULE:</strong> {gs.key_rule}
                      </div>
                      <div style={{ fontSize: 11, background: '#e3f2fd', borderRadius: 6,
                        padding: '6px 10px', color: '#1a237e' }}>
                        <strong>TARGETED Rx:</strong> {gs.targeted_rx}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', background: '#fff',
            borderRadius: 10, overflow: 'hidden', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1b5e20', color: '#fff' }}>
                {['Gene','Syndrome','Locus','Inh','n','Mean Age Dx','CR n','EZH2 Mut','BM Inv','Transform','Grade 3B','FLIPI ≥3','Stage 4','5yr Alive'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown.gene_data || {}).map(([gene, d], idx) => {
                const color = GENE_COLORS[gene] || '#555';
                return (
                  <tr key={gene} style={{ background: idx % 2 === 0 ? '#fff' : '#fafafa',
                    borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '9px 12px', fontWeight: 800, color }}>{gene}</td>
                    <td style={{ padding: '9px 12px', fontSize: 10, maxWidth: 160 }}>{d.syndrome?.split('-').slice(0,3).join('-')}</td>
                    <td style={{ padding: '9px 12px' }}>{d.locus}</td>
                    <td style={{ padding: '9px 12px', fontSize: 10 }}>{d.inheritance?.split(' ').slice(0,2).join(' ')}</td>
                    <td style={{ padding: '9px 12px', fontWeight: 700 }}>{d.n}</td>
                    <td style={{ padding: '9px 12px', color }}>{d.avg_age_dx} yr</td>
                    <td style={{ padding: '9px 12px' }}>{d.cr_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.ezh2_mutated_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.bm_involved_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.transformed_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.grade3b_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.flipi_high_n}</td>
                    <td style={{ padding: '9px 12px' }}>{d.stage4_n}</td>
                    <td style={{ padding: '9px 12px', fontWeight: 700, color: '#1b5e20' }}>{d.five_yr_alive_n}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(380px,1fr))', gap: 16 }}>
          {Object.entries(breakdown.gene_data || {}).map(([gene, d]) => {
            const color = GENE_COLORS[gene] || '#555';
            return (
              <div key={gene} style={{ background: '#fff', border: `2px solid ${color}33`,
                borderRadius: 12, padding: '16px 18px' }}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
                  <span style={{ fontSize: 18, fontWeight: 800, color }}>{gene}</span>
                  <span style={{ marginLeft: 8, fontSize: 11, color: '#888' }}>{d.locus}</span>
                  <span style={{ marginLeft: 'auto', fontSize: 11, background: `${color}18`,
                    color, borderRadius: 4, padding: '2px 8px' }}>n={d.n}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                  {[
                    ['Avg Age Dx', `${d.avg_age_dx} yr`],
                    ['CR (1st-line)', d.cr_n],
                    ['EZH2 Mutated', d.ezh2_mutated_n],
                    ['BM Involved', d.bm_involved_n],
                    ['Transformed', d.transformed_n],
                    ['Grade 3B', d.grade3b_n],
                    ['FLIPI ≥3', d.flipi_high_n],
                    ['Stage 4', d.stage4_n],
                    ['5yr Alive', d.five_yr_alive_n],
                    ['Avg FLIPI', d.avg_flipi],
                    ['LDH Elevated', d.ldh_elevated_n],
                    ['Concurrent Breast', d.concurrent_breast_n],
                  ].map(([k, v]) => (
                    <div key={k} style={{ background: '#f8f9fa', borderRadius: 6,
                      padding: '5px 8px', fontSize: 11 }}>
                      <div style={{ color: '#aaa', fontSize: 10 }}>{k}</div>
                      <div style={{ fontWeight: 700, color }}>{v}</div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 11, background: '#fff3e0', borderRadius: 6,
                  padding: '6px 10px', color: '#bf360c', marginBottom: 6 }}>
                  <strong>KEY AVOID:</strong> {d.key_avoid?.split(';')[0]}
                </div>
                <div style={{ fontSize: 11, background: '#e8f5e9', borderRadius: 6,
                  padding: '6px 10px', color: '#1b5e20', marginBottom: 6 }}>
                  <strong>KEY RULE:</strong> {d.key_rule?.split(';')[0]}
                </div>
                <div style={{ fontSize: 11, background: '#e3f2fd', borderRadius: 6,
                  padding: '6px 10px', color: '#1a237e' }}>
                  <strong>Rx:</strong> {d.targeted_rx?.split(';')[0]}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          {(definitions.key_clinical_distinctions || []).length > 0 && (
            <div style={{ background: '#fff', borderRadius: 12, padding: '18px 22px', marginBottom: 20 }}>
              <div style={{ fontSize: 15, fontWeight: 800, color: '#1b5e20', marginBottom: 12 }}>
                Key Clinical Distinctions — Hereditary Follicular Lymphoma Predisposition
              </div>
              {definitions.key_clinical_distinctions.map((d, i) => (
                <div key={i} style={{ background: '#e8f5e9', borderLeft: '4px solid #1b5e20',
                  borderRadius: 6, padding: '10px 14px', marginBottom: 8, fontSize: 12, color: '#1b5e20' }}>
                  {d}
                </div>
              ))}
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(380px,1fr))', gap: 14 }}>
            {(definitions.definitions || []).map((def, i) => {
              const gene = def.term?.split(' — ')[0];
              const color = GENE_COLORS[gene] || '#555';
              const expanded = expandedDef === i;
              return (
                <div key={i} style={{ background: '#fff', border: `2px solid ${color}22`,
                  borderRadius: 10, padding: '14px 18px', cursor: 'pointer' }}
                  onClick={() => setExpandedDef(expanded ? null : i)}>
                  <div style={{ fontWeight: 700, color, marginBottom: 6, fontSize: 13 }}>
                    {def.term}
                  </div>
                  {expanded && (
                    <pre style={{ fontSize: 11, color: '#555', whiteSpace: 'pre-wrap',
                      background: '#f8f9fa', borderRadius: 6, padding: '10px', marginTop: 8 }}>
                      {def.definition}
                    </pre>
                  )}
                  {!expanded && (
                    <div style={{ fontSize: 11, color: '#aaa' }}>Click to expand definition…</div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
