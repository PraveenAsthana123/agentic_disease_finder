'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-bone-marrow-failure-syndrome-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'FANCA':  '#b71c1c',   // deep red         -- FA-A most common; DEB/MMC; RIC-HSCT
  'FANCC':  '#880e4f',   // deep crimson      -- FA-C Ashkenazi IVS4+4A>T
  'FANCG':  '#4a148c',   // deep purple       -- FA-G highest AML; monosomy7
  'DKC1':   '#1565c0',   // deep blue         -- DC XLR; mucocutaneous triad; IPF
  'TERC':   '#2e7d32',   // forest green      -- DC AD; anticipation; telomeropathy
  'RPS19':  '#e65100',   // burnt orange      -- DBA; elevated eADA; triphalangeal thumb
  'SBDS':   '#006064',   // dark teal         -- SDS; EPI at birth; i7q benign
  'MPL':    '#33691e',   // olive green       -- CAMT; amegakaryocytosis; TPO-RAs inactive
};

const GENE_INFO = {
  'FANCA':  { full: 'Fanconi-Anemia-Group-A / most-common-~60%-FA / DEB/MMC-pathognomonic / AML-MDS-20-30% / RIC-HSCT-mandatory', locus: '16q24.3', size: '1455 aa / 163 kDa', inh: 'AR LOF' },
  'FANCC':  { full: 'Fanconi-Anemia-Group-C / IVS4+4A>T-Ashkenazi-founder-PATHOGNOMONIC / 322delG-non-Jewish / earlier-onset-severe-aplasia / somatic-mosaicism-trap', locus: '9q22.32', size: '558 aa / 63 kDa', inh: 'AR LOF' },
  'FANCG':  { full: 'Fanconi-Anemia-Group-G / XRCC9 / highest-AML-MDS-risk-FA / monosomy7=URGENT-HSCT / poor-HSCT-outcomes-vs-FA-A', locus: '9p13.3', size: '622 aa / 68 kDa', inh: 'AR LOF' },
  'DKC1':   { full: 'Dyskeratosis-Congenita-XLR / DKC1-dyskerin / mucocutaneous-triad-PATHOGNOMONIC / IPF-dominant-cause-of-death / danazol-70%-response', locus: 'Xq28', size: '514 aa / 58 kDa', inh: 'XLR LOF' },
  'TERC':   { full: 'Telomerase-RNA-component / DC-AD / genetic-anticipation-PATHOGNOMONIC / 5-10%-idiopathic-AA=telomeropathy / danazol-TERT-upregulation', locus: '3q26.2', size: '451 nt RNA', inh: 'AD LOF (haploinsufficiency)' },
  'RPS19':  { full: 'Diamond-Blackfan-Anemia / 25%-DBA-cases / elevated-eADA-PATHOGNOMONIC / triphalangeal-thumb / no-steroids-<1yr / corticosteroids-60-80%', locus: '19q13.2', size: '145 aa / 17 kDa', inh: 'AD LOF (haploinsufficiency)' },
  'SBDS':   { full: 'Shwachman-Diamond-Syndrome / EPI-at-birth-PATHOGNOMONIC / neutropenia-100% / i7q-del20q-SDS-benign / AML-MDS-20-30% / PERT-first-line', locus: '7q11.21', size: '250 aa / 29 kDa', inh: 'AR LOF (gene conversion)' },
  'MPL':    { full: 'CAMT / W515X-most-common / amegakaryocytosis / TPO-markedly-elevated / Type-I-pancytopenia-by-3-5yr / TPO-RAs-INACTIVE / HSCT-only-cure', locus: '1p34.2', size: '635 aa / 84 kDa', inh: 'AR LOF' },
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
          <div style={{ fontWeight: 600 }}>{geneAtlas.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge text={`BMF: ${(geneAtlas.bmf_risk || '').split(';')[0]}`} color={bg} />
      </div>
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      {geneAtlas.key_rule && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ✓ <b>Key rule:</b> {geneAtlas.key_rule}
        </div>
      )}
      {geneAtlas.key_avoid && (
        <div style={{ fontSize: 12, background: '#ffebee', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ✗ <b>Avoid:</b> {geneAtlas.key_avoid}
        </div>
      )}
      {geneAtlas.targeted_rx && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          💊 <b>Treatment:</b> {geneAtlas.targeted_rx}
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 10 }}>
        {[
          ['N', cohort.n],
          ['Avg age', cohort.mean_age != null ? cohort.mean_age + 'yr' : '—'],
          ['HSCT', cohort.hsct_count != null ? `${cohort.hsct_count}/40` : '—'],
          gene === 'FANCA' || gene === 'FANCC' || gene === 'FANCG'
            ? ['Aplasia', cohort.aplasia_count != null ? `${cohort.aplasia_count}/40` : '—']
            : gene === 'DKC1'
            ? ['Triad', cohort.triad_complete != null ? `${cohort.triad_complete}/40` : '—']
            : gene === 'TERC'
            ? ['Aplasia', cohort.aplasia_count != null ? `${cohort.aplasia_count}/40` : '—']
            : gene === 'RPS19'
            ? ['Steroid Resp', cohort.steroid_response != null ? `${cohort.steroid_response}/40` : '—']
            : gene === 'SBDS'
            ? ['EPI at birth', cohort.epi_at_birth != null ? `${cohort.epi_at_birth}/40` : '—']
            : ['ICH', cohort.ich_count != null ? `${cohort.ich_count}/40` : '—'],
        ].map(([label, val]) => (
          <div key={label} style={{ background: '#f5f5f5', padding: 8, borderRadius: 4, textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#888' }}>{label}</div>
            <div style={{ fontWeight: 700, fontSize: 16 }}>{val ?? '—'}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HBMFAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefinitions(df);
    }).catch(e => setErr(String(e))).finally(() => setLoading(false));
  }, []);

  const PAGE_COLOR = '#37474f';  // blue-grey dark — BMF/hematology theme

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24, fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ background: PAGE_COLOR, color: '#fff', borderRadius: 10, padding: '20px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>🩸 Hereditary Bone Marrow Failure Syndrome Atlas</h1>
        <div style={{ fontSize: 13, marginTop: 6, opacity: 0.85 }}>
          Complete 8-Gene BMF Reference — Fanconi Anemia · Dyskeratosis Congenita · Diamond-Blackfan Anemia · Shwachman-Diamond · CAMT
        </div>
        <div style={{ fontSize: 12, marginTop: 4, opacity: 0.7 }}>
          FANCA · FANCC · FANCG · DKC1 · TERC · RPS19 · SBDS · MPL · 320 patients · seeds 3694–3701
        </div>
      </div>

      {err && <div style={{ background: '#ffebee', color: '#c62828', padding: 12, borderRadius: 6, marginBottom: 16 }}>Error: {err}</div>}
      {loading && <div style={{ padding: 24, textAlign: 'center', color: '#666' }}>Loading atlas data…</div>}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 20, border: 'none', cursor: 'pointer',
            background: tab === t ? PAGE_COLOR : '#e0e0e0',
            color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 24 }}>
            {[
              ['Total Patients', overview.summary?.total_n, PAGE_COLOR],
              ['Genes Covered', overview.summary?.genes, '#455a64'],
              ['Syndromes', overview.summary?.syndromes_covered, '#546e7a'],
              ['HSCT Required', `${overview.summary?.hsct_required_n} (${overview.summary?.hsct_pct}%)`, '#b71c1c'],
              ['FA Aplasia n', overview.summary?.fa_aplasia_n, '#880e4f'],
              ['FA AML/MDS n', overview.summary?.fa_aml_mds_n, '#4a148c'],
              ['DC IPF (DKC1+TERC)', overview.summary?.dkc1_terc_ipf_total, '#1565c0'],
              ['DBA Steroid Resp', overview.summary?.rps19_steroid_response_n, '#e65100'],
              ['MPL ICH n', overview.summary?.mpl_ich_n, '#33691e'],
              ['SBDS EPI at birth', overview.summary?.sbds_epi_at_birth_n, '#006064'],
            ].map(([label, val, bg]) => (
              <div key={label} style={{ background: bg, color: '#fff', borderRadius: 8, padding: '14px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 22, fontWeight: 700 }}>{val ?? '—'}</div>
              </div>
            ))}
          </div>

          <h3 style={{ color: PAGE_COLOR, marginBottom: 10 }}>Clinical Highlights</h3>
          <div style={{ display: 'grid', gap: 8, marginBottom: 20 }}>
            {overview.clinical_highlights?.map((h, i) => (
              <div key={i} style={{ background: '#f5f5f5', padding: '10px 14px', borderRadius: 6, fontSize: 13, borderLeft: `4px solid ${PAGE_COLOR}` }}>
                {h}
              </div>
            ))}
          </div>

          <h3 style={{ color: PAGE_COLOR, marginBottom: 10 }}>Key Pharmacological Rules</h3>
          <div style={{ display: 'grid', gap: 8, marginBottom: 20 }}>
            {overview.key_pharmacological_rules?.map((r, i) => (
              <div key={i} style={{ background: '#e8f5e9', padding: '10px 14px', borderRadius: 6, fontSize: 13, borderLeft: '4px solid #2e7d32' }}>
                <b>✓</b> {r}
              </div>
            ))}
          </div>

          <h3 style={{ color: PAGE_COLOR, marginBottom: 10 }}>Cascade Testing</h3>
          <div style={{ background: '#e3f2fd', padding: '12px 16px', borderRadius: 6, fontSize: 13 }}>
            {overview.cascade_testing}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: PAGE_COLOR, color: '#fff' }}>
                {['Gene', 'Syndrome', 'Locus', 'Size', 'Inh.', 'Pathognomonic', 'Key Rule', 'Avoid'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.atlas_genes?.map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', verticalAlign: 'top' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || PAGE_COLOR }}>{g.gene}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 180 }}>{g.syndrome}</td>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap' }}>{g.locus}</td>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap' }}>{GENE_INFO[g.gene]?.size || '—'}</td>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap' }}>{g.inheritance?.split(' ')[0] + ' ' + g.inheritance?.split(' ')[1] || g.inheritance}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 200, color: '#bf360c' }}>{g.pathognomonic}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 200, color: '#1b5e20' }}>{g.key_rule}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 180, color: '#c62828' }}>{g.key_avoid}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            {Object.entries(GENE_COLORS).map(([gene, color]) => (
              <GeneCard key={gene} gene={gene} color={color} info={GENE_INFO[gene]} breakdown={breakdown} />
            ))}
          </div>

          <h3 style={{ color: PAGE_COLOR }}>Syndrome Group Summary</h3>
          {Object.entries(breakdown.syndrome_groups || {}).map(([grp, info]) => (
            <div key={grp} style={{ border: `1px solid #ddd`, borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 8, color: PAGE_COLOR }}>{grp.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 12, marginBottom: 4 }}><b>Genes:</b> {info.genes?.join(', ')}</div>
              {info.diagnostic_test && <div style={{ fontSize: 12, marginBottom: 4 }}><b>Diagnosis:</b> {info.diagnostic_test}</div>}
              {info.treatment && <div style={{ fontSize: 12, marginBottom: 4 }}><b>Treatment:</b> {info.treatment}</div>}
              {info.hsct_conditioning && <div style={{ fontSize: 12, marginBottom: 4 }}><b>HSCT conditioning:</b> {info.hsct_conditioning}</div>}
              {info.caveat && <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4 }}>⚠ {info.caveat}</div>}
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          {definitions.definitions?.map((d, i) => (
            <div key={i} style={{ marginBottom: 16, padding: 16, border: '1px solid #e0e0e0', borderRadius: 8 }}>
              <div style={{ fontWeight: 700, fontSize: 15, color: PAGE_COLOR, marginBottom: 6 }}>{d.term}</div>
              <div style={{ fontSize: 13, lineHeight: 1.6, color: '#444' }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 32, padding: '12px 16px', background: '#f5f5f5', borderRadius: 6, fontSize: 11, color: '#888' }}>
        Hereditary-Bone-Marrow-Failure-Syndrome-Atlas · FANCA-FANCC-FANCG-DKC1-TERC-RPS19-SBDS-MPL · 320 patients (8×40) · seeds 3694–3701 · Generated by agenticfinder
      </div>
    </div>
  );
}
