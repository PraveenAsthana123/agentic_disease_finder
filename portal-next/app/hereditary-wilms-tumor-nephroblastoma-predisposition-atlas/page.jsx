'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-wilms-tumor-nephroblastoma-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'WT1':    '#1a237e',   // deep navy          -- WAGR/Denys-Drash/Frasier; aniridia PATHOGNOMONIC
  'AMER1':  '#4e342e',   // dark brown         -- OSCS females; XLR Wilms males
  'DROSHA': '#880e4f',   // deep crimson       -- miRNA Microprocessor; E1147K; bilateral WT
  'DGCR8':  '#1b5e20',   // forest green       -- DiGeorge 22q11.2; R1353H; miRNA axis
  'SIX1':   '#e65100',   // burnt orange       -- Q177R hotspot 10-12%; unfavorable histology
  'SIX2':   '#bf360c',   // deep orange-red    -- Q177R hotspot 3-5%; parallel SIX1
  'CTR9':   '#006064',   // dark teal          -- PAF1 complex; AR biallelic; PLNR; excellent Px
  'CTNNB1': '#4a148c',   // deep purple        -- beta-catenin Wnt; ID+epilepsy; AFP hepatoblastoma
};

const GENE_INFO = {
  'WT1':    { full: 'Wilms-Tumour-Protein / WAGR-del11p13-Aniridia-PATHOGNOMONIC / Denys-Drash-R394W-DMS-WT-90pct-PATHOGNOMONIC / Frasier-IVS9-FSGS-Gonadoblastoma-PATHOGNOMONIC / Bilateral-WT-20pct', locus: '11p13', size: '485 aa / 54 kDa', inh: 'AD LOF' },
  'AMER1':  { full: 'WTX-APC-Membrane-Recruiter / XLR-LOF-males-Wilms / OSCS-females-striated-bone-PATHOGNOMONIC / Somatic-15-20pct-sporadic-WT / Wnt-pathway-AMER1-destruction-complex', locus: 'Xq11.2', size: '1135 aa / 123 kDa', inh: 'XLR LOF' },
  'DROSHA': { full: 'miRNA-Microprocessor-DROSHA+DGCR8 / E1147K-hotspot-7-12pct / bilateral-WT-germline / blastemal-predominant-unfavorable-histology / co-mutation-SIX1-SIX2', locus: '5p13.3', size: '1374 aa / 159 kDa', inh: 'AD LOF' },
  'DGCR8':  { full: 'DiGeorge-22q11.2 / Microprocessor-DROSHA-partner / R1353H-hotspot-5-8pct / bilateral-WT-germline / cardiac-thymic-hypocalcemia-22q11.2-deletion', locus: '22q11.2', size: '773 aa / 86 kDa', inh: 'AD LOF' },
  'SIX1':   { full: 'Sine-oculis-homeobox-1 / Q177R-hotspot-10-12pct / Unfavorable-histology-PATHOGNOMONIC / DACH1-co-repressor-disrupted / NEVER-de-escalate-SIX1-Q177R', locus: '14q23.1', size: '284 aa / 32 kDa', inh: 'AD GOF' },
  'SIX2':   { full: 'Sine-oculis-homeobox-2 / Q177R-hotspot-3-5pct / Parallel-SIX1-cognate-residue / blastemal-unfavorable / test-SIX1-SIX2-as-paired-panel', locus: '2p21', size: '268 aa / 30 kDa', inh: 'AD GOF' },
  'CTR9':   { full: 'PAF1-complex-scaffold / AR-biallelic-LOF / familial-WT-no-syndrome / PLNR-PATHOGNOMONIC / excellent-prognosis / parents-healthy-obligate-het', locus: '11p15.3', size: '983 aa / 111 kDa', inh: 'AR LOF (biallelic)' },
  'CTNNB1': { full: 'beta-catenin-Wnt-effector / exon-3-phosphodegron-GOF / ID+epilepsy+autism-germline / Wilms+hepatoblastoma / AFP-every-3mo-birth-age4 / NOT-FAP-no-colorectal', locus: '3p22.1', size: '781 aa / 92 kDa', inh: 'AD GOF' },
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
        <Badge text={`Tumour: ${(geneAtlas.bmf_risk || '').split(';')[0]}`} color={bg} />
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
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ✗ <b>Avoid:</b> {geneAtlas.key_avoid}
        </div>
      )}
      {geneAtlas.surveillance && (
        <div style={{ fontSize: 12, color: '#555', marginBottom: 4 }}>
          <b>Surveillance:</b> {geneAtlas.surveillance}
        </div>
      )}
      {geneAtlas.targeted_rx && (
        <div style={{ fontSize: 12, color: '#555' }}>
          <b>Treatment:</b> {geneAtlas.targeted_rx}
        </div>
      )}
      {(cohort.bilateral_n !== undefined) && (
        <div style={{ fontSize: 12, marginTop: 6, color: '#666' }}>
          Cohort n={cohort.n} · bilateral WT: {cohort.bilateral_n} · metastatic: {cohort.metastatic_n}
        </div>
      )}
    </div>
  );
}

export default function WilmsTumorAtlasPage() {
  const [tab, setTab]           = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreak]   = useState(null);
  const [defs, setDefs]         = useState(null);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreak(br); setDefs(df); setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#1a237e' }}>Loading Wilms Tumour Atlas…</div>;

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e,#0d47a1)', color: '#fff', borderRadius: 10, padding: '24px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🧬 Hereditary Wilms Tumour / Nephroblastoma Predisposition Atlas</h1>
        <div style={{ fontSize: 14, opacity: 0.9, marginTop: 6 }}>
          Complete 8-Gene Reference — WT1 · AMER1 · DROSHA · DGCR8 · SIX1 · SIX2 · CTR9 · CTNNB1
        </div>
        <div style={{ fontSize: 12, opacity: 0.8, marginTop: 4 }}>
          320-Patient Aggregate (8×40) · Seeds 3710-3717 · Wilms Tumour Germline Predisposition
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600,
            background: tab === i ? '#1a237e' : '#e8eaf6', color: tab === i ? '#fff' : '#1a237e'
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 0 && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Predisposition Genes', value: overview.gene_count },
              { label: 'Seeds', value: overview.seeds },
              { label: 'Clinical Categories', value: Object.keys(overview.clinical_categories || {}).length },
            ].map(s => (
              <div key={s.label} style={{ background: '#e8eaf6', borderRadius: 8, padding: 16, textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#1a237e' }}>{s.value}</div>
                <div style={{ fontSize: 12, color: '#555' }}>{s.label}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff3e0', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <b>🔑 Key Clinical Rules (8-Gene Wilms Atlas)</b>
            <ul style={{ margin: '8px 0 0', paddingLeft: 18, fontSize: 13 }}>
              {(overview.key_clinical_rules || []).map((r, i) => <li key={i} style={{ marginBottom: 4 }}>{r}</li>)}
            </ul>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 16, marginBottom: 20 }}>
            {Object.entries(overview.clinical_categories || {}).map(([cat, genes]) => (
              <div key={cat} style={{ background: '#f3e5f5', borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 700, color: '#4a148c', marginBottom: 6, textTransform: 'capitalize' }}>
                  {cat.replace(/_/g, ' ')}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {genes.map(g => (
                    <span key={g} style={{ background: GENE_COLORS[g] || '#555', color: '#fff', borderRadius: 4, padding: '3px 10px', fontSize: 13, fontWeight: 700 }}>{g}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#1a237e' }}>Cohort Statistics per Gene</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene','n','Mean Age (yr)','% Female','Syndrome/Category','Inheritance'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(overview.cohort_stats || []).map((s, i) => (
                <tr key={s.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[s.gene] || '#333' }}>{s.gene}</td>
                  <td style={{ padding: '6px 10px' }}>{s.n}</td>
                  <td style={{ padding: '6px 10px' }}>{s.mean_age}</td>
                  <td style={{ padding: '6px 10px' }}>{s.pct_female}%</td>
                  <td style={{ padding: '6px 10px', fontSize: 11 }}>{s.syndrome}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11 }}>{s.inheritance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 1 && breakdown && (
        <div>
          <h3 style={{ color: '#1a237e' }}>Gene-Level Breakdown — 320-Patient Aggregate</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene','n','Mean Age','Bilateral WT','Metastatic','Syndrome','Inheritance'].map(h => (
                  <th key={h} style={{ padding: '7px 9px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown.gene_data || {}).map(([gene, d], i) => (
                <tr key={gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '6px 9px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                  <td style={{ padding: '6px 9px' }}>{d.n}</td>
                  <td style={{ padding: '6px 9px' }}>{d.mean_age} yr</td>
                  <td style={{ padding: '6px 9px' }}>{d.bilateral_n}</td>
                  <td style={{ padding: '6px 9px' }}>{d.metastatic_n}</td>
                  <td style={{ padding: '6px 9px', fontSize: 11 }}>{d.syndrome}</td>
                  <td style={{ padding: '6px 9px', fontSize: 11 }}>{d.inheritance}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 14, marginTop: 20 }}>
            {[
              { label: 'Bilateral WT predisposition', genes: breakdown.summary?.bilateral_wt_genes },
              { label: 'Unfavorable histology marker genes', genes: breakdown.summary?.unfavorable_histology_genes },
              { label: 'AFP surveillance required', genes: breakdown.summary?.surveillance_afp_required },
            ].map(item => (
              <div key={item.label} style={{ background: '#e8eaf6', borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 700, color: '#1a237e', marginBottom: 6, fontSize: 13 }}>{item.label}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                  {(item.genes || []).map(g => (
                    <span key={g} style={{ background: GENE_COLORS[g] || '#555', color: '#fff', borderRadius: 4, padding: '3px 8px', fontSize: 12, fontWeight: 600 }}>{g}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 2 && breakdown && (
        <div>
          <h3 style={{ color: '#1a237e' }}>Clinical Atlas — Per-Gene Detailed Cards</h3>
          {Object.keys(GENE_COLORS).map(gene => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} breakdown={breakdown} />
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 3 && defs && (
        <div>
          <h3 style={{ color: '#1a237e' }}>Clinical Definitions & Critical Rules</h3>

          <div style={{ background: '#fce4ec', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <b>⚠ Critical Clinical Rules — Wilms Tumour Atlas</b>
            <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 10 }}>
              {Object.entries(defs.critical_clinical_rules || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#fff', borderRadius: 6, padding: 10, fontSize: 12 }}>
                  <div style={{ fontWeight: 700, color: '#c62828', marginBottom: 3 }}>{k.replace(/-/g,' ')}</div>
                  <div>{v}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 14 }}>
            {(defs.definitions || []).map(d => (
              <div key={d.term} style={{ border: '1px solid #e8eaf6', borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 700, color: '#1a237e', marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 13 }}>{d.definition}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
