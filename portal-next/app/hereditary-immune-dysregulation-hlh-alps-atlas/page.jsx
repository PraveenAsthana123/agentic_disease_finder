'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-immune-dysregulation-hlh-alps-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'PRF1':   '#b71c1c',  // deep red           -- AR; FHL2; perforin absent; most common FHL
  'UNC13D': '#880e4f',  // deep magenta       -- AR; FHL3; MUNC13-4; perforin PRESENT; CD107a absent
  'STX11':  '#e65100',  // burnt orange       -- AR; FHL4; Kurdish/Turkish founder; SNARE docking
  'STXBP2': '#e65100',  // orange-brown       -- AR; FHL5; MUNC18-2; IBD-like enteropathy; neonatal
  'LYST':   '#1a237e',  // deep navy          -- AR; Chediak-Higashi; GIANT GRANULES PATHOGNOMONIC
  'RAB27A': '#004d40',  // dark teal          -- AR; Griscelli type 2; SILVERY HAIR PATHOGNOMONIC
  'FAS':    '#4a148c',  // deep purple        -- AD/AR; ALPS TypeIa; DNT cells PATHOGNOMONIC; Evans
  'FOXP3':  '#1b5e20',  // dark green         -- XL; IPEX; neonatal T1DM PATHOGNOMONIC; Treg absent
};

const GENE_INFO = {
  'PRF1':   { full: 'PRF1-555aa / AR-FHL2 / Perforin-Absent-PATHOGNOMONIC / Most-Common-FHL-30-40pct / Ferritin-GT10000 / CD107a-Absent / HSCT-Only-Cure', locus: '10q22.1', size: '555 aa / 61 kDa',  inh: 'AR' },
  'UNC13D': { full: 'UNC13D-MUNC13-4-1090aa / AR-FHL3 / CD107a-Absent-Perforin-PRESENT-DISTINGUISHER / Vesicle-Priming-Defect / HLH-2004-HSCT', locus: '17q25.3', size: '1090 aa / 122 kDa', inh: 'AR' },
  'STX11':  { full: 'STX11-287aa / AR-FHL4 / Kurdish-Turkish-Founder / SNARE-Docking-Defect / STX11-STXBP2-Interdependency / CD107a-Absent', locus: '6q24.2',  size: '287 aa / 33 kDa',  inh: 'AR' },
  'STXBP2': { full: 'STXBP2-MUNC18-2-592aa / AR-FHL5 / IBD-Like-Enteropathy-DISTINGUISHING / Neonatal-HLH / STX11-Chaperone / STX11-Protein-Reduced', locus: '19p13.2', size: '592 aa / 67 kDa',  inh: 'AR' },
  'LYST':   { full: 'LYST-3801aa / AR-Chediak-Higashi / GIANT-GRANULES-Neutrophils-PATHOGNOMONIC / Partial-Albinism / 80-90pct-HLH-Accelerated-Phase / Neurodegeneration-Persists-Post-HSCT', locus: '1q42.3',  size: '3801 aa / 430 kDa', inh: 'AR' },
  'RAB27A': { full: 'RAB27A-221aa / AR-Griscelli-Type2 / SILVERY-METALLIC-HAIR-PATHOGNOMONIC / No-Giant-Granules-vs-LYST / HLH / HSCT-Curative-No-Neurodegeneration', locus: '15q21.3', size: '221 aa / 25 kDa',  inh: 'AR' },
  'FAS':    { full: 'FAS-CD95-335aa / AD-TypeIa-AR-TypeIb / ALPS / DNT-Cells-Elevated-PATHOGNOMONIC / Evans-Syndrome / Lymphoma-10-15x / Sirolimus-First-Line', locus: '10q23.31', size: '335 aa / 36 kDa', inh: 'AD/AR' },
  'FOXP3':  { full: 'FOXP3-431aa / XL-IPEX / Neonatal-T1DM-Birth-6wks-PATHOGNOMONIC / Enteropathy / Eczema / Treg-Absent / Calcineurin-Inhibitor-Bridge / HSCT-Urgent', locus: 'Xp11.22', size: '431 aa / 47 kDa',  inh: 'XL' },
};

const PHENOTYPE_ROWS = [
  { gene: 'PRF1',   group: 'FHL-Granule', hlh: '✓✓✓', cd107a: '✗',   perforin: '✗',   giant_g: '−',   silver_h: '−',   dnt: '−',   evans: '−',   neo_t1dm: '−', unique: 'Ferritin >10,000 + perforin ABSENT (flow) — FHL2; HLH-2004 → HSCT only cure' },
  { gene: 'UNC13D', group: 'FHL-Granule', hlh: '✓✓✓', cd107a: '✗',   perforin: '✓',   giant_g: '−',   silver_h: '−',   dnt: '−',   evans: '−',   neo_t1dm: '−', unique: 'CD107a absent + PERFORIN PRESENT — KEY FHL3 vs FHL2 distinction; MUNC13-4 priming defect' },
  { gene: 'STX11',  group: 'FHL-Granule', hlh: '✓✓✓', cd107a: '✗',   perforin: '✓',   giant_g: '−',   silver_h: '−',   dnt: '−',   evans: '−',   neo_t1dm: '−', unique: 'Kurdish/Turkish consanguineous; CD107a absent; STXBP2-STX11 interdependency: check both' },
  { gene: 'STXBP2', group: 'FHL-Granule', hlh: '✓✓✓', cd107a: '✗',   perforin: '✓',   giant_g: '−',   silver_h: '−',   dnt: '−',   evans: '±',   neo_t1dm: '−', unique: 'Neonatal HLH + IBD-like enteropathy DISTINGUISHING; STX11 protein reduced even when STX11 intact' },
  { gene: 'LYST',   group: 'LRO',         hlh: '✓✓',  cd107a: '✗',   perforin: '±',   giant_g: '✓✓✓',silver_h: '✓✓', dnt: '−',   evans: '−',   neo_t1dm: '−', unique: 'GIANT GRANULES on blood film PATHOGNOMONIC; neurodegeneration continues post-HSCT' },
  { gene: 'RAB27A', group: 'LRO',         hlh: '✓✓',  cd107a: '✗',   perforin: '±',   giant_g: '−',   silver_h: '✓✓✓',dnt: '−',   evans: '−',   neo_t1dm: '−', unique: 'SILVERY HAIR PATHOGNOMONIC; no giant granules vs LYST; HSCT curative (no neurodegeneration)' },
  { gene: 'FAS',    group: 'ALPS',        hlh: '±',   cd107a: '−',   perforin: '−',   giant_g: '−',   silver_h: '−',   dnt: '✓✓✓',evans: '✓✓', neo_t1dm: '−', unique: 'DNT cells >1.5% PATHOGNOMONIC; Evans syndrome; lymphoma 10–15×; sirolimus; avoid splenectomy' },
  { gene: 'FOXP3',  group: 'Treg',        hlh: '±',   cd107a: '−',   perforin: '−',   giant_g: '−',   silver_h: '−',   dnt: '−',   evans: '±',   neo_t1dm: '✓✓✓', unique: 'Neonatal T1DM (birth–6 weeks) + enteropathy + eczema PATHOGNOMONIC; X-linked males; HSCT urgent' },
];

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, cohort }) {
  const groupLabel = ['PRF1','UNC13D','STX11','STXBP2'].includes(gene) ? 'FHL Granule Secretion'
                   : ['LYST','RAB27A'].includes(gene) ? 'LRO Trafficking Defect'
                   : gene === 'FAS' ? 'ALPS / Lymphoproliferation'
                   : 'IPEX / Treg Defect';
  const groupColor = ['PRF1','UNC13D','STX11','STXBP2'].includes(gene) ? '#b71c1c'
                   : ['LYST','RAB27A'].includes(gene) ? '#1a237e'
                   : gene === 'FAS' ? '#4a148c'
                   : '#1b5e20';
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: color, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div style={{ background: groupColor, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>{groupLabel}</div>
        <div>
          <div style={{ fontWeight: 600, fontSize: 13 }}>{cohort?.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      {cohort?.pathognomonic && (
        <div style={{ background: '#fff8e1', border: '1px solid #ffc107', borderRadius: 4, padding: '6px 10px', marginBottom: 8, fontSize: 12 }}>
          <strong>PATHOGNOMONIC:</strong> {cohort.pathognomonic}
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {cohort?.hlh_pct > 50 && <Badge text={`HLH ${cohort.hlh_pct}%`} color="#c62828" />}
        {cohort?.cd107a_absent_pct > 50 && <Badge text={`CD107a absent ${cohort.cd107a_absent_pct}%`} color="#ad1457" />}
        {cohort?.perforin_absent_pct > 50 && <Badge text={`Perforin absent ${cohort.perforin_absent_pct}%`} color="#4e342e" />}
        {cohort?.perforin_absent_pct < 20 && cohort?.cd107a_absent_pct > 50 && <Badge text="Perforin PRESENT" color="#388e3c" />}
        {cohort?.giant_granules_pct > 50 && <Badge text={`Giant granules ${cohort.giant_granules_pct}%`} color="#1565c0" />}
        {cohort?.silver_hair_pct > 50 && <Badge text={`Silvery hair ${cohort.silver_hair_pct}%`} color="#00695c" />}
        {cohort?.dnt_elevated_pct > 50 && <Badge text={`DNT elevated ${cohort.dnt_elevated_pct}%`} color="#6a1b9a" />}
        {cohort?.evans_pct > 30 && <Badge text={`Evans ${cohort.evans_pct}%`} color="#7b1fa2" />}
        {cohort?.neonatal_t1dm_pct > 50 && <Badge text={`Neonatal T1DM ${cohort.neonatal_t1dm_pct}%`} color="#2e7d32" />}
        {cohort?.enteropathy_pct > 30 && <Badge text={`Enteropathy ${cohort.enteropathy_pct}%`} color="#5d4037" />}
        {cohort?.hsct_performed_pct > 40 && <Badge text={`HSCT ${cohort.hsct_performed_pct}%`} color="#0277bd" />}
      </div>
    </div>
  );
}

export default function HeredImmuneDysregHlhAlpsAtlas() {
  const [tab, setTab]       = useState('Overview');
  const [overview, setOv]   = useState(null);
  const [breakdown, setBk]  = useState(null);
  const [definitions, setDf]= useState(null);
  const [loading, setLoading]= useState(false);
  const [error, setError]   = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOv(ov); setBk(bk); setDf(df); setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Immune Dysregulation HLH/ALPS Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const cohortMap = {};
  if (breakdown?.gene_cohorts) {
    breakdown.gene_cohorts.forEach(c => { cohortMap[c.gene] = c; });
  }

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <div style={{ background: 'linear-gradient(135deg,#b71c1c,#4a148c)', color: '#fff', borderRadius: 10, padding: '24px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>🧬 Hereditary-Immune-Dysregulation-HLH-ALPS-Atlas</h1>
        <p style={{ margin: '8px 0 0', opacity: 0.9, fontSize: 14 }}>
          Complete 8-Gene Reference — PRF1 · UNC13D · STX11 · STXBP2 · LYST · RAB27A · FAS · FOXP3
          <br/>Familial HLH (FHL2-5) · Chediak-Higashi · Griscelli Type 2 · ALPS · IPEX | IUIS PID Category 4
        </p>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text="320 Patients (8×40)" color="rgba(255,255,255,0.2)" />
          <Badge text="Seeds 3894–3901" color="rgba(255,255,255,0.2)" />
          <Badge text="Ferritin >10k PATHOGNOMONIC" color="#c62828" />
          <Badge text="CD107a Mobilisation Assay" color="#ad1457" />
          <Badge text="GIANT GRANULES PATHOGNOMONIC" color="#1565c0" />
          <Badge text="SILVERY HAIR PATHOGNOMONIC" color="#00695c" />
          <Badge text="DNT Cells PATHOGNOMONIC" color="#6a1b9a" />
          <Badge text="Neonatal T1DM PATHOGNOMONIC" color="#2e7d32" />
          <Badge text="HSCT Only Cure (FHL/GS/CHS)" color="#0277bd" />
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 14,
              background: tab === t ? '#b71c1c' : '#f5f5f5',
              color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400
            }}>{t}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 12, marginBottom: 24 }}>
            {[
              { label: 'HLH', value: `${overview.aggregate_stats?.hlh_pct}%`, color: '#c62828' },
              { label: 'Ferritin >10k', value: `${overview.aggregate_stats?.ferritin_gt10k_pct}%`, color: '#b71c1c' },
              { label: 'CD107a Absent', value: `${overview.aggregate_stats?.cd107a_absent_pct}%`, color: '#ad1457' },
              { label: 'Giant Granules', value: `${overview.aggregate_stats?.giant_granules_pct}%`, color: '#1565c0' },
              { label: 'Silvery Hair', value: `${overview.aggregate_stats?.silver_hair_pct}%`, color: '#00695c' },
              { label: 'DNT Elevated', value: `${overview.aggregate_stats?.dnt_elevated_pct}%`, color: '#6a1b9a' },
              { label: 'Neonatal T1DM', value: `${overview.aggregate_stats?.neonatal_t1dm_pct}%`, color: '#2e7d32' },
              { label: 'HSCT Performed', value: `${overview.aggregate_stats?.hsct_performed_pct}%`, color: '#0277bd' },
            ].map(s => (
              <div key={s.label} style={{ background: '#fff', border: '1px solid #eee', borderRadius: 8, padding: 16, textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: s.color }}>{s.value}</div>
                <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>{s.label}</div>
              </div>
            ))}
          </div>

          <div style={{ marginBottom: 20 }}>
            <h3 style={{ color: '#b71c1c', borderBottom: '2px solid #b71c1c', paddingBottom: 6 }}>Diagnostic Groups — Mechanistic Classification</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {Object.entries(breakdown?.diagnostic_groups || {}).map(([key, grp]) => (
                <div key={key} style={{ border: '1px solid #eee', borderRadius: 8, padding: 16 }}>
                  <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 6 }}>{grp.genes?.join(' · ')}</div>
                  <div style={{ fontSize: 13, color: '#333', marginBottom: 6 }}><strong>Mechanism:</strong> {grp.mechanism}</div>
                  <div style={{ fontSize: 12, color: '#555', marginBottom: 4 }}><strong>Shared:</strong> {grp.shared_feature}</div>
                  <div style={{ fontSize: 12, color: '#7b1fa2' }}><strong>DDx:</strong> {grp.key_ddx}</div>
                </div>
              ))}
            </div>
          </div>

          <h3 style={{ color: '#b71c1c', borderBottom: '2px solid #b71c1c', paddingBottom: 6 }}>Gene Cards</h3>
          {GENES.map(g => (
            <GeneCard key={g} gene={g} color={GENE_COLORS[g]} info={GENE_INFO[g]} cohort={cohortMap[g]} />
          ))}

          <div style={{ marginTop: 24 }}>
            <h3 style={{ color: '#4a148c', borderBottom: '2px solid #4a148c', paddingBottom: 6 }}>CASCADE Testing Concepts</h3>
            <ol style={{ paddingLeft: 20 }}>
              {(overview.cascade_testing_concepts || []).map((c, i) => (
                <li key={i} style={{ marginBottom: 8, fontSize: 13 }}>{c}</li>
              ))}
            </ol>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && (
        <div>
          <h3 style={{ color: '#b71c1c' }}>Gene-by-Gene Phenotype Comparison — Immune Dysregulation Panel</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#b71c1c', color: '#fff' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Gene</th>
                  <th style={{ padding: '8px 10px' }}>Group</th>
                  <th style={{ padding: '8px 10px' }}>HLH</th>
                  <th style={{ padding: '8px 10px' }}>CD107a</th>
                  <th style={{ padding: '8px 10px' }}>Perforin</th>
                  <th style={{ padding: '8px 10px' }}>Giant G.</th>
                  <th style={{ padding: '8px 10px' }}>Silver H.</th>
                  <th style={{ padding: '8px 10px' }}>DNT↑</th>
                  <th style={{ padding: '8px 10px' }}>Evans</th>
                  <th style={{ padding: '8px 10px' }}>Neo T1DM</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Unique / Pathognomonic</th>
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center', fontSize: 11 }}>{row.group}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{row.hlh}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center', color: row.cd107a === '✗' ? '#c62828' : '#333' }}>{row.cd107a}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center', color: row.perforin === '✗' ? '#c62828' : row.perforin === '✓' ? '#2e7d32' : '#333' }}>{row.perforin}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{row.giant_g}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{row.silver_h}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{row.dnt}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{row.evans}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{row.neo_t1dm}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 style={{ color: '#1a237e', marginTop: 24 }}>Gene Metadata</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))', gap: 12 }}>
            {GENES.map(g => {
              const m = breakdown?.gene_cohorts?.find(c => c.gene === g);
              return (
                <div key={g} style={{ border: `2px solid ${GENE_COLORS[g]}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[g], fontSize: 16, marginBottom: 6 }}>{g}</div>
                  <div style={{ fontSize: 12 }}><strong>Locus:</strong> {m?.locus}</div>
                  <div style={{ fontSize: 12 }}><strong>Size:</strong> {m?.aa} aa / {m?.kda} kDa</div>
                  <div style={{ fontSize: 12 }}><strong>Inheritance:</strong> {m?.inheritance}</div>
                  <div style={{ fontSize: 12 }}><strong>OMIM gene:</strong> {m?.mim_gene}</div>
                  <div style={{ fontSize: 12 }}><strong>OMIM disease:</strong> {m?.mim_disease}</div>
                  <div style={{ fontSize: 12, marginTop: 6, fontStyle: 'italic', color: '#555' }}>{m?.syndrome?.replace(/-/g,' ')}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* PATHWAY MAP TAB */}
      {tab === 'Pathway Map' && (
        <div>
          <h3 style={{ color: '#b71c1c' }}>Immune Dysregulation Pathway — Disease Groups</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            {/* FHL Granule Secretion */}
            <div style={{ border: '2px solid #b71c1c', borderRadius: 10, padding: 16 }}>
              <div style={{ background: '#b71c1c', color: '#fff', borderRadius: 6, padding: '6px 12px', fontWeight: 700, marginBottom: 12 }}>
                FHL — Cytotoxic Granule Secretion Defects (FHL2–5)
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                <div>🔵 CTL/NK activated → lytic granule movement to immunological synapse</div>
                <div style={{ color: '#b71c1c', fontWeight: 600 }}>❌ PRF1 — pore-forming protein absent → granules exit but CANNOT kill (FHL2)</div>
                <div style={{ color: '#880e4f', fontWeight: 600 }}>❌ UNC13D — priming step blocked → granules dock but CANNOT fuse (FHL3)</div>
                <div style={{ color: '#e65100', fontWeight: 600 }}>❌ STX11 — SNARE docking blocked → granules arrive but CANNOT dock (FHL4)</div>
                <div style={{ color: '#e65100', fontWeight: 600 }}>❌ STXBP2 — STX11 chaperone absent → STX11 degraded → FHL5 + IBD</div>
                <div style={{ marginTop: 8, fontSize: 12, color: '#555' }}>→ All FHL: NK CD107a absent (K562 assay). Differentiate by perforin expression (absent=FHL2, present=FHL3-5).</div>
              </div>
            </div>
            {/* LRO Trafficking */}
            <div style={{ border: '2px solid #1a237e', borderRadius: 10, padding: 16 }}>
              <div style={{ background: '#1a237e', color: '#fff', borderRadius: 6, padding: '6px 12px', fontWeight: 700, marginBottom: 12 }}>
                LRO Trafficking Defects — Chediak-Higashi & Griscelli Type 2
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                <div>🔵 Lysosome-related organelles (LROs) in CTLs, melanocytes, neutrophils</div>
                <div style={{ color: '#1a237e', fontWeight: 600 }}>❌ LYST — LRO fission defective → GIANT GRANULES all cell types → PATHOGNOMONIC on blood film</div>
                <div style={{ color: '#004d40', fontWeight: 600 }}>❌ RAB27A — granule tethering fails → granules NORMAL SIZE but CANNOT dock → no giant granules</div>
                <div style={{ marginTop: 8, fontSize: 12, color: '#555' }}>Both: partial albinism (silvery hair) + HLH. Key DDx: blood film (LYST = giant granules; RAB27A = normal). LYST: neurodegeneration post-HSCT persists. RAB27A: HSCT fully curative.</div>
              </div>
            </div>
            {/* ALPS */}
            <div style={{ border: '2px solid #4a148c', borderRadius: 10, padding: 16 }}>
              <div style={{ background: '#4a148c', color: '#fff', borderRadius: 6, padding: '6px 12px', fontWeight: 700, marginBottom: 12 }}>
                ALPS — Autoimmune Lymphoproliferative Syndrome (FAS/CD95)
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                <div>🔵 Antigen cleared → activated lymphocytes should undergo RICD via FAS-FASL apoptosis</div>
                <div style={{ color: '#4a148c', fontWeight: 600 }}>❌ FAS (AD/AR) — RICD fails → lymphocyte accumulation → lymphadenopathy + splenomegaly</div>
                <div>→ Accumulated T cells → double-negative T cells (DNT: CD4–CD8–B220+TCRαβ+) PATHOGNOMONIC</div>
                <div>→ Autoimmunity: Evans syndrome (AIHA + ITP); cytopenias</div>
                <div style={{ marginTop: 8, fontSize: 12, color: '#555' }}>Treatment: sirolimus (not steroids) for cytopenias. Lymphoma 10–15× risk — mandatory surveillance. Avoid splenectomy.</div>
              </div>
            </div>
            {/* IPEX */}
            <div style={{ border: '2px solid #1b5e20', borderRadius: 10, padding: 16 }}>
              <div style={{ background: '#1b5e20', color: '#fff', borderRadius: 6, padding: '6px 12px', fontWeight: 700, marginBottom: 12 }}>
                IPEX — Immune Polyendocrinopathy Enteropathy X-linked (FOXP3)
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                <div>🔵 Normal: FOXP3+ Tregs suppress autoreactive T cells in every organ</div>
                <div style={{ color: '#1b5e20', fontWeight: 600 }}>❌ FOXP3 absent → NO Tregs → autoreactive T cells attack pancreas + gut + skin from birth</div>
                <div>→ Neonatal T1DM (birth–6 weeks) PATHOGNOMONIC + intractable enteropathy + eczema</div>
                <div style={{ marginTop: 8, fontSize: 12, color: '#555' }}>X-linked: males only. Calcineurin inhibitor bridge. HSCT urgently — every month delay = more organ destruction. Matched sibling best.</div>
              </div>
            </div>
          </div>

          {/* Pathognomonic Features Summary */}
          <h3 style={{ color: '#b71c1c' }}>Pathognomonic Features — Quick Reference</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))', gap: 10 }}>
            {Object.entries(breakdown?.pathognomonic_features || {}).map(([key, val]) => (
              <div key={key} style={{ background: '#fff8e1', border: '1px solid #ffc107', borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 700, color: '#e65100', marginBottom: 4, fontSize: 12 }}>{key.replace(/_/g,' ')}</div>
                <div style={{ fontSize: 12 }}>{val}</div>
              </div>
            ))}
          </div>

          {/* Key Distinguishers */}
          <h3 style={{ color: '#4a148c', marginTop: 20 }}>Critical Distinguishers</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {Object.entries(definitions?.key_distinguishers || {}).map(([key, val]) => (
              <div key={key} style={{ border: '1px solid #ce93d8', borderRadius: 8, padding: 12, background: '#f3e5f5' }}>
                <div style={{ fontWeight: 700, color: '#4a148c', marginBottom: 4, fontSize: 12 }}>{key.replace(/_/g,' ')}</div>
                <div style={{ fontSize: 12 }}>{val}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#b71c1c' }}>Gene Definitions — Mechanisms and Clinical Essentials</h3>
          {Object.entries(definitions.definitions || {}).map(([gene, def]) => (
            <div key={gene} style={{ border: `2px solid ${GENE_COLORS[gene]}`, borderRadius: 8, padding: 16, marginBottom: 14 }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 16, marginBottom: 6 }}>{gene}</div>
              <div style={{ fontSize: 13, lineHeight: 1.6 }}>{def}</div>
            </div>
          ))}

          <h3 style={{ color: '#4a148c', marginTop: 24 }}>Diagnostic Pathway — Step-by-Step</h3>
          <ol style={{ paddingLeft: 20 }}>
            {(definitions.diagnostic_pathway || []).map((step, i) => (
              <li key={i} style={{ marginBottom: 10, fontSize: 13 }}>{step}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
