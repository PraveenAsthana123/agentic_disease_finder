'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-scid-atlas';
const TABS = ['Overview', 'Gene Table', 'Immunophenotype Map', 'Definitions'];

const GENE_COLORS = {
  'IL2RG':  '#b71c1c',  // deep red        -- XL-SCID T-B+NK-; most common 40-50%; gene therapy first
  'ADA':    '#e65100',  // burnt orange     -- ADA-SCID T-B-NK-; Strimvelis EMA2016; enzyme replacement
  'RAG1':   '#1565c0',  // deep blue        -- T-B-NK+ SCID; Omenn syndrome hypomorphic
  'RAG2':   '#0277bd',  // blue             -- T-B-NK+ SCID; R229Q North African founder; Omenn
  'DCLRE1C':'#6a1b9a',  // deep purple      -- Artemis Radiosensitive SCID; Navajo/Apache; NO TBI/busulfan
  'JAK3':   '#00695c',  // teal             -- AR T-B+NK-; female key; phenocopy XL-SCID
  'IL7R':   '#1b5e20',  // dark green       -- T-B+NK+; selective T absent; best HSCT outcome
  'AK2':    '#37474f',  // blue-grey        -- Reticular dysgenesis; T-B-NK-+agranulocytosis; bilateral deafness
};

const GENE_INFO = {
  'IL2RG':  { full: 'IL-2-receptor-γ-chain-369aa / XL-SCID-most-common-40-50pct / T-minus-B-plus-NK-minus / gene-therapy-first-retroviral / TREC-zero-NBS', locus: 'Xq13.1', size: '369 aa / 42.3 kDa', inh: 'XL' },
  'ADA':    { full: 'Adenosine-deaminase-363aa / ADA-SCID-T-minus-B-minus-NK-minus / Strimvelis-EMA2016-gene-therapy / elapegademase-enzyme-replacement / ADA-RBC-activity-absent-diagnostic', locus: '20q13.12', size: '363 aa / 40.8 kDa', inh: 'AR' },
  'RAG1':   { full: 'RAG1-recombinase-1043aa / T-minus-B-minus-NK-plus-complete-LOF / Omenn-syndrome-hypomorphic-erythroderma-eosinophilia / 11p13-sequence-RAG1-and-RAG2-always', locus: '11p13', size: '1043 aa / 119 kDa', inh: 'AR' },
  'RAG2':   { full: 'RAG2-DNA-binding-527aa / T-minus-B-minus-NK-plus / R229Q-North-African-Omenn-founder / PHD-finger-H3K4me3-chromatin-reading / 11p13-allelic-RAG1', locus: '11p13', size: '527 aa / 59.3 kDa', inh: 'AR' },
  'DCLRE1C':{ full: 'Artemis-endonuclease-692aa / RADIOSENSITIVE-SCID-T-minus-B-minus-NK-plus / Navajo-Apache-founder / busulfan-TBI-ABSOLUTELY-CI / reduced-intensity-conditioning-only', locus: '10p13', size: '692 aa / 78.3 kDa', inh: 'AR' },
  'JAK3':   { full: 'JAK3-kinase-1124aa / AR-T-minus-B-plus-NK-minus / phenocopy-XL-SCID / female-T-B+NK-minus-test-JAK3-not-IL2RG / JAK3-inhibitor-iatrogenic-SCID', locus: '19p13.11', size: '1124 aa / 124.8 kDa', inh: 'AR' },
  'IL7R':   { full: 'IL7Rα-CD127-459aa / T-minus-B-plus-NK-plus-UNIQUE / IL-7-non-redundant-human-thymic-T-development / best-HSCT-outcome-all-SCID-types / maternal-T-engraftment-40pct', locus: '5p13.2', size: '459 aa / 50 kDa', inh: 'AR' },
  'AK2':    { full: 'AK2-mitochondrial-adenylate-kinase-239aa / Reticular-dysgenesis-T-minus-B-minus-NK-minus-agranulocytosis / bilateral-congenital-sensorineural-deafness-PATHOGNOMONIC / empty-bone-marrow / fatal-3-6-weeks', locus: '1p34.1', size: '239 aa / 26.5 kDa', inh: 'AR' },
};

const PHENOTYPE_ROWS = [
  { gene: 'IL2RG',   t: '−', b: '+', nk: '−', granulocytes: '+', deafness: '−', radiosensitive: '−', frequency: '40-50%', treatment: 'HSCT / Gene therapy' },
  { gene: 'ADA',     t: '−', b: '−', nk: '−', granulocytes: '+', deafness: '−', radiosensitive: '−', frequency: '10-20%', treatment: 'HSCT / Strimvelis / Elapegademase' },
  { gene: 'RAG1',    t: '−', b: '−', nk: '+', granulocytes: '+', deafness: '−', radiosensitive: '−', frequency: '5-10%', treatment: 'HSCT (± Omenn immunosuppression)' },
  { gene: 'RAG2',    t: '−', b: '−', nk: '+', granulocytes: '+', deafness: '−', radiosensitive: '−', frequency: '3-8%', treatment: 'HSCT (± Omenn immunosuppression)' },
  { gene: 'DCLRE1C', t: '−', b: '−', nk: '+', granulocytes: '+', deafness: '−', radiosensitive: '✓✓✓', frequency: '1-5%', treatment: 'HSCT RIC ONLY — NO TBI/busulfan' },
  { gene: 'JAK3',    t: '−', b: '+', nk: '−', granulocytes: '+', deafness: '−', radiosensitive: '−', frequency: '2-6%', treatment: 'HSCT (identical to IL2RG-SCID)' },
  { gene: 'IL7R',    t: '−', b: '+', nk: '+', granulocytes: '+', deafness: '−', radiosensitive: '−', frequency: '5-10%', treatment: 'HSCT (best outcome — selective T)' },
  { gene: 'AK2',     t: '−', b: '−', nk: '−', granulocytes: '−', deafness: '✓✓✓', radiosensitive: '−', frequency: '<1%', treatment: 'Urgent HSCT (myeloablative + lymphoid)' },
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
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: color, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{cohort?.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>{info?.full?.replace(/-/g, ' ')}</div>
      {cohort?.pathognomonic && (
        <div style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 4, padding: '6px 10px', fontSize: 12, marginBottom: 8 }}>
          <strong>PATHOGNOMONIC:</strong> {cohort.pathognomonic}
        </div>
      )}
      {cohort?.first_line_treatment && (
        <div style={{ background: '#e8f5e9', border: '1px solid #4caf50', borderRadius: 4, padding: '6px 10px', fontSize: 12, marginBottom: 8 }}>
          <strong>Treatment:</strong> {cohort.first_line_treatment}
        </div>
      )}
      {cohort?.critical_rules?.length > 0 && (
        <div style={{ marginTop: 8 }}>
          {cohort.critical_rules.map(r => (
            <Badge key={r} text={r.replace(/-/g, ' ')} color={color} />
          ))}
        </div>
      )}
      {cohort?.n && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
          <Badge text={`n=${cohort.n}`} color={color} />
          <Badge text={`onset ${cohort.mean_onset_age}yr`} color="#555" />
          <Badge text={`dx-lag ${cohort.mean_dx_lag_yrs}yr`} color="#888" />
          {cohort.t_cell_absent_n > 0 && <Badge text={`T− ${cohort.t_cell_absent_n}/40`} color="#b71c1c" />}
          {cohort.b_cell_absent_n > 0 && <Badge text={`B− ${cohort.b_cell_absent_n}/40`} color="#e65100" />}
          {cohort.nk_cell_absent_n > 0 && <Badge text={`NK− ${cohort.nk_cell_absent_n}/40`} color="#6a1b9a" />}
          {cohort.omenn_n > 0 && <Badge text={`Omenn ${cohort.omenn_n}/40`} color="#c62828" />}
          {cohort.deafness_n > 0 && <Badge text={`deaf ${cohort.deafness_n}/40`} color="#37474f" />}
          {cohort.gene_therapy_eligible_n > 0 && <Badge text={`gene-Rx ${cohort.gene_therapy_eligible_n}/40`} color="#1b5e20" />}
          {cohort.hsct_curative_n > 0 && <Badge text={`HSCT-cure ${cohort.hsct_curative_n}/40`} color="#1565c0" />}
        </div>
      )}
    </div>
  );
}

export default function HereditorySCIDAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#888' }}>Loading Hereditary SCID Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  const cohortByGene = {};
  (breakdown?.genes || []).forEach(g => { cohortByGene[g.gene] = g; });

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #b71c1c 0%, #1565c0 50%, #1b5e20 100%)', borderRadius: 10, padding: '20px 28px', marginBottom: 24, color: '#fff' }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🧬 Hereditary SCID Atlas</h1>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.9 }}>
          Complete 8-Gene Reference · IL2RG · ADA · RAG1 · RAG2 · DCLRE1C · JAK3 · IL7R · AK2 · {overview?.total_patients} patients · seeds {overview?.seeds}
        </div>
        <div style={{ marginTop: 4, fontSize: 12, opacity: 0.8 }}>
          Severe Combined Immunodeficiency · T/B/NK Immunophenotype · TREC Newborn Screening · HSCT · Gene Therapy · Radiosensitivity
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(i)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
              background: tab === i ? '#b71c1c' : '#f5f5f5',
              color: tab === i ? '#fff' : '#333',
            }}
          >{t}</button>
        ))}
      </div>

      {/* Tab 0: Overview */}
      {tab === 0 && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#b71c1c' },
              { label: 'Genes Covered', value: overview.genes?.length, color: '#6a1b9a' },
              { label: 'Cohort per Gene', value: overview.cohort_size, color: '#1565c0' },
              { label: 'Seed Range', value: overview.seeds, color: '#1b5e20' },
            ].map(k => (
              <div key={k.label} style={{ background: '#fff', border: `2px solid ${k.color}`, borderRadius: 8, padding: 16, textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: k.color }}>{k.value}</div>
                <div style={{ fontSize: 13, color: '#666', marginTop: 4 }}>{k.label}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Key Clinical Rules</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 12, marginBottom: 24 }}>
            {Object.entries(overview.key_facts || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#fff8e1', border: '1px solid #ffa000', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#e65100', textTransform: 'uppercase', marginBottom: 4 }}>
                  {k.replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Diagnostic Approach by Immunophenotype</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 12, marginBottom: 24 }}>
            {Object.entries(overview.diagnostic_approach || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#e8f5e9', border: '1px solid #43a047', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#1b5e20', textTransform: 'uppercase', marginBottom: 4 }}>
                  {k.replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Gene Cohorts Overview</h2>
          {overview.cohorts?.map(c => (
            <GeneCard
              key={c.gene}
              gene={c.gene}
              color={GENE_COLORS[c.gene] || '#555'}
              info={GENE_INFO[c.gene]}
              cohort={c}
            />
          ))}
        </div>
      )}

      {/* Tab 1: Gene Table */}
      {tab === 1 && breakdown && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Gene Reference Table</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#b71c1c', color: '#fff' }}>
                  {['#', 'Gene', 'Protein', 'Locus', 'Inh.', 'Syndrome', 'n', 'Onset (yr)', 'Dx-Lag', 'T−', 'B−', 'NK−', 'Omenn', 'Deaf', 'Gene-Rx', 'HSCT-cure'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes?.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px', color: '#888' }}>{g.rank}</td>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{ background: GENE_COLORS[g.gene] || '#555', color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{g.gene}</span>
                    </td>
                    <td style={{ padding: '7px 10px', fontSize: 11, color: '#444' }}>{g.aa}aa / {g.mw_kda}kDa</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.inheritance}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.syndrome?.split('/')[0]}</td>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#555' }}>{g.n}</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_onset_age}</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_dx_lag_yrs}yr</td>
                    <td style={{ padding: '7px 10px', color: '#b71c1c', fontWeight: 700 }}>{g.t_cell_absent_n}</td>
                    <td style={{ padding: '7px 10px', color: '#e65100', fontWeight: 700 }}>{g.b_cell_absent_n}</td>
                    <td style={{ padding: '7px 10px', color: '#6a1b9a', fontWeight: 700 }}>{g.nk_cell_absent_n}</td>
                    <td style={{ padding: '7px 10px', color: '#c62828' }}>{g.omenn_n}</td>
                    <td style={{ padding: '7px 10px', color: '#37474f', fontWeight: g.deafness_n > 0 ? 700 : 400 }}>{g.deafness_n}</td>
                    <td style={{ padding: '7px 10px', color: '#1b5e20' }}>{g.gene_therapy_eligible_n}</td>
                    <td style={{ padding: '7px 10px', color: '#1565c0' }}>{g.hsct_curative_n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 20 }}>
            <h3 style={{ color: '#b71c1c', marginBottom: 12 }}>Critical Rules per Gene</h3>
            {breakdown.genes?.map(g => (
              <div key={g.gene} style={{ marginBottom: 14, padding: 12, background: '#fafafa', borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#555', marginBottom: 6 }}>{g.gene} — {g.syndrome?.split('/')[0]}</div>
                <div style={{ marginBottom: 6, fontSize: 12, color: '#444' }}>
                  <strong>PATHOGNOMONIC:</strong> {g.pathognomonic}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {g.critical_rules?.map(r => (
                    <Badge key={r} text={r.replace(/-/g, ' ')} color={GENE_COLORS[g.gene] || '#555'} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 2: Immunophenotype Map */}
      {tab === 2 && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>SCID Immunophenotype Classification</h2>

          <div style={{ background: '#fff3e0', border: '2px solid #ff6f00', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#e65100', margin: '0 0 10px' }}>TREC Newborn Screening → Immunophenotyping Guide</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
              {[
                { pattern: 'TREC = 0 → T−B+NK−', dx: 'IL2RG (male) or JAK3 (female/male IL2RG−)', color: '#b71c1c' },
                { pattern: 'TREC = 0 → T−B−NK+', dx: 'RAG1, RAG2, or DCLRE1C (Artemis); test radiosensitivity', color: '#1565c0' },
                { pattern: 'TREC = 0 → T−B−NK−', dx: 'ADA (check enzyme in RBCs) or AK2 (+ agranulocytosis + deafness)', color: '#e65100' },
                { pattern: 'TREC = 0 → T−B+NK+', dx: 'IL7R (pure T cell deficiency — BEST HSCT outcome)', color: '#1b5e20' },
              ].map(p => (
                <div key={p.pattern} style={{ background: '#fff', border: `2px solid ${p.color}`, borderRadius: 6, padding: 12 }}>
                  <div style={{ fontWeight: 700, color: p.color, marginBottom: 4 }}>{p.pattern}</div>
                  <div style={{ fontSize: 13 }}>{p.dx}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene', 'T cells', 'B cells', 'NK cells', 'Granulocytes', 'Deafness', 'Radiosensitive', 'Frequency', 'Curative Treatment'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{ background: GENE_COLORS[row.gene] || '#555', color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{row.gene}</span>
                    </td>
                    <td style={{ padding: '7px 10px', color: row.t === '−' ? '#b71c1c' : '#1b5e20', fontWeight: 700, fontSize: 16 }}>{row.t}</td>
                    <td style={{ padding: '7px 10px', color: row.b === '−' ? '#e65100' : '#1b5e20', fontWeight: 700, fontSize: 16 }}>{row.b}</td>
                    <td style={{ padding: '7px 10px', color: row.nk === '−' ? '#6a1b9a' : '#1b5e20', fontWeight: 700, fontSize: 16 }}>{row.nk}</td>
                    <td style={{ padding: '7px 10px', color: row.granulocytes === '−' ? '#c62828' : '#1b5e20', fontWeight: 700, fontSize: 16 }}>{row.granulocytes}</td>
                    <td style={{ padding: '7px 10px', color: row.deafness !== '−' ? '#37474f' : '#ccc', fontWeight: row.deafness !== '−' ? 800 : 400 }}>{row.deafness}</td>
                    <td style={{ padding: '7px 10px', color: row.radiosensitive !== '−' ? '#6a1b9a' : '#ccc', fontWeight: row.radiosensitive !== '−' ? 800 : 400 }}>{row.radiosensitive}</td>
                    <td style={{ padding: '7px 10px', fontSize: 12 }}>{row.frequency}</td>
                    <td style={{ padding: '7px 10px', fontSize: 12 }}>{row.treatment}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ background: '#fce4ec', border: '2px solid #c62828', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h3 style={{ color: '#c62828', margin: '0 0 10px' }}>⚠️ Critical Safety Rules</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14 }}>
              <li><strong>DCLRE1C/Artemis:</strong> RADIOSENSITIVE SCID — busulfan and TBI ABSOLUTELY CONTRAINDICATED. Use treosulfan-fludarabine ONLY.</li>
              <li><strong>AK2/Reticular Dysgenesis:</strong> agranulocytosis = zero innate immunity — strict isolation (positive pressure room) from birth until HSCT.</li>
              <li><strong>ALL SCID:</strong> NO live vaccines (BCG, rotavirus, MMR, varicella) until immune reconstitution confirmed post-HSCT.</li>
              <li><strong>IL7R:</strong> 40% maternal T cell engraftment — screen for maternal T cells before conditioning (GVH if missed).</li>
              <li><strong>JAK3:</strong> T-B+NK− in a female = JAK3 not IL2RG. IL2RG gene therapy does NOT correct JAK3 deficiency.</li>
            </ul>
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && defs && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Clinical Definitions</h2>
          {Object.entries(defs.definitions || {}).map(([title, text]) => (
            <div key={title} style={{ marginBottom: 20, padding: 16, background: '#fafafa', borderRadius: 8, borderLeft: '4px solid #b71c1c' }}>
              <h3 style={{ color: '#b71c1c', margin: '0 0 10px', fontSize: 15 }}>{title}</h3>
              <p style={{ margin: 0, fontSize: 13, lineHeight: 1.7, color: '#333' }}>{text}</p>
            </div>
          ))}

          <div style={{ marginTop: 24, padding: 16, background: '#e3f2fd', borderRadius: 8 }}>
            <h3 style={{ color: '#1565c0', margin: '0 0 10px' }}>Standards & References</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
              {defs.standards?.map(s => <li key={s} style={{ marginBottom: 6 }}>{s}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
