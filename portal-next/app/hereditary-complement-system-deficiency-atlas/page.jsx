'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-complement-system-deficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'Complement Pathways', 'Definitions'];

const GENE_COLORS = {
  'C1QA': '#b71c1c',  // deep red         -- C1q deficiency; SLE 93% HIGHEST autoimmunity
  'C2':   '#c62828',  // red              -- Most common complement deficiency; 28-bp deletion; SLE 50%
  'C3':   '#e65100',  // burnt orange     -- Central pivot; CH50=0 AND AH50=0; MPGN-II/DDD
  'CFB':  '#1565c0',  // deep blue        -- Alternative pathway; AH50=0, CH50=normal
  'CFP':  '#6a1b9a',  // deep purple      -- Properdin XL; meningococcal 7000x risk HIGHEST
  'C7':   '#00695c',  // teal             -- Terminal MAC; CH50=0, AH50=0, C3 normal; eculizumab CI
  'CFH':  '#1b5e20',  // dark green       -- Factor H; aHUS; eculizumab FDA2007 FIRST complement Rx
  'CD46': '#37474f',  // blue-grey        -- MCP; aHUS incomplete penetrance; flow cytometry diagnostic
};

const GENE_INFO = {
  'C1QA': { full: 'C1q-subunit-A-245aa / C1q-deficiency / AR / SLE-93pct-HIGHEST-autoimmunity / apoptotic-cell-clearance-defective / CH50-absent-AH50-normal', locus: '1p36.11', size: '245 aa / 26.7 kDa', inh: 'AR' },
  'C2':   { full: 'Complement-C2-898aa / Most-common-complement-deficiency-Europeans / AR / 28-bp-deletion-95pct-alleles / SLE-50pct / CH50-absent-AH50-normal', locus: '6p21.32', size: '898 aa / 102.6 kDa', inh: 'AR' },
  'C3':   { full: 'Complement-C3-1663aa / Central-pivot-all-3-pathways / AR / CH50-absent-AND-AH50-absent-PATHOGNOMONIC / encapsulated-bacteria-most-severe / MPGN-type-II-DDD', locus: '19p13.3', size: '1663 aa / 187.1 kDa', inh: 'AR' },
  'CFB':  { full: 'Factor-B-764aa / Alternative-pathway-C3-convertase / AR-LOF-rare / AH50-absent-CH50-normal / GOF-causes-C3-glomerulopathy-OPPOSITE', locus: '6p21.32', size: '764 aa / 86.9 kDa', inh: 'AR' },
  'CFP':  { full: 'Properdin-442aa / ONLY-positive-regulator-complement / XL-males-hemizygous / meningococcal-7000x-risk-HIGHEST / AH50-reduced-CH50-normal', locus: 'Xp11.23', size: '442 aa / 53.1 kDa', inh: 'XL' },
  'C7':   { full: 'Complement-C7-843aa / Terminal-MAC-C5b6789 / AR / CH50-absent-AH50-absent-C3-C4-NORMAL-PATHOGNOMONIC / neisserial-tropism / Japan-South-Africa-founder / eculizumab-CI', locus: '5p13.1', size: '843 aa / 92.0 kDa', inh: 'AR' },
  'CFH':  { full: 'Factor-H-1231aa / Primary-AP-negative-regulator / AD-LOF / aHUS-TMA-triad-PATHOGNOMONIC / MPGN-II-biallelic / eculizumab-FDA2007-FIRST-complement-Rx / meningococcal-vaccine-MANDATORY', locus: '1q31.3', size: '1231 aa / 139.1 kDa', inh: 'AD' },
  'CD46': { full: 'MCP-CD46-347aa / Membrane-cofactor-protein / AD-LOF / aHUS-incomplete-penetrance-50pct / trigger-dependent-pregnancy-infection / MCP-absent-flow-cytometry-DIAGNOSTIC / eculizumab-discontinuable', locus: '1q32.2', size: '347 aa / 35.5 kDa', inh: 'AD' },
};

const PATHWAY_ROWS = [
  { gene: 'C1QA', pathway: 'Classical',    component: 'C1q recognition', ch50: '0 (absent)', ah50: 'Normal', c3: 'Normal', c4: 'Normal/Low', phenotype: 'SLE 93%', meningococcal: '25%' },
  { gene: 'C2',   pathway: 'Classical',    component: 'C2 serine protease', ch50: '0 (absent)', ah50: 'Normal', c3: 'Normal', c4: 'Normal/slightly ↓', phenotype: 'SLE 50%', meningococcal: '35%' },
  { gene: 'C3',   pathway: 'All (central)',component: 'C3 central pivot', ch50: '0 (absent)', ah50: '0 (absent)', c3: 'LOW', c4: 'Normal', phenotype: 'Severe infections', meningococcal: '70%' },
  { gene: 'CFB',  pathway: 'Alternative',  component: 'C3bBb convertase', ch50: 'Normal', ah50: '0 (absent)', c3: 'Normal', c4: 'Normal', phenotype: 'Infections', meningococcal: '55%' },
  { gene: 'CFP',  pathway: 'Alternative',  component: 'Properdin stabiliser', ch50: 'Normal', ah50: 'Reduced/0', c3: 'Normal', c4: 'Normal', phenotype: 'Meningococcal 7000×', meningococcal: '95%' },
  { gene: 'C7',   pathway: 'Terminal MAC', component: 'MAC assembly C5b-6-7', ch50: '0 (absent)', ah50: '0 (absent)', c3: 'NORMAL', c4: 'NORMAL', phenotype: 'Neisserial tropism', meningococcal: '90%' },
  { gene: 'CFH',  pathway: 'Regulation AP',component: 'Factor H AP regulator', ch50: 'Variable', ah50: 'Reduced', c3: 'Low', c4: 'Normal/Low', phenotype: 'aHUS / MPGN-II', meningococcal: '15%' },
  { gene: 'CD46', pathway: 'Regulation AP',component: 'MCP cell-surface cofactor', ch50: 'Variable', ah50: 'Variable', c3: 'Normal/Low', c4: 'Normal', phenotype: 'aHUS (triggers)', meningococcal: '5%' },
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
          {cohort.sle_n > 0 && <Badge text={`SLE ${cohort.sle_n}/40`} color="#b71c1c" />}
          {cohort.bacterial_infection_n > 0 && <Badge text={`infections ${cohort.bacterial_infection_n}/40`} color="#e65100" />}
          {cohort.meningococcal_n > 0 && <Badge text={`meningococcal ${cohort.meningococcal_n}/40`} color="#6a1b9a" />}
          {cohort.ahus_n > 0 && <Badge text={`aHUS ${cohort.ahus_n}/40`} color="#1b5e20" />}
          {cohort.renal_involvement_n > 0 && <Badge text={`renal ${cohort.renal_involvement_n}/40`} color="#0277bd" />}
          {cohort.stroke_n > 0 && <Badge text={`stroke ${cohort.stroke_n}/40`} color="#c62828" />}
        </div>
      )}
    </div>
  );
}

export default function HereditaryComplementSystemDeficiencyAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#888' }}>Loading Hereditary Complement System Deficiency Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  const cohortByGene = {};
  (breakdown?.genes || []).forEach(g => { cohortByGene[g.gene] = g; });

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #b71c1c 0%, #1b5e20 50%, #1565c0 100%)', borderRadius: 10, padding: '20px 28px', marginBottom: 24, color: '#fff' }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🧬 Hereditary Complement System Deficiency Atlas</h1>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.9 }}>
          Complete 8-Gene Reference · C1QA · C2 · C3 · CFB · CFP · C7 · CFH · CD46 · {overview?.total_patients} patients · seeds {overview?.seeds}
        </div>
        <div style={{ marginTop: 4, fontSize: 12, opacity: 0.8 }}>
          Classical Pathway · Alternative Pathway · Terminal MAC · Complement Regulation · aHUS · SLE-like · Meningococcal Risk
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

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Diagnostic Approach</h2>
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
                  {['#', 'Gene', 'Protein', 'Locus', 'Inh.', 'Pathway', 'Syndrome', 'n', 'Onset (yr)', 'Dx-Lag', 'SLE', 'Infections', 'Meningo.', 'aHUS', 'Renal', 'Stroke'].map(h => (
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
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.mechanism}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.syndrome?.split('/')[0]}</td>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#555' }}>{g.n}</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_onset_age}</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_dx_lag_yrs}yr</td>
                    <td style={{ padding: '7px 10px' }}>{g.sle_n}</td>
                    <td style={{ padding: '7px 10px' }}>{g.bacterial_infection_n}</td>
                    <td style={{ padding: '7px 10px', color: '#6a1b9a', fontWeight: 700 }}>{g.meningococcal_n}</td>
                    <td style={{ padding: '7px 10px', color: '#1b5e20', fontWeight: 700 }}>{g.ahus_n}</td>
                    <td style={{ padding: '7px 10px' }}>{g.renal_involvement_n}</td>
                    <td style={{ padding: '7px 10px', color: '#c62828' }}>{g.stroke_n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 2: Complement Pathways */}
      {tab === 2 && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Complement Pathway Classification</h2>
          <div style={{ background: '#fff3e0', border: '2px solid #ff6f00', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#e65100', margin: '0 0 10px' }}>CH50 / AH50 Interpretation Matrix</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
              {[
                { pattern: 'CH50=0, AH50=Normal', dx: 'Classical pathway (C1q/C1r/C1s → C4 → C2)', color: '#b71c1c' },
                { pattern: 'CH50=0, AH50=0, C3 LOW', dx: 'C3 deficiency — central pivot', color: '#e65100' },
                { pattern: 'CH50=0, AH50=0, C3 Normal', dx: 'Terminal MAC (C5-C9) deficiency', color: '#00695c' },
                { pattern: 'AH50=0, CH50=Normal', dx: 'Alternative pathway (CFB or CFP/properdin)', color: '#1565c0' },
              ].map(p => (
                <div key={p.pattern} style={{ background: '#fff', border: `2px solid ${p.color}`, borderRadius: 6, padding: 12 }}>
                  <div style={{ fontWeight: 700, color: p.color, marginBottom: 4 }}>{p.pattern}</div>
                  <div style={{ fontSize: 13 }}>{p.dx}</div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene', 'Pathway', 'Component', 'CH50', 'AH50', 'C3', 'C4', 'Phenotype', 'Meningococcal %'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PATHWAY_ROWS.map((r, i) => (
                  <tr key={r.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{ background: GENE_COLORS[r.gene] || '#555', color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{r.gene}</span>
                    </td>
                    <td style={{ padding: '7px 10px', fontSize: 12 }}>{r.pathway}</td>
                    <td style={{ padding: '7px 10px', fontSize: 12 }}>{r.component}</td>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: r.ch50.includes('0') && !r.ch50.includes('Variable') ? '#c00' : '#1b5e20' }}>{r.ch50}</td>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: r.ah50 === 'Normal' ? '#1b5e20' : '#c00' }}>{r.ah50}</td>
                    <td style={{ padding: '7px 10px' }}>{r.c3}</td>
                    <td style={{ padding: '7px 10px' }}>{r.c4}</td>
                    <td style={{ padding: '7px 10px', fontSize: 12 }}>{r.phenotype}</td>
                    <td style={{ padding: '7px 10px', color: '#6a1b9a', fontWeight: 700 }}>{r.meningococcal}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 16, marginTop: 28 }}>Eculizumab/Ravulizumab Warning</h2>
          <div style={{ background: '#ffebee', border: '2px solid #c62828', borderRadius: 8, padding: 16 }}>
            <div style={{ fontWeight: 700, color: '#c62828', marginBottom: 8 }}>⚠ MENINGOCOCCAL VACCINES MANDATORY BEFORE ECULIZUMAB</div>
            <div style={{ fontSize: 13, marginBottom: 8 }}>
              Eculizumab/ravulizumab block C5 → prevent MAC assembly → 1000–2000× meningococcal risk.
              Meningococcal ACWY conjugate + serogroup B (Bexsero/Trumenba) vaccines ≥2 weeks before treatment.
            </div>
            <div style={{ fontWeight: 700, color: '#c62828', marginBottom: 8 }}>⛔ ECULIZUMAB CONTRAINDICATED IN C5–C9 TERMINAL DEFICIENCY</div>
            <div style={{ fontSize: 13 }}>
              Patients with C7 (or C5/C6/C8/C9) deficiency have NO MAC — eculizumab blocks their only remaining neisserial defence.
              Anti-C5 is absolutely contraindicated in terminal complement deficiencies.
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && defs && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Clinical Definitions & Standards</h2>
          {Object.entries(defs.definitions || {}).map(([k, v]) => (
            <div key={k} style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <h3 style={{ color: '#b71c1c', margin: '0 0 8px', fontSize: 15 }}>{k}</h3>
              <p style={{ margin: 0, fontSize: 13, lineHeight: 1.7, color: '#333' }}>{v}</p>
            </div>
          ))}
          {defs.standards?.length > 0 && (
            <div style={{ background: '#e3f2fd', border: '1px solid #1565c0', borderRadius: 8, padding: 16, marginTop: 16 }}>
              <h3 style={{ color: '#1565c0', margin: '0 0 10px' }}>Clinical Standards & References</h3>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {defs.standards.map(s => <li key={s} style={{ marginBottom: 6, fontSize: 13 }}>{s}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
