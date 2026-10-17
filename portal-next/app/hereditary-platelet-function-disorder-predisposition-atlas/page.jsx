'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-platelet-function-disorder-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ITGA2B': '#1a237e',  // deep indigo       -- GT-Type-I; ALL-agonist-ABSENT; eptifibatide-CI-ABSOLUTELY; rFVIIa
  'ITGB3':  '#880e4f',  // deep crimson       -- GT-Type-II; NAITP-HPA-1a; eptifibatide-CI-ABSOLUTELY; maternal-IVIG
  'GP1BA':  '#004d40',  // deep teal          -- BSS-TypeA1; giant-platelets-PATHOGNOMONIC; DDAVP-INEFFECTIVE; rFVIIa
  'GP1BB':  '#bf360c',  // burnt orange       -- BSS-22q; DiGeorge-cardiac-75pct-PATHOGNOMONIC; MLPA-MANDATORY
  'NBEAL2': '#4a148c',  // deep purple        -- GPS; alpha-granule-ABSENT; myelofibrosis-35pct; splenectomy-AVOID
  'MYH9':   '#006064',  // dark cyan          -- MYH9-RD; Dohle-bodies-PATHOGNOMONIC; BLEEDING-MILD; ACEi-nephritis
  'HPS1':   '#33691e',  // olive green        -- HPS1; OCA-PATHOGNOMONIC; pulmonary-fibrosis-DOMINANT-DEATH; pirfenidone
  'P2RY12': '#e65100',  // deep amber         -- P2Y12-deficiency; thienopyridines-CONTRAINDICATED-ABSOLUTELY; aspirin-intact
};

const GENE_INFO = {
  'ITGA2B': { full: 'Glanzmann-Thrombasthenia-Type-I / ALL-agonist-aggregation-ABSENT / eptifibatide-tirofiban-CI-ABSOLUTELY / rFVIIa-FIRST-LINE / HLA-matched-platelets-FIRST-transfusion / HSCT-curative-refractory', locus: '17q21.31', size: '1039 aa / 120 kDa', inh: 'AR LOF biallelic' },
  'ITGB3':  { full: 'Glanzmann-Thrombasthenia-Type-II-NAITP / HPA-1a-alloantigen / NAITP-neonatal-ICH-20-25pct / eptifibatide-CI-ABSOLUTELY / maternal-IVIG-antenatal / washed-maternal-platelets-neonatal', locus: '17q21.32', size: '788 aa / 87 kDa', inh: 'AR LOF biallelic' },
  'GP1BA':  { full: 'Bernard-Soulier-TypeA1 / giant-platelets-10-15fL-PATHOGNOMONIC / vWF-RCo-ABSENT / DDAVP-INEFFECTIVE-no-GPIba / rFVIIa-FIRST-LINE / RIPA-ABSENT-distinguishes-from-vWD2B-OPPOSITE', locus: '17p13.2', size: '626 aa / 73 kDa', inh: 'AR LOF biallelic' },
  'GP1BB':  { full: 'Bernard-Soulier-22q-DiGeorge / TBX1-co-deletion / cardiac-75pct-PATHOGNOMONIC / MLPA-MANDATORY-standard-seq-MISSES / T-cell-lymphopenia / live-vaccines-CI-until-T-cell-verified', locus: '22q11.21', size: '206 aa / 22 kDa', inh: 'AR LOF / del22q11.2' },
  'NBEAL2': { full: 'Gray-Platelet-Syndrome / alpha-granule-ABSENT-PATHOGNOMONIC / grey-platelets-smear-PATHOGNOMONIC / PF4-lactoferrin-ABSENT / myelofibrosis-35pct-progressive / splenectomy-AVOID-accelerates-fibrosis / ruxolitinib', locus: '3p21.31', size: '2754 aa / 313 kDa', inh: 'AR LOF biallelic' },
  'MYH9':   { full: 'MYH9-RD-May-Hegglin-Sebastian-Fechtner-Epstein / giant-platelets-Dohle-bodies-PATHOGNOMONIC / BLEEDING-MILD / ACEi-ARB-MANDATORY-nephritis-15pct / DDAVP-EFFECTIVE-unlike-BSS / audiometry-annually-from-age-10', locus: '22q12.3', size: '1960 aa / 224 kDa', inh: 'AD LOF heterozygous' },
  'HPS1':   { full: 'Hermansky-Pudlak-Syndrome-Type1 / OCA-nystagmus-PATHOGNOMONIC / dense-granule-absent / pulmonary-fibrosis-DOMINANT-CAUSE-DEATH / pirfenidone-MANDATORY-HPS-IPF / PFTs-annually-from-age-20 / Puerto-Rican-founder-1in1800', locus: '10q24.2', size: '700 aa / 79 kDa', inh: 'AR LOF biallelic' },
  'P2RY12': { full: 'P2Y12-Deficiency / clopidogrel-prasugrel-ticagrelor-CONTRAINDICATED-ABSOLUTELY / ADP-aggregation-ABSENT / aspirin-COX1-INTACT / ACS-use-abciximab-instead / document-prominently-medical-record', locus: '3q25.1', size: '342 aa / 40 kDa', inh: 'AR LOF biallelic' },
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
  const agg = cohort.aggregation_pct || {};
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
        <Badge text={`Bleeding: ${geneAtlas.bleeding_risk?.split(';')[0] || ''}`} color={bg} />
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
      {Object.keys(agg).length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#555', marginBottom: 4 }}>Aggregation response (% responders):</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {Object.entries(agg).map(([ag, pct]) => (
              <div key={ag} style={{ textAlign: 'center', minWidth: 60 }}>
                <div style={{
                  height: 36, width: 54, margin: '0 auto',
                  background: pct === 0 ? '#ef5350' : pct < 50 ? '#ff9800' : '#4caf50',
                  borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: '#fff', fontWeight: 700, fontSize: 13
                }}>{pct === 0 ? '✗' : `${pct}%`}</div>
                <div style={{ fontSize: 10, marginTop: 2 }}>{ag.replace('_', '\u00a0')}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, marginTop: 10 }}>
        {[
          ['N', cohort.n],
          ['Avg age', cohort.mean_age + 'yr'],
          ['Avg PLT', cohort.mean_plt + 'K'],
          ['Avg MPV', cohort.mean_mpv + 'fL'],
          ['Severe bleed', `${cohort.bleeding_severe_count}/40`],
          ['rFVIIa used', `${cohort.rfviia_use_count}/40`],
          ['DDAVP eff.', `${cohort.ddavp_effective_count}/40`],
        ].map(([label, val]) => (
          <div key={label} style={{ textAlign: 'center', background: '#f5f5f5', borderRadius: 4, padding: '4px 2px' }}>
            <div style={{ fontSize: 15, fontWeight: 700 }}>{val}</div>
            <div style={{ fontSize: 10, color: '#666' }}>{label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HerPlateletFunctionPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefinitions(df);
    }).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40 }}>Loading Hereditary Platelet Function Disorder Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const summary = overview?.clinical_summary || {};
  const perGene = overview?.per_gene || [];

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui,sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        🧬 Hereditary Platelet Function Disorder Predisposition Atlas
      </h1>
      <div style={{ fontSize: 13, color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Reference — ITGA2B · ITGB3 · GP1BA · GP1BB · NBEAL2 · MYH9 · HPS1 · P2RY12 &nbsp;|&nbsp;
        320-patient aggregate (8×40) &nbsp;|&nbsp; Seeds {overview?.seeds}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            fontWeight: tab === t ? 700 : 400,
            background: tab === t ? '#1a237e' : '#e8eaf6', color: tab === t ? '#fff' : '#333',
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && (
        <div>
          {/* Summary cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 20 }}>
            {[
              ['Total Patients', overview?.total_patients, '#1a237e'],
              ['Genes', overview?.n_genes, '#4a148c'],
              ['Severe Bleed', summary.severe_bleed_total, '#b71c1c'],
              ['rFVIIa Used', summary.rfviia_use_total, '#004d40'],
              ['Myelofibrosis', summary.myelofibrosis_total, '#880e4f'],
              ['SNHL', summary.snhl_total, '#006064'],
              ['Pulm. Fibrosis', summary.pulmonary_fibrosis_total, '#33691e'],
              ['Alloimmunised', summary.alloimmunisation_total, '#bf360c'],
            ].map(([label, val, color]) => (
              <div key={label} style={{ background: color, color: '#fff', borderRadius: 8, padding: '16px 18px', textAlign: 'center' }}>
                <div style={{ fontSize: 26, fontWeight: 800 }}>{val}</div>
                <div style={{ fontSize: 12 }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Key clinical distinctions */}
          <div style={{ background: '#fff8e1', border: '1px solid #f9a825', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>⚡ Key Clinical Distinctions</h3>
            {(overview?.key_clinical_distinctions || []).map((d, i) => (
              <div key={i} style={{ fontSize: 12, padding: '4px 0', borderBottom: '1px solid #ffe082' }}>{d}</div>
            ))}
          </div>

          {/* Per-gene summary bar */}
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Per-Gene Summary</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#e8eaf6' }}>
                  {['Gene', 'Syndrome', 'Avg PLT', 'MPV (fL)', 'Severe Bleed %', 'rFVIIa %', 'DDAVP eff. %', 'Key Rule'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {perGene.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f5f5f5' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
                    <td style={{ padding: '7px 10px' }}>{g.bleeding_risk?.split(';')[0]}</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_plt}K</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_mpv}</td>
                    <td style={{ padding: '7px 10px' }}>{g.severe_bleed_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{g.rfviia_pct}%</td>
                    <td style={{ padding: '7px 10px' }}>{g.ddavp_effective_pct}%</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.key_rule?.split(';')[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {Object.entries(GENE_COLORS).map(([gene, color]) => (
              <GeneCard key={gene} gene={gene} color={color} info={GENE_INFO[gene]} breakdown={breakdown} />
            ))}
          </div>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && (
        <div>
          {(breakdown?.atlas_genes || []).map(g => (
            <div key={g.gene} style={{
              border: `2px solid ${GENE_COLORS[g.gene]}`, borderRadius: 8, padding: 16, marginBottom: 16
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                <div style={{ background: GENE_COLORS[g.gene], color: '#fff', borderRadius: 6, padding: '6px 16px', fontWeight: 700, fontSize: 20 }}>{g.gene}</div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 15 }}>{g.syndrome}</div>
                  <div style={{ fontSize: 12, color: '#666' }}>{g.locus} · {GENE_INFO[g.gene]?.size} · {g.inheritance}</div>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div style={{ background: '#fff3e0', borderRadius: 6, padding: '8px 12px' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#e65100', marginBottom: 4 }}>⚠ PATHOGNOMONIC</div>
                  <div style={{ fontSize: 12 }}>{g.pathognomonic}</div>
                </div>
                <div style={{ background: '#e8f5e9', borderRadius: 6, padding: '8px 12px' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#2e7d32', marginBottom: 4 }}>✓ KEY RULE</div>
                  <div style={{ fontSize: 12 }}>{g.key_rule}</div>
                </div>
                <div style={{ background: '#ffebee', borderRadius: 6, padding: '8px 12px' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#c62828', marginBottom: 4 }}>✗ AVOID</div>
                  <div style={{ fontSize: 12 }}>{g.key_avoid}</div>
                </div>
                <div style={{ background: '#e3f2fd', borderRadius: 6, padding: '8px 12px' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#1565c0', marginBottom: 4 }}>💊 TARGETED Rx</div>
                  <div style={{ fontSize: 12 }}>{g.targeted_rx}</div>
                </div>
              </div>
              <div style={{ marginTop: 8, background: '#f5f5f5', borderRadius: 6, padding: '8px 12px' }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#555', marginBottom: 4 }}>📋 SURVEILLANCE</div>
                <div style={{ fontSize: 12 }}>{g.surveillance}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && (
        <div>
          {(definitions?.definitions || []).map((d, i) => (
            <div key={i} style={{ borderBottom: '1px solid #e0e0e0', padding: '10px 0' }}>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{d.term}</div>
              <div style={{ fontSize: 12, color: '#444', lineHeight: 1.5 }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
