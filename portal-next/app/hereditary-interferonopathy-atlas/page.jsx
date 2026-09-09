'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-interferonopathy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TREX1:    '#b71c1c',  // deep red     — most common AGS, cGAS-STING, FCL
  RNASEH2B: '#1565c0',  // deep blue    — mildest AGS, European founder
  RNASEH2A: '#0d47a1',  // navy         — catalytic subunit, severe AGS4
  RNASEH2C: '#283593',  // dark indigo  — structural subunit, AGS2, historically first
  SAMHD1:   '#4a148c',  // dark purple  — AGS5, cerebrovascular stroke distinctive
  ADAR1:    '#2e7d32',  // deep green   — AGS6, striatal necrosis + DSH skin
  IFIH1:    '#e65100',  // burnt orange — AGS7/MDA5 GOF, milder AGS, baricitinib evidence
  STING1:   '#004d40',  // dark teal    — SAVI, vasculitis + ILD, JAK inhibitors highly effective
};

const GENE_INFO = {
  TREX1:    { full: 'TREX1 / 314aa',    locus: '3p21.31',  size: '314 aa / 36 kDa',  inh: 'AD/AR',    disease: 'AGS1 / FCL / SLE — MOST COMMON AGS GENE (~25% of all AGS); AR biallelic: severe Aicardi-Goutières syndrome (encephalopathy + calcifications); AD heterozygous: FCL (Familial Chilblain Lupus — cold-triggered acral ulcers) + SLE risk; TREX1 degrades cytosolic ssDNA → cGAS-STING activation when absent; ISG score elevated; JAK inhibitors + hydroxychloroquine (FCL)' },
  RNASEH2B: { full: 'RNASEH2B / 312aa', locus: '13q14.3',  size: '312 aa / 36 kDa',  inh: 'AR',       disease: 'AGS3 — MILDEST AGS PHENOTYPE + MOST COMMON AR-AGS IN EUROPE (~35%); p.Ala177Thr Northern European founder (hypomorphic → residual RNase H2 activity → milder phenotype); SPASTIC PARAPLEGIA + PRESERVED LANGUAGE + COGNITION = KEY DDx from other AGS genes; non-catalytic scaffold subunit connecting PCNA to RNase H2 complex' },
  RNASEH2A: { full: 'RNASEH2A / 299aa', locus: '19p13.13', size: '299 aa / 33 kDa',  inh: 'AR',       disease: 'AGS4 — CATALYTIC SUBUNIT (all RNase H2 enzymatic activity); biallelic null = complete RNase H2 loss → highest ISG score among RNase H2 subunits; severe AGS with microcephaly + cerebral atrophy + calcifications; cerebrovascular complications (rare); early progressive encephalopathy; palliative care early integration' },
  RNASEH2C: { full: 'RNASEH2C / 164aa', locus: '11q13.1',  size: '164 aa / 18 kDa',  inh: 'AR',       disease: 'AGS2 — SMALLEST RNase H2 subunit (structural bridge); HISTORICALLY FIRST RNase H2 gene described in AGS (Crow 2006); complex destabilisation → complete enzymatic loss → severe AGS similar to RNASEH2A; early progressive encephalopathy + calcifications; test RNASEH2A/B/C together on RNase H2 panel' },
  SAMHD1:   { full: 'SAMHD1 / 626aa',   locus: '20q11.23', size: '626 aa / 72 kDa',  inh: 'AR/AD-FCL', disease: 'AGS5 / FCL2 — CEREBROVASCULAR DISEASE + ISCHAEMIC STROKE IN CHILDHOOD PATHOGNOMONIC (unique among AGS subtypes); dNTPase = HIV-1 restriction factor (Vpx degrades SAMHD1); FCL2 (AD heterozygous) = cold-triggered acral ulcers like TREX1-FCL; calcifications + elevated ISG; aspirin for stroke prevention; MRA annually for cerebrovascular risk' },
  ADAR1:    { full: 'ADAR1 / 1226aa',   locus: '1q21.3',   size: '1226 aa / 139 kDa', inh: 'AD/AR',   disease: 'AGS6 / DSH — BILATERAL STRIATAL NECROSIS (bilateral caudate + putamen DWI restriction on MRI) PATHOGNOMONIC for ADAR1-AGS (unique among interferonopathies); DSH (Dyschromatosis Symmetrica Hereditaria): mixed hypo/hyperpigmented macules on extremities = AD p.Gly1007Arg; A-to-I RNA editing enzyme — deficiency → unedited dsRNA → MDA5/IFIH1 activation → IFN; biallelic null = severe, p.Gly1007Arg AD = DSH ± mild AGS' },
  IFIH1:    { full: 'IFIH1/MDA5 / 1025aa', locus: '2q24.2', size: '1025 aa / 117 kDa', inh: 'AD GOF', disease: 'AGS7 — MDA5 CYTOSOLIC RNA HELICASE GOF; MILDER AGS PHENOTYPE (often preserved cognition); SINGLETON DE NOVO dominant mutations common; BEST JAK INHIBITOR RESPONSE (baricitinib clinical trial evidence in IFIH1-AGS7); PARADOX: IFIH1 LOF variants PROTECT against Type 1 Diabetes (reduced enteroviral MDA5 sensing → less beta-cell destruction); MDA5-MAVS-TBK1-IRF3 cascade constitutively activated' },
  STING1:   { full: 'STING1/TMEM173 / 379aa', locus: '5q31.2', size: '379 aa / 42 kDa', inh: 'AD GOF', disease: 'SAVI — CUTANEOUS VASCULITIS (necrotic ulcers nose/ears/digits) PATHOGNOMONIC; INTERSTITIAL LUNG DISEASE (ILD) = main mortality cause; DISTINCT from AGS (no calcifications, no encephalopathy); cGAS-STING pathway GOF (constitutive STING1 without cGAMP); HIGHEST ISG scores; JAK INHIBITORS (ruxolitinib/baricitinib) MOST EFFECTIVE — wound healing of necrotic ulcers + ILD stabilisation; onset in infancy; early treatment prevents digit loss' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px', fontSize: 11, fontWeight: 700, marginRight: 4,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: '#fff', border: `2px solid ${color || '#e0e0e0'}`,
      borderRadius: 10, padding: '14px 18px', minWidth: 120, textAlign: 'center',
    }}>
      <div style={{ fontSize: 26, fontWeight: 800, color: color || '#333' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#888' }}>{sub}</div>}
    </div>
  );
}

export default function HeredIFNAtlasPage() {
  const [tab, setTab] = useState(0);
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
    }).catch(e => setError(String(e))).finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#b71c1c', fontWeight: 700 }}>Loading Hereditary Type I Interferonopathy Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui, sans-serif', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#b71c1c', marginBottom: 4 }}>
          🧬 Hereditary Type I Interferonopathy Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Atlas — TREX1 · RNASEH2B · RNASEH2A · RNASEH2C · SAMHD1 · ADAR1 · IFIH1 · STING1 —
          320 patients (8 × 40, seeds 2318-2325) | AGS1-7 + SAVI
        </div>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 22 }}>
        <StatCard label="Total Patients" value={overview.n_patients} color="#b71c1c" />
        <StatCard label="Genes" value={overview.n_genes} color="#2e7d32" />
        <StatCard label="Seed Range" value="2318-2325" color="#e65100" />
        <StatCard label="AGS Genes" value={overview.ags_genes?.length ?? 7} sub="TREX1 · RNASEH2A/B/C · SAMHD1 · ADAR1 · IFIH1" color="#1565c0" />
        <StatCard label="SAVI" value={overview.savi_genes?.length ?? 1} sub="STING1 (GOF)" color="#004d40" />
        <StatCard label="Skin Phenotype" value={overview.skin_phenotype_genes?.length ?? 3} sub="TREX1 · SAMHD1 · ADAR1" color="#4a148c" />
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', border: 'none', cursor: 'pointer', fontWeight: 700, fontSize: 13,
            background: tab === i ? '#b71c1c' : '#f5f5f5',
            color: tab === i ? '#fff' : '#555',
            borderRadius: '6px 6px 0 0',
          }}>{t}</button>
        ))}
      </div>

      {/* Tab 0: Overview */}
      {tab === 0 && (
        <div>
          {/* Category groupings */}
          <div style={{ marginBottom: 20 }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: '#b71c1c', marginBottom: 10 }}>Disease Categories</h3>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {overview.interferonopathy_categories && Object.entries(overview.interferonopathy_categories).map(([cat, gns]) => (
                <div key={cat} style={{ background: '#ffebee', borderRadius: 8, padding: '10px 16px', minWidth: 220 }}>
                  <div style={{ fontWeight: 700, color: '#b71c1c', fontSize: 13, marginBottom: 6 }}>{cat}</div>
                  {gns.map(g => (
                    <div key={g} style={{ fontSize: 12, color: '#333', marginBottom: 2 }}>
                      <span style={{ background: GENE_COLORS[g] + '22', color: GENE_COLORS[g], borderRadius: 3, padding: '1px 5px', fontWeight: 700, marginRight: 4 }}>{g}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Key clinical pearls */}
          <div style={{ background: '#fff3e0', borderRadius: 10, padding: '16px 20px', marginBottom: 20 }}>
            <h3 style={{ fontSize: 15, fontWeight: 700, color: '#e65100', marginBottom: 10 }}>🔑 Key Clinical Pearls</h3>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(overview.key_clinical_pearls || []).map((p, i) => (
                <li key={i} style={{ fontSize: 13, color: '#333', marginBottom: 6 }}>{p}</li>
              ))}
            </ul>
          </div>

          {/* Emergency flags */}
          {overview.clinical_emergency_flags && (
            <div style={{ background: '#ffebee', borderRadius: 10, padding: '16px 20px', marginBottom: 20 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#b71c1c', marginBottom: 10 }}>🚨 Clinical Emergency Flags</h3>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {overview.clinical_emergency_flags.map((f, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#333', marginBottom: 5 }}>{f}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Diagnostic algorithm */}
          {overview.diagnostic_algorithm && (
            <div style={{ background: '#e8f5e9', borderRadius: 10, padding: '16px 20px' }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#2e7d32', marginBottom: 10 }}>🔬 Diagnostic Algorithm</h3>
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                {overview.diagnostic_algorithm.map((s, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>{s}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}

      {/* Tab 1: Gene Table */}
      {tab === 1 && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#b71c1c', color: '#fff' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Locus / Size</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Inheritance</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Disease / Key Feature</th>
              </tr>
            </thead>
            <tbody>
              {genes.map((g, i) => {
                const info = GENE_INFO[g];
                return (
                  <tr key={g} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff', borderBottom: '1px solid #e0e0e0' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 800, color: GENE_COLORS[g], fontSize: 14 }}>{g}</td>
                    <td style={{ padding: '10px 12px', color: '#555' }}>
                      <div style={{ fontWeight: 600 }}>{info.locus}</div>
                      <div style={{ fontSize: 11, color: '#888' }}>{info.size}</div>
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={info.inh} color={GENE_COLORS[g]} />
                    </td>
                    <td style={{ padding: '10px 12px', color: '#333', lineHeight: 1.5 }}>{info.disease}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Tab 2: Clinical Atlas */}
      {tab === 2 && breakdown && (
        <div>
          {genes.map(g => {
            const gb = breakdown.gene_breakdown?.[g];
            if (!gb) return null;
            return (
              <div key={g} style={{
                border: `2px solid ${GENE_COLORS[g]}44`,
                borderRadius: 10, marginBottom: 18, padding: '14px 18px',
                borderLeft: `5px solid ${GENE_COLORS[g]}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ fontSize: 18, fontWeight: 800, color: GENE_COLORS[g] }}>{g}</span>
                  <Badge text={GENE_INFO[g].inh} color={GENE_COLORS[g]} />
                  <Badge text={GENE_INFO[g].locus} color="#555" />
                  <span style={{ fontSize: 12, color: '#888' }}>{gb.n_patients} pts</span>
                  {gb.lung_disease && <Badge text="ILD Risk" color="#004d40" />}
                  {gb.cerebrovascular_risk && <Badge text="CVD/Stroke" color="#4a148c" />}
                  {gb.skin_phenotype && <Badge text="Skin" color="#e65100" />}
                </div>
                <div style={{ fontSize: 12, color: '#444', marginBottom: 10, lineHeight: 1.5 }}>
                  {gb.interferonopathy_category}
                </div>
                <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 12 }}>
                  {gb.complication_distribution && (
                    <div>
                      <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>Complications</div>
                      {Object.entries(gb.complication_distribution).slice(0, 4).map(([k, v]) => (
                        <div key={k} style={{ color: '#444' }}>{k}: <b>{v}</b></div>
                      ))}
                    </div>
                  )}
                  {gb.isg_distribution && (
                    <div>
                      <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>ISG Level</div>
                      {Object.entries(gb.isg_distribution).map(([k, v]) => (
                        <div key={k} style={{ color: '#444' }}>{k}: <b>{v}</b></div>
                      ))}
                      <div style={{ color: '#888', marginTop: 2 }}>Avg ISG: <b>{gb.avg_isg_score_sd} SD</b></div>
                    </div>
                  )}
                  {gb.severity_distribution && (
                    <div>
                      <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>Severity</div>
                      {Object.entries(gb.severity_distribution).map(([k, v]) => (
                        <div key={k} style={{ color: '#444' }}>{k}: <b>{v}</b></div>
                      ))}
                    </div>
                  )}
                  {gb.treatment_distribution && (
                    <div>
                      <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>Treatments</div>
                      {Object.entries(gb.treatment_distribution).slice(0, 4).map(([k, v]) => (
                        <div key={k} style={{ color: '#444' }}>{k}: <b>{v}</b></div>
                      ))}
                    </div>
                  )}
                  <div>
                    <div style={{ fontWeight: 700, color: '#555', marginBottom: 4 }}>Key Stats</div>
                    <div style={{ color: '#444' }}>Seizures: <b>{gb.pct_seizures}%</b></div>
                    {gb.cerebrovascular_risk && <div style={{ color: '#b71c1c' }}>CVD events: <b>{gb.pct_cerebrovascular}%</b></div>}
                    {gb.skin_phenotype && <div style={{ color: '#e65100' }}>Skin phenotype: <b>{gb.pct_skin_phenotype}%</b></div>}
                    {gb.lung_disease && <div style={{ color: '#004d40' }}>ILD present: <b>{gb.pct_lung_disease}%</b></div>}
                    <div style={{ color: '#888' }}>Dx delay: <b>{gb.avg_diagnosis_delay_months} mo</b></div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && definitions && (
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 700, color: '#b71c1c', marginBottom: 12 }}>Gene Entries</h3>
          {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
            <div key={gene} style={{
              border: `1px solid ${GENE_COLORS[gene] || '#ccc'}44`,
              borderRadius: 8, padding: '12px 16px', marginBottom: 14,
              borderLeft: `4px solid ${GENE_COLORS[gene] || '#ccc'}`,
            }}>
              <div style={{ fontWeight: 800, color: GENE_COLORS[gene] || '#333', fontSize: 15, marginBottom: 4 }}>{gene}</div>
              <div style={{ fontSize: 12, color: '#666', marginBottom: 6 }}>
                {entry.protein_size} · {entry.locus} · {entry.inheritance}
              </div>
              <div style={{ fontSize: 12, color: '#333', marginBottom: 4 }}><b>Disease:</b> {entry.disease_name}</div>
              <div style={{ fontSize: 12, color: '#333', marginBottom: 4 }}><b>IFN Pathway:</b> {entry.ifn_pathway}</div>
              <div style={{ fontSize: 12, color: '#333', marginBottom: 4 }}><b>Pathognomonic:</b> {entry.pathognomonic?.slice(0, 300)}</div>
              {entry.key_features && (
                <ul style={{ margin: '4px 0', paddingLeft: 18, fontSize: 12, color: '#444' }}>
                  {entry.key_features.slice(0, 4).map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              )}
            </div>
          ))}

          {definitions.interferonopathy_glossary && (
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#2e7d32', marginBottom: 10, marginTop: 20 }}>Interferonopathy Glossary</h3>
              {Object.entries(definitions.interferonopathy_glossary).map(([term, def]) => (
                <div key={term} style={{ background: '#e8f5e9', borderRadius: 6, padding: '10px 14px', marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, color: '#2e7d32', fontSize: 13, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{def}</div>
                </div>
              ))}
            </div>
          )}

          {definitions.treatment_glossary && (
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#e65100', marginBottom: 10, marginTop: 20 }}>Treatment Glossary</h3>
              {Object.entries(definitions.treatment_glossary).map(([term, def]) => (
                <div key={term} style={{ background: '#fff3e0', borderRadius: 6, padding: '10px 14px', marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, color: '#e65100', fontSize: 13, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{def}</div>
                </div>
              ))}
            </div>
          )}

          {definitions.diagnostic_tests && (
            <div>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#4a148c', marginBottom: 10, marginTop: 20 }}>Diagnostic Tests</h3>
              {Object.entries(definitions.diagnostic_tests).map(([term, def]) => (
                <div key={term} style={{ background: '#f3e5f5', borderRadius: 6, padding: '10px 14px', marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, color: '#4a148c', fontSize: 13, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#333', lineHeight: 1.5 }}>{def}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
