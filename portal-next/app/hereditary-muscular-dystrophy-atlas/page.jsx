'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-muscular-dystrophy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  DMD:    '#b71c1c',  // deep red      — Duchenne/Becker, most common XLR MD, Elevidys FDA2023
  DMPK:   '#c62828',  // red           — DM1 Steinert, CTG repeat, multisystem, anaesthesia risk
  SMCHD1: '#1565c0',  // deep blue     — FSHD2, D4Z4 epigenetic, asymmetric weakness PATHOGNOMONIC
  EMD:    '#0d47a1',  // navy          — EDMD1 emerin, contractures before weakness, ICD mandatory
  LMNA:   '#6a1b9a',  // deep purple   — EDMD2 + DCM, most lethal MD, ICD regardless of LVEF
  CAPN3:  '#1b5e20',  // dark green    — LGMD-R1, most common AR-LGMD 30%, no cardiac
  DYSF:   '#e65100',  // burnt orange  — LGMD-R2 Miyoshi, CK 50-100x, steroids ABSOLUTE CI
  GNE:    '#004d40',  // dark teal     — GNE myopathy IBM2, quadriceps spared PATHOGNOMONIC
};

const GENE_INFO = {
  DMD:    { full: 'DMD/Dystrophin / 3685aa', locus: 'Xp21.2', inh: 'XLR', disease: 'Duchenne/Becker MD — CK 10,000-100,000 IU/L HIGHEST all MD; calf pseudohypertrophy PATHOGNOMONIC; ELEVIDYS (SRP-9001) FDA 2023 gene therapy age 4-17; exon-skipping genotype-specific (eteplirsen/golodirsen/casimersen); steroids mandatory; ACEi from age 10; SUCCINYLCHOLINE ABSOLUTE CI' },
  DMPK:   { full: 'DMPK/Myotonin-kinase / 629aa', locus: '19q13.32', inh: 'AD (CTG repeat)', disease: 'DM1 Steinert — ANAESTHESIA EXTREME RISK: succinylcholine ABSOLUTE CI, volatile agents worsen; CTG ANTICIPATION maternal transmission → congenital DM1; myotonia → mexiletine; cardiac pacemaker/ICD (AV block 30% sudden death); annual ECG + Holter' },
  SMCHD1: { full: 'SMCHD1 / 2005aa', locus: '18p11.32', inh: 'AD (haploinsufficiency)', disease: 'FSHD2 — ASYMMETRY PATHOGNOMONIC (one side much weaker); DIGENIC: requires permissive 4qA haplotype + D4Z4 methylation <18%; D4Z4 methylation assay MANDATORY to confirm; Losmapimod phase 3 trial; Beevor sign; scapular fixation surgery if winging severe' },
  EMD:    { full: 'EMD/Emerin / 254aa', locus: 'Xq28', inh: 'XLR', disease: 'EDMD1 — CONTRACTURES BEFORE WEAKNESS PATHOGNOMONIC (temporal sequence); ICD MANDATORY (pacemaker alone insufficient — SCD risk); emerin IHC on blood cells diagnostic; female carriers 30% cardiac (surveillance mandatory); humeral-peroneal weakness distribution' },
  LMNA:   { full: 'LMNA/Lamin A/C / 664aa', locus: '1q22', inh: 'AD', disease: 'EDMD2 + LMNA-DCM — MOST LETHAL MD; ICD REGARDLESS of LVEF (Padua score ≥4); non-missense = highest cardiac risk (+2 Padua); FLECAINIDE/PROPAFENONE ABSOLUTELY CI (mortality increase); mean cardiac event age 36y; cardiac MRI midwall LGE diagnostic; LVAD bridge to transplant' },
  CAPN3:  { full: 'CAPN3/Calpain-3 / 821aa', locus: '15q15.1', inh: 'AR (biallelic)', disease: 'LGMD-R1 — MOST COMMON LGMD ~30% worldwide; pelvifemoral pattern; NO CARDIAC (key DDx vs dystrophinopathy, laminopathy); NORMAL CK possible (unique among LGMD — do NOT exclude); Basque founder del550-572; western blot + enzyme activity for confirmation; AAV gene therapy trials' },
  DYSF:   { full: 'DYSF/Dysferlin / 2080aa', locus: '2p13.2', inh: 'AR (biallelic)', disease: 'LGMD-R2/Miyoshi — CK 50-100x ULN PATHOGNOMONIC (highest of all LGMD); STEROIDS ABSOLUTE CI (accelerate disease); inflammatory biopsy mimics polymyositis — DO NOT immunosuppress; dysferlin flow cytometry on monocytes diagnostic; no cardiac (key DDx)' },
  GNE:    { full: 'GNE/GNE-Myopathy-IBM2 / 722aa', locus: '9p13.3', inh: 'AR (biallelic)', disease: 'GNE Myopathy/IBM2 — QUADRICEPS SPARED despite severe foot drop PATHOGNOMONIC; rimmed vacuoles biopsy; CK normal/mildly elevated (1-10x); Middle Eastern Jewish founder M712T/V727M; GRACE trial failed (no approved therapy); distinguish from sporadic IBM (sIBM age >50)' },
};

function GeneChip({ gene }) {
  return (
    <span style={{
      background: GENE_COLORS[gene] || '#555',
      color: '#fff',
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 12,
      fontWeight: 700,
      marginRight: 4,
      display: 'inline-block',
    }}>{gene}</span>
  );
}

function MetricCard({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 140, flex: '1 1 140px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryMuscularDystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' || tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true);
    setError(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (ep === 'overview') setOverview(d);
        else if (ep === 'breakdown') setBreakdown(d);
        else setDefinitions(d);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1e293b,#0f172a)', padding: '28px 32px 20px', borderBottom: '1px solid #1e3a5f' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>
          Hereditary Muscular Dystrophy Atlas
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          🧬 Hereditary-Muscular-Dystrophy-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene Hereditary Muscular Dystrophy Reference — DMD · DMPK (DM1) · SMCHD1 (FSHD2) · EMD (EDMD1) · LMNA (EDMD2) · CAPN3 (LGMD-R1) · DYSF (LGMD-R2) · GNE (IBM2)
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
          {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #1e293b', background: '#0f172a', paddingLeft: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: 'none', border: 'none', color: tab === t ? '#38bdf8' : '#64748b',
            borderBottom: tab === t ? '2px solid #38bdf8' : '2px solid transparent',
            padding: '12px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>
        {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
        {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
              {overview.subtitle} · {overview.total_patients} patients · seeds {overview.seed_range}
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
              <MetricCard label="Mean Dx Delay" value={`${overview.aggregate_stats?.mean_dx_delay_months}mo`} sub="across all MDs" />
              <MetricCard label="Genes Covered" value={overview.aggregate_stats?.genes_covered || 8} sub="hereditary MD loci" />
              <MetricCard label="Patients/Gene" value={overview.aggregate_stats?.patients_per_gene || 40} sub="standardised cohort" />
            </div>

            {/* Top Alerts */}
            {overview.top_alerts && overview.top_alerts.length > 0 && (
              <div style={{ background: '#1e293b', borderRadius: 10, padding: 16, marginBottom: 20, borderLeft: '4px solid #f87171' }}>
                <div style={{ color: '#f87171', fontWeight: 700, fontSize: 12, marginBottom: 10 }}>🚨 TOP CLINICAL ALERTS</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {overview.top_alerts.map((alert, i) => (
                    <div key={i} style={{ color: '#e2e8f0', fontSize: 12, paddingLeft: 8, borderLeft: '2px solid #334155' }}>
                      {alert.substring(0, 250)}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 14 }}>
              {(overview.genes || []).map(gs => (
                <div key={gs.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gs.gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <GeneChip gene={gs.gene} />
                    <span style={{ color: '#94a3b8', fontSize: 11 }}>{gs.locus} · {gs.aa}aa / {gs.kDa}kDa · {gs.inheritance?.split('—')[0]?.trim()}</span>
                  </div>
                  <div style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{gs.protein?.substring(0, 80)}</div>
                  <div style={{ color: '#94a3b8', fontSize: 11 }}>{gs.gene_class?.substring(0, 180)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
                  {['Gene', 'Locus', 'Size (aa)', 'OMIM', 'Inheritance', 'Mean Dx Delay', 'Patients', 'Key Class'].map(h => (
                    <th key={h} style={{ padding: '10px 10px', textAlign: 'left', borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.map(g => (
                  <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '9px 10px' }}><GeneChip gene={g.gene} /></td>
                    <td style={{ padding: '9px 10px', color: '#94a3b8' }}>{g.locus}</td>
                    <td style={{ padding: '9px 10px', color: '#94a3b8', textAlign: 'center' }}>{g.aa}</td>
                    <td style={{ padding: '9px 10px', color: '#64748b', fontSize: 11 }}>{g.omim_gene}</td>
                    <td style={{ padding: '9px 10px', color: '#94a3b8', maxWidth: 140 }}>{g.inheritance?.split('—')[0]?.trim()?.substring(0, 30)}</td>
                    <td style={{ padding: '9px 10px', color: '#fbbf24', textAlign: 'center' }}>
                      {g.computed?.mean_dx_delay_months != null
                        ? `${g.computed.mean_dx_delay_months}mo`
                        : `${(g.dx_delay_distribution || []).reduce((s, e) => s + e.months, 0) / Math.max((g.dx_delay_distribution || []).length, 1) | 0}mo`}
                    </td>
                    <td style={{ padding: '9px 10px', color: '#a78bfa', textAlign: 'center' }}>{g.stats?.n_patients || 40}</td>
                    <td style={{ padding: '9px 10px', color: '#94a3b8', maxWidth: 200 }}>{g.gene_class?.substring(0, 60)}…</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'grid', gap: 20 }}>
            {breakdown.map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 12, padding: 20, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 15 }}>{g.protein?.substring(0, 70)}</span>
                  <span style={{ color: '#64748b', fontSize: 11 }}>{g.locus} · {g.aa}aa · {g.inheritance?.split('—')[0]?.trim()?.substring(0, 25)}</span>
                </div>

                {/* Disease info from GENE_INFO */}
                <div style={{ color: '#cbd5e1', fontSize: 12, marginBottom: 12, background: '#0f172a', padding: '8px 12px', borderRadius: 6 }}>
                  {GENE_INFO[g.gene]?.disease}
                </div>

                {/* Key Alerts */}
                {g.key_alerts && g.key_alerts.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ color: '#f87171', fontSize: 11, fontWeight: 600, marginBottom: 6 }}>🚨 KEY ALERTS ({g.key_alerts.length})</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {g.key_alerts.slice(0, 4).map((alert, i) => (
                        <div key={i} style={{ color: '#94a3b8', fontSize: 11, paddingLeft: 8, borderLeft: '2px solid #334155' }}>
                          {alert.substring(0, 200)}
                        </div>
                      ))}
                      {g.key_alerts.length > 4 && (
                        <div style={{ color: '#64748b', fontSize: 10 }}>+{g.key_alerts.length - 4} more alerts…</div>
                      )}
                    </div>
                  </div>
                )}

                {/* Stats */}
                {g.stats && Object.keys(g.stats).length > 0 && (
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11, color: '#64748b' }}>
                    {Object.entries(g.stats).slice(0, 5).map(([k, v]) => (
                      <span key={k}>{k.replace(/_/g, ' ')}: <b style={{ color: '#94a3b8' }}>{String(v).substring(0, 25)}</b></span>
                    ))}
                  </div>
                )}

                {/* Sample patients */}
                {g.sample_patients && g.sample_patients.length > 0 && (
                  <div style={{ marginTop: 12, borderTop: '1px solid #334155', paddingTop: 10 }}>
                    <div style={{ color: '#64748b', fontSize: 10, marginBottom: 6 }}>SAMPLE PATIENTS (n={g.stats?.n_patients || 40})</div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {g.sample_patients.slice(0, 5).map(p => (
                        <div key={p.patient_id} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px', fontSize: 10, color: '#94a3b8' }}>
                          <b>{p.patient_id}</b> · dx {p.age_at_dx}y · CK {p.ck_x_uln}× · {p.variant_class?.substring(0, 20)}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {/* Concepts */}
            {definitions.concepts && (
              <div style={{ marginBottom: 28 }}>
                <div style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 16, marginBottom: 14 }}>Classification & Concepts</div>
                <div style={{ display: 'grid', gap: 12 }}>
                  {Object.entries(definitions.concepts).map(([term, def]) => (
                    <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
                      <div style={{ color: '#38bdf8', fontWeight: 700, marginBottom: 6 }}>{term}</div>
                      <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.6 }}>{String(def).substring(0, 500)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Pharmacological distinctions */}
            {definitions.pharmacological_distinctions && definitions.pharmacological_distinctions.length > 0 && (
              <div style={{ marginBottom: 28 }}>
                <div style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 16, marginBottom: 14 }}>Pharmacological Distinctions</div>
                <div style={{ display: 'grid', gap: 10 }}>
                  {definitions.pharmacological_distinctions.map((item, i) => (
                    <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 14, borderLeft: '3px solid #34d399' }}>
                      <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{item.substring(0, 400)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Key standards */}
            {definitions.key_standards && definitions.key_standards.length > 0 && (
              <div>
                <div style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 16, marginBottom: 14 }}>Key Standards & References</div>
                <div style={{ display: 'grid', gap: 10 }}>
                  {definitions.key_standards.map((item, i) => (
                    <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 14, borderLeft: '3px solid #a78bfa' }}>
                      <div style={{ color: '#94a3b8', fontSize: 12 }}>{item.substring(0, 300)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
