'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-alps-fas-fasl-lymphoproliferation-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'FAS':    '#b71c1c',  // deep red      — FAS/TNFRSF6-ALPS Ia most common >70% all ALPS
  'FASLG':  '#e65100',  // deep orange   — FASLG SLE-like ALPS Ib rare
  'CASP10': '#1565c0',  // deep blue     — CASP10 ALPS IIb DN defective T-cell apoptosis
  'CASP8':  '#4a148c',  // deep purple   — CASP8/CED dual immunodeficiency + lymphoproliferation UNIQUE
  'FADD':   '#2e7d32',  // dark green    — FADD cardiomyopathy+encephalopathy DISTINCTIVE triad
  'PRKCD':  '#f9a825',  // amber         — PKCδ B-cell tolerance SLE-like lupus nephritis
  'NRAS':   '#00695c',  // dark teal     — NRAS RALD type1 somatic GOF MEKi cobimetinib
  'KRAS':   '#6a1b9a',  // purple        — KRAS RALD type2 monocytosis JMML-risk MEKi
};

const GENE_INFO = {
  'FAS':    { full: 'FAS / TNFRSF6 / CD95 / APO-1 / 335aa', locus: '10q23.31', size: '335 aa / 36 kDa (death receptor; FAS trimerization → FADD-DD:DD → CASP8-DED:DED → DISC → caspase cascade → T cell AICD; DN heterozygous most common; ALPS type Ia ≥70% all ALPS)', inh: 'AD (DN) / AR' },
  'FASLG':  { full: 'FASLG / FAS Ligand / CD95L / TNFSF6 / 281aa', locus: '1q24.3', size: '281 aa / 31 kDa (type II transmembrane; FasL on CTL kills FAS-expressing targets; soluble sFasL cleaved by ADAM10/17; ALPS type Ib; SLE-like with anti-dsDNA, lupus nephritis, uveitis)', inh: 'AD / AR' },
  'CASP10': { full: 'CASP10 / Caspase-10 / 521aa', locus: '2q33.1', size: '521 aa / 59 kDa (DED-containing initiator caspase; DISC amplification alongside CASP8; DN variants p.Ile406Leu, p.Val410Ile; ALPS type IIb; reduced not absent apoptosis; lower lymphoma risk than FAS)', inh: 'AD (DN)' },
  'CASP8':  { full: 'CASP8 / Caspase-8 / 479aa (CED)', locus: '2q33.1', size: '479 aa / 55 kDa (master initiator caspase extrinsic pathway; NF-κB activation TLR/TCR; AR biallelic = DUAL: CID + ALPS; hypogammaglobulinemia DISTINCTIVE; necroptosis dysregulation; 2q33.1 adjacent CASP10)', inh: 'AR (biallelic)' },
  'FADD':   { full: 'FADD / FAS-Associated Via Death Domain / 208aa', locus: '11q13.3', size: '208 aa / 23 kDa (DD + DED adaptor; FAS-DD→FADD-DD→CASP8-DED; upstream all extrinsic apoptosis; AR biallelic; cardiomyopathy 65% DISTINCTIVE + encephalopathy 55% + ALPS + infections triad)', inh: 'AR (biallelic)' },
  'PRKCD':  { full: 'PRKCD / PKCδ / Protein Kinase C Delta / 683aa', locus: '3p21.3', size: '683 aa / 78 kDa (serine/threonine kinase; caspase-3-activated during apoptosis → amplifies B cell apoptosis; B cell tolerance checkpoint; AR biallelic; SLE-like onset <5yr; ANA >1:640; lupus nephritis 62%; consanguinity)', inh: 'AR (biallelic)' },
  'NRAS':   { full: 'NRAS / N-Ras / 189aa (Somatic GOF — ALPS-RALD1)', locus: '1p13.2', size: '189 aa / 21 kDa (small GTPase; somatic GOF Gly12/Gly13/Gln61; constitutive ERK/PI3K → lymphocyte survival → escape AICD despite FAS intact; apoptosis assay NORMAL; ddPCR required; MEKi cobimetinib 88% response)', inh: 'Somatic mosaic GOF' },
  'KRAS':   { full: 'KRAS / K-Ras / 189aa (Somatic GOF — ALPS-RALD2)', locus: '12p12.1', size: '189 aa / 21 kDa (somatic GOF same hotspots as NRAS; RALD type2; monocytosis MORE prominent; JMML risk if myeloid progenitor; KRAS also causes Noonan/CFC germline — RALD is somatic only; MEKi cobimetinib 85% response)', inh: 'Somatic mosaic GOF' },
};

const PATHOGNOMONIC = {
  'FAS':    'DNT cells ≥1.5% + apoptosis assay ABSENT (FasL + CH11 both negative)',
  'FASLG':  'Apoptosis absent to FasL ONLY; CH11 (anti-FAS agonist) RESTORES apoptosis — receptor intact',
  'CASP10': 'DNT elevated + apoptosis REDUCED (not absent) — partial DISC defect; CASP8 activity (IETD-AMC) reduced',
  'CASP8':  'ALPS + hypogammaglobulinemia (<500 mg/dL) + herpesvirus = CED — ONLY ALPS type with IgG low',
  'FADD':   'ALPS + dilated CARDIOMYOPATHY in child = FADD until proven otherwise; echo mandatory at diagnosis',
  'PRKCD':  'SLE-like onset <5yr + consanguinity + ANA ≥1:640 + anti-dsDNA very high + DNT may be NORMAL',
  'NRAS':   'DNT elevated + organomegaly + apoptosis assay NORMAL → RALD; ddPCR for NRAS Gly12/13/Gln61 in lymphocytes',
  'KRAS':   'Lymphocytosis + MONOCYTOSIS >1.5×10⁹/L + DNT + apoptosis assay NORMAL → KRAS-RALD; JMML workup mandatory',
};

export default function HereditaryALPSFASFASLPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedDef, setExpandedDef] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#1a1a2e', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-ALPS-FAS-FASL-Lymphoproliferation-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#1a1a2e', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#4a148c 50%,#1565c0 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Primary Immunodeficiency · ALPS · FAS Pathway · Lymphoproliferation</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-ALPS-FAS-FASL-Lymphoproliferation-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · FAS-TNFRSF6 · FASLG · CASP10 · CASP8 · FADD · PRKCD · NRAS · KRAS ·
          320 Patients (8×40) · Seeds 4062–4069
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'DNT Hallmark', val: '≥1.5% CD3+TCRαβ+CD4−CD8−' },
            { label: 'Apoptosis Assay', val: 'FasL + CH11 stimulated' },
          ].map(b => (
            <div key={b.label} style={{ background: 'rgba(255,255,255,0.15)', borderRadius: 8, padding: '6px 14px', fontSize: 13 }}>
              <span style={{ color: 'rgba(255,255,255,0.7)' }}>{b.label}: </span>
              <strong style={{ color: '#fff' }}>{b.val}</strong>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid #30363d', background: '#161b22' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: '12px 24px', border: 'none', background: 'none', color: activeTab === t ? '#58a6ff' : '#8b949e',
              borderBottom: activeTab === t ? '2px solid #58a6ff' : '2px solid transparent',
              cursor: 'pointer', fontSize: 14, fontWeight: activeTab === t ? 600 : 400 }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ padding: '24px 28px' }}>

        {/* OVERVIEW TAB */}
        {activeTab === 'Overview' && overview && (
          <div>
            <div style={{ marginBottom: 20, padding: 16, background: '#161b22', borderRadius: 8, border: '1px solid #30363d' }}>
              <h3 style={{ margin: '0 0 12px', color: '#58a6ff', fontSize: 15 }}>Atlas Key Facts</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 10 }}>
                {overview.key_facts && Object.entries(overview.key_facts).map(([k, v]) => (
                  <div key={k} style={{ padding: '8px 12px', background: '#0d1117', borderRadius: 6, border: '1px solid #30363d', fontSize: 12 }}>
                    <strong style={{ color: '#e6edf3', textTransform: 'uppercase', fontSize: 10, letterSpacing: 1 }}>{k.replace(/_/g, ' ')}: </strong>
                    <span style={{ color: '#8b949e' }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(300px,1fr))', gap: 16 }}>
              {overview.genes?.map(gene => {
                const c = overview.cohort_summary?.[gene];
                if (!c) return null;
                return (
                  <div key={gene} style={{ background: '#161b22', borderRadius: 10, border: `1px solid ${GENE_COLORS[gene]}44`, overflow: 'hidden' }}>
                    <div style={{ background: GENE_COLORS[gene], padding: '10px 14px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <strong style={{ color: '#fff', fontSize: 15 }}>{gene}</strong>
                        <span style={{ color: 'rgba(255,255,255,0.8)', fontSize: 11 }}>{c.inheritance}</span>
                      </div>
                      <div style={{ color: 'rgba(255,255,255,0.85)', fontSize: 11, marginTop: 3 }}>{c.alps_type}</div>
                      <div style={{ color: 'rgba(255,255,255,0.75)', fontSize: 11 }}>{c.locus} · {c.size_aa}aa / {c.kda}kDa</div>
                    </div>
                    <div style={{ padding: '10px 14px' }}>
                      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 6 }}>{GENE_INFO[gene]?.size}</div>
                      <div style={{ fontSize: 11, color: '#ffa657', marginBottom: 6, padding: '4px 8px', background: '#2d1f00', borderRadius: 4 }}>
                        ⚡ {PATHOGNOMONIC[gene]}
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
                        {[
                          ['Patients', c.n],
                          ['DNT Positive', `${c.dnt_positive_pct}%`],
                          ['Organomegaly', `${c.organomegaly_pct}%`],
                          ['Autoimmune Cytopenia', `${c.autoimmune_cytopenia_pct}%`],
                          ['Lymphoma Risk', `${c.lymphoma_risk_pct}%`],
                          ['Apoptosis Defect', `${c.apoptosis_assay_abnormal_pct}%`],
                          ['Sirolimus Response', `${c.sirolimus_response_pct}%`],
                          ['HSCT Required', `${c.hsct_required_pct}%`],
                          ['5-yr Survival', `${c.survival_5yr_pct}%`],
                          ['Hypogammaglobulin.', `${c.hypogammaglobulinemia_pct}%`],
                          ['Vit B12 Elevated', `${c.vit_b12_elevated_pct}%`],
                          ['Seed', c.seed],
                        ].map(([lbl, val]) => (
                          <div key={lbl} style={{ background: '#0d1117', borderRadius: 4, padding: '3px 6px' }}>
                            <div style={{ color: '#8b949e', fontSize: 10 }}>{lbl}</div>
                            <div style={{ color: '#e6edf3', fontWeight: 600 }}>{val}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {activeTab === 'Gene Table' && breakdown && (
          <div>
            <div style={{ marginBottom: 16, padding: 12, background: '#161b22', borderRadius: 8, border: '1px solid #30363d', fontSize: 12, color: '#8b949e' }}>
              <strong style={{ color: '#58a6ff' }}>Diagnostic Groups: </strong>
              {breakdown.diagnostic_groups && Object.entries(breakdown.diagnostic_groups).map(([gk, gv]) => (
                <span key={gk} style={{ marginRight: 16 }}>
                  <strong style={{ color: '#e6edf3' }}>{gk.replace(/_/g, ' ')}</strong>: {gv.genes?.join(', ')} — {gv.key_test}
                </span>
              ))}
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#161b22' }}>
                    {['Gene', 'ALPS Type', 'N', 'DNT↑%', 'Organomeg%', 'Cytopenia%', 'Lymphoma%', 'Apoptosis Defect%', 'Sirolimus%', 'HSCT%', '5yr Surv%', 'Hypogam%', 'B12↑%', 'Mean DNT%', 'Mean Age Onset', 'Mean B12', 'MEKi%', 'Cardio%', 'Nephritis%'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', color: '#8b949e', fontWeight: 600, textAlign: 'left', borderBottom: '1px solid #30363d', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(breakdown.cohort_stats || {}).map(([gene, cs]) => (
                    <tr key={gene} style={{ borderBottom: '1px solid #21262d' }}>
                      <td style={{ padding: '8px 10px' }}>
                        <span style={{ background: GENE_COLORS[gene], color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700, fontSize: 11 }}>{gene}</span>
                      </td>
                      <td style={{ padding: '8px 10px', color: '#8b949e', fontSize: 11, maxWidth: 180 }}>{cs.alps_type}</td>
                      <td style={{ padding: '8px 10px', color: '#e6edf3', fontWeight: 600 }}>{cs.n}</td>
                      <td style={{ padding: '8px 10px', color: cs.dnt_elevated_n / cs.n > 0.8 ? '#7ee787' : '#e6edf3' }}>{Math.round(cs.dnt_elevated_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: '#e6edf3' }}>{Math.round(cs.organomegaly_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: '#e6edf3' }}>{Math.round(cs.cytopenia_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: cs.lymphoma_n > 5 ? '#f85149' : '#e6edf3' }}>{Math.round(cs.lymphoma_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: cs.apoptosis_defect_n / cs.n > 0.8 ? '#7ee787' : '#ffa657' }}>{Math.round(cs.apoptosis_defect_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: '#e6edf3' }}>{Math.round(cs.sirolimus_resp_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: cs.hsct_n > 6 ? '#ffa657' : '#e6edf3' }}>{Math.round(cs.hsct_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: cs.survived_5yr_n / cs.n > 0.9 ? '#7ee787' : '#ffa657' }}>{Math.round(cs.survived_5yr_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: cs.hypogam_n > 20 ? '#f85149' : '#e6edf3' }}>{Math.round(cs.hypogam_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: '#e6edf3' }}>{Math.round(cs.b12_high_n / cs.n * 100)}%</td>
                      <td style={{ padding: '8px 10px', color: cs.mean_dnt_pct > 3 ? '#ffa657' : '#e6edf3' }}>{cs.mean_dnt_pct}%</td>
                      <td style={{ padding: '8px 10px', color: '#8b949e' }}>{cs.mean_age_onset} yr</td>
                      <td style={{ padding: '8px 10px', color: '#8b949e' }}>{cs.mean_b12} pg/mL</td>
                      <td style={{ padding: '8px 10px', color: cs.mek_resp_n > 5 ? '#7ee787' : '#8b949e' }}>{cs.mek_resp_n > 0 ? `${Math.round(cs.mek_resp_n / cs.n * 100)}%` : '—'}</td>
                      <td style={{ padding: '8px 10px', color: cs.cardiomyopathy_n > 10 ? '#f85149' : '#8b949e' }}>{cs.cardiomyopathy_n > 0 ? `${Math.round(cs.cardiomyopathy_n / cs.n * 100)}%` : '—'}</td>
                      <td style={{ padding: '8px 10px', color: cs.nephritis_n > 10 ? '#f85149' : '#8b949e' }}>{cs.nephritis_n > 0 ? `${Math.round(cs.nephritis_n / cs.n * 100)}%` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {activeTab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 16, marginBottom: 20 }}>
              {breakdown.diagnostic_groups && Object.entries(breakdown.diagnostic_groups).map(([gk, gv]) => (
                <div key={gk} style={{ background: '#161b22', borderRadius: 10, border: '1px solid #30363d', padding: 16 }}>
                  <div style={{ fontWeight: 700, color: '#58a6ff', fontSize: 14, marginBottom: 8 }}>{gk.replace(/_/g, ' ').toUpperCase()}</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                    {gv.genes?.map(g => (
                      <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 700 }}>{g}</span>
                    ))}
                  </div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 6 }}>{gv.description}</div>
                  <div style={{ fontSize: 11, color: '#ffa657', padding: '4px 8px', background: '#2d1f00', borderRadius: 4 }}>
                    Key Test: {gv.key_test}
                  </div>
                </div>
              ))}
            </div>
            {/* Treatment matrix */}
            {definitions?.treatment_matrix && (
              <div style={{ background: '#161b22', borderRadius: 10, border: '1px solid #30363d', padding: 16 }}>
                <h3 style={{ margin: '0 0 14px', color: '#58a6ff', fontSize: 15 }}>Treatment Matrix</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(300px,1fr))', gap: 10 }}>
                  {Object.entries(definitions.treatment_matrix).map(([k, v]) => (
                    <div key={k} style={{ background: '#0d1117', borderRadius: 8, padding: '10px 12px', border: '1px solid #21262d' }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: '#e6edf3', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5 }}>{k.replace(/_/g, ' ')}</div>
                      <div style={{ fontSize: 11, color: '#8b949e' }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {/* Key distinguishers */}
            {definitions?.key_distinguishers && (
              <div style={{ background: '#161b22', borderRadius: 10, border: '1px solid #30363d', padding: 16, marginTop: 16 }}>
                <h3 style={{ margin: '0 0 14px', color: '#58a6ff', fontSize: 15 }}>Key Distinguishers</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 8 }}>
                  {Object.entries(definitions.key_distinguishers).map(([k, v]) => (
                    <div key={k} style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', border: '1px solid #21262d', fontSize: 11 }}>
                      <strong style={{ color: '#58a6ff' }}>{k.replace(/_/g, ' ')}: </strong>
                      <span style={{ color: '#8b949e' }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {activeTab === 'Definitions' && definitions && (
          <div>
            {definitions.definitions?.map((d, i) => (
              <div key={i} style={{ marginBottom: 12, background: '#161b22', borderRadius: 8, border: '1px solid #30363d', overflow: 'hidden' }}>
                <button
                  onClick={() => setExpandedDef(expandedDef === i ? null : i)}
                  style={{ width: '100%', padding: '12px 16px', background: 'none', border: 'none', textAlign: 'left', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#e6edf3', fontWeight: 600, fontSize: 14 }}>{d.term}</span>
                  <span style={{ color: '#58a6ff', fontSize: 18 }}>{expandedDef === i ? '−' : '+'}</span>
                </button>
                {expandedDef === i && (
                  <div style={{ padding: '0 16px 16px', color: '#8b949e', fontSize: 13, lineHeight: 1.7, borderTop: '1px solid #21262d' }}>
                    <div style={{ marginTop: 12 }}>{d.definition}</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
