'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-immune-dysregulation-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FAS:    '#b71c1c',  // deep red         — ALPS-Ia, most common, FAS apoptosis, DNT cells
  FASLG:  '#c62828',  // red              — ALPS-Ib, FasL deficiency, sFasL elevated
  CASP10: '#1565c0',  // deep blue        — ALPS-IIa, caspase-10, somatic mosaicism
  CASP8:  '#0d47a1',  // navy             — ALPS-IIb + CID, dual role, AR, infections + lympho
  FOXP3:  '#6a1b9a',  // deep purple      — IPEX, TREG absent, neonatal T1DM, X-linked
  WAS:    '#1b5e20',  // dark green       — Wiskott-Aldrich, microthrombocytopenia, small platelets
  DOCK8:  '#e65100',  // burnt orange     — DOCK8 deficiency, HSV dissemination, CD8 lymphopenia
  STAT3:  '#004d40',  // dark teal        — STAT3-GOF, JAK inhibitors, lymphoproliferation + autoimmunity
};

const GENE_INFO = {
  FAS:    { full: 'FAS/CD95/APO-1 / 335aa', locus: '10q23.31', size: '335 aa / 38 kDa', inh: 'AD (heterozygous)', disease: 'ALPS Type Ia — MOST COMMON ALPS (~70%); chronic non-malignant lymphadenopathy + splenomegaly + autoimmune cytopenias TRIAD; DNT cells (CD3+CD4-CD8-TCRαβ+) >1.5% PATHOGNOMONIC; sirolimus ACNS 2020 first-line; lymphoma 10-50x risk' },
  FASLG:  { full: 'FASLG/FasL/CD95L / 281aa', locus: '1q24.3', size: '281 aa / 37 kDa', inh: 'AD (rare)', disease: 'ALPS Type Ib — same ALPS triad as FAS-Ia; FAS apoptosis assay NORMAL (receptor intact); elevated sFasL (>200 pg/mL) BIOMARKER; FASLG functional assay impaired; genetic testing required for distinction' },
  CASP10: { full: 'CASP10/Caspase-10 / 521aa', locus: '2q33.1', size: '521 aa / 59 kDa', inh: 'AD', disease: 'ALPS Type IIa — identical ALPS phenotype; all FAS-pathway assays NORMAL; somatic reversion mosaicism documented (germline testing may miss); CASP10 sequencing or deep sequencing of DNT cells required' },
  CASP8:  { full: 'CASP8/Caspase-8 / 479aa', locus: '2q33.1', size: '479 aa / 55 kDa', inh: 'AR (biallelic)', disease: 'ALPS Type IIb + Combined Immunodeficiency — UNIQUE: ALPS lymphoproliferation + recurrent bacterial/herpesviral infections SIMULTANEOUSLY PATHOGNOMONIC; T-cell activation also impaired; AR (biallelic), unlike CASP10 AD; IVIG + antivirals mandatory' },
  FOXP3:  { full: 'FOXP3/Scurfin / 431aa', locus: 'Xp11.23', size: '431 aa / 47 kDa', inh: 'XLR (boys)', disease: 'IPEX — neonatal T1DM (within weeks of birth) + intractable secretory enteropathy + severe eczema TRIAD IN BOYS PATHOGNOMONIC; TREG absent (CD4+CD25+FoxP3+ <1%); FATAL without HSCT in severe; tacrolimus/sirolimus bridge' },
  WAS:    { full: 'WAS/WASp / 502aa', locus: 'Xp11.22', size: '502 aa / 53 kDa', inh: 'XLR (boys)', disease: 'Wiskott-Aldrich Syndrome — microthrombocytopenia + eczema + combined immunodeficiency TRIAD; SMALL PLATELETS (MPV <7 fL) PATHOGNOMONIC — KEY DDx from ITP where MPV is HIGH; HSCT curative all three; gene therapy OTL-103 EMA 2022' },
  DOCK8:  { full: 'DOCK8 / 2099aa', locus: '9p24.3', size: '2099 aa / 238 kDa', inh: 'AR (biallelic)', disease: 'DOCK8 Deficiency — eczema + DISSEMINATED HSV/MOLLUSCUM (cutaneous herpesviral) + very high IgE; disseminated HSV PATHOGNOMONIC vs STAT3-HIES; progressive CD8 lymphopenia; lymphoma 10-15%; HSCT curative; prophylactic aciclovir MANDATORY' },
  STAT3:  { full: 'STAT3-GOF / 770aa', locus: '17q21.2', size: '770 aa / 92 kDa', inh: 'AD GOF', disease: 'STAT3 Gain-of-Function — lymphoproliferation + multi-organ autoimmunity + SHORT STATURE + early-onset T1DM; DISTINCT from STAT3-LOF (HIES1 — opposite phenotype); TREG reduced; JAK inhibitors (ruxolitinib/tofacitinib) HIGHLY EFFECTIVE DRAMATIC RESPONSE' },
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

export default function ImmuneDysregulationAtlasPage() {
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
          Hereditary Immune Dysregulation Atlas
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          🧬 Hereditary-Immune-Dysregulation-Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 6 }}>
          Complete 8-Gene Immune Dysregulation Reference — ALPS (FAS/FASLG/CASP10/CASP8) · IPEX (FOXP3) · WAS · DOCK8 · STAT3-GOF
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
              {overview.subtitle} · {overview.total_patients} patients · seeds {overview.seeds}
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
              <MetricCard label="Autoimmunity" value={`${overview.aggregate_metrics.autoimmunity_pct}%`} sub="across all genes" />
              <MetricCard label="Lymphoma Risk" value={`${overview.aggregate_metrics.lymphoma_pct}%`} sub="surveillance mandatory" />
              <MetricCard label="HSCT Received" value={`${overview.aggregate_metrics.hsct_received_pct}%`} sub="curative for WAS/FOXP3/DOCK8" />
              <MetricCard label="Sirolimus/JAK-i" value={`${overview.aggregate_metrics.sirolimus_or_jak_inhibitor_pct}%`} sub="targeted therapy" />
              <MetricCard label="ALPS DNT Elevated" value={`${overview.aggregate_metrics.alps_dnt_elevated_pct}%`} sub=">1.5% of lymphocytes" />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 14 }}>
              {Object.entries(overview.gene_summary || {}).map(([gene, gs]) => (
                <div key={gene} style={{ background: '#1e293b', borderRadius: 10, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <GeneChip gene={gene} />
                    <span style={{ color: '#94a3b8', fontSize: 11 }}>{gs.locus} · {gs.protein_size} · {gs.inheritance}</span>
                  </div>
                  <div style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{gs.disease_category}</div>
                  <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 8 }}>{gs.pathognomonic?.substring(0, 180)}…</div>
                  <div style={{ display: 'flex', gap: 16, fontSize: 11, color: '#64748b' }}>
                    <span>Autoimmune: <b style={{ color: '#fbbf24' }}>{gs.autoimmunity_pct}%</b></span>
                    <span>Lymphoma: <b style={{ color: '#f87171' }}>{gs.lymphoma_pct}%</b></span>
                    <span>HSCT: <b style={{ color: '#34d399' }}>{gs.hsct_pct}%</b></span>
                  </div>
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
                  {['Gene','Locus','Size','Inh','Disease','Autoimmunity%','Lymphoma%','HSCT%','Siro/JAK%','Avg DNT%','Avg Dx Age'].map(h => (
                    <th key={h} style={{ padding: '10px 10px', textAlign: 'left', borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.gene_breakdowns.map(g => (
                  <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '9px 10px' }}><GeneChip gene={g.gene} /></td>
                    <td style={{ padding: '9px 10px', color: '#94a3b8' }}>{g.locus}</td>
                    <td style={{ padding: '9px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                    <td style={{ padding: '9px 10px', color: '#94a3b8' }}>{g.inheritance?.split(';')[0]}</td>
                    <td style={{ padding: '9px 10px', color: '#cbd5e1', maxWidth: 200 }}>{g.disease_category?.substring(0, 60)}…</td>
                    <td style={{ padding: '9px 10px', color: '#fbbf24', textAlign: 'center' }}>{g.autoimmunity_pct}%</td>
                    <td style={{ padding: '9px 10px', color: '#f87171', textAlign: 'center' }}>{g.lymphoma_pct}%</td>
                    <td style={{ padding: '9px 10px', color: '#34d399', textAlign: 'center' }}>{g.hsct_pct}%</td>
                    <td style={{ padding: '9px 10px', color: '#38bdf8', textAlign: 'center' }}>{g.sirolimus_jak_pct}%</td>
                    <td style={{ padding: '9px 10px', color: '#a78bfa', textAlign: 'center' }}>{g.avg_dnt_pct}%</td>
                    <td style={{ padding: '9px 10px', color: '#94a3b8', textAlign: 'center' }}>{g.avg_age_dx_years}y</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'grid', gap: 20 }}>
            {breakdown.gene_breakdowns.map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 12, padding: 20, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 15 }}>{g.disease_category}</span>
                  <span style={{ color: '#64748b', fontSize: 11 }}>{g.locus} · {g.protein_size} · {g.inheritance?.split(';')[0]}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                  <div>
                    <div style={{ color: '#38bdf8', fontSize: 11, fontWeight: 600, marginBottom: 4 }}>PATHOGNOMONIC</div>
                    <div style={{ color: '#cbd5e1', fontSize: 12 }}>{g.pathognomonic?.substring(0, 350)}</div>
                  </div>
                  <div>
                    <div style={{ color: '#fbbf24', fontSize: 11, fontWeight: 600, marginBottom: 4 }}>TREATMENT</div>
                    <div style={{ color: '#cbd5e1', fontSize: 12 }}>{g.treatment?.substring(0, 350)}</div>
                  </div>
                </div>
                <div style={{ marginBottom: 10 }}>
                  <div style={{ color: '#f87171', fontSize: 11, fontWeight: 600, marginBottom: 4 }}>KEY DDx</div>
                  <div style={{ color: '#94a3b8', fontSize: 12 }}>{g.key_ddx?.substring(0, 280)}</div>
                </div>
                <div style={{ display: 'flex', gap: 20, fontSize: 11 }}>
                  <span style={{ color: '#64748b' }}>Autoimmunity: <b style={{ color: '#fbbf24' }}>{g.autoimmunity_risk?.substring(0, 50)}</b></span>
                  <span style={{ color: '#64748b' }}>Lymphoma: <b style={{ color: '#f87171' }}>{g.lymphoma_risk?.substring(0, 40)}</b></span>
                </div>
                <div style={{ display: 'flex', gap: 20, fontSize: 11, marginTop: 4 }}>
                  <span style={{ color: '#64748b' }}>Sirolimus: <b style={{ color: '#34d399' }}>{g.sirolimus_response?.substring(0, 40)}</b></span>
                  <span style={{ color: '#64748b' }}>HSCT: <b style={{ color: '#a78bfa' }}>{g.hsct_required}</b></span>
                  <span style={{ color: '#64748b' }}>Onset: <b style={{ color: '#94a3b8' }}>{g.onset_age?.substring(0, 40)}</b></span>
                </div>
                {g.sample_patients?.length > 0 && (
                  <div style={{ marginTop: 12, borderTop: '1px solid #334155', paddingTop: 10 }}>
                    <div style={{ color: '#64748b', fontSize: 10, marginBottom: 6 }}>SAMPLE PATIENTS (n={g.n_patients})</div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {g.sample_patients.map(p => (
                        <div key={p.patient_id} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px', fontSize: 10, color: '#94a3b8' }}>
                          <b>{p.patient_id}</b> · Dx {p.age_at_diagnosis_years}y · DNT {p.dnt_pct_of_lymphocytes}% · {p.outcome}
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
            <div style={{ display: 'grid', gap: 16, marginBottom: 32 }}>
              {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
                <div key={gene} style={{ background: '#1e293b', borderRadius: 10, padding: 18, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <GeneChip gene={gene} />
                    <span style={{ color: '#e2e8f0', fontWeight: 700 }}>{entry.disease_name}</span>
                    <span style={{ color: '#64748b', fontSize: 11 }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</span>
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8 }}>{entry.disease_pathway?.substring(0, 300)}</div>
                  <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                    {(entry.key_features || []).map((f, i) => (
                      <div key={i} style={{ background: '#0f172a', borderRadius: 4, padding: '3px 8px', fontSize: 11, color: '#cbd5e1' }}>• {f}</div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 16, marginBottom: 14 }}>
              Immune Dysregulation Glossary
            </div>
            <div style={{ display: 'grid', gap: 12 }}>
              {Object.entries(definitions.immune_dysregulation_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
                  <div style={{ color: '#38bdf8', fontWeight: 700, marginBottom: 6 }}>{term}</div>
                  <div style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
