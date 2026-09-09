'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ectopia-lentis-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FBN1:     '#1a237e',  // deep indigo       — Marfan, fibrillin-1, superotemporal EL, aortic root
  CBS:      '#b71c1c',  // deep red           — Homocystinuria, inferonasal EL, thromboembolism, B6-test
  ADAMTSL4: '#0277bd',  // steel blue         — Isolated EL/ELP, pure ocular, no systemic
  ADAMTS10: '#004d40',  // dark teal          — WMS2, microspherophakia, inverse Marfan, brachydactyly
  ADAMTS17: '#1b5e20',  // dark green         — WMS4, milder WMS2, skin laxity subset
  LTBP2:    '#4a148c',  // deep purple        — Microspherophakia + glaucoma, Gulf Arab founder
  SUOX:     '#e65100',  // burnt orange       — Sulfite oxidase deficiency, neonatal seizures, sulfite dipstick
  FBN2:     '#37474f',  // blue-grey          — CCA/Beals, crumpled ear, contractures, no aortic dilation
};

const GENE_INFO = {
  FBN1:     { full: 'FBN1 / Fibrillin-1 / 2871aa', locus: '15q21.1', size: '2871 aa / 350 kDa', inh: 'AD', disease: 'Marfan syndrome — BILATERAL SUPEROTEMPORAL ECTOPIA LENTIS PATHOGNOMONIC; aortic root dilation (Z>2); tall marfanoid habitus + arachnodactyly; wrist/thumb signs; LOSARTAN + beta-blocker mandatory; prophylactic aortic root surgery >50mm; Revised Ghent 2010 criteria' },
  CBS:      { full: 'CBS / Cystathionine Beta-Synthase / 551aa', locus: '21q22.3', size: '551 aa / 63 kDa', inh: 'AR', disease: 'Classic homocystinuria — BILATERAL INFERONASAL ECTOPIA LENTIS PATHOGNOMONIC (OPPOSITE to Marfan); elevated plasma homocysteine >100 µmol/L; THROMBOEMBOLISM risk DVT/PE/stroke; B6-RESPONSIVENESS TEST MANDATORY (50% respond); ANAESTHESIA HIGH RISK — LMWH + hydration pre-op; marfanoid but inferonasal EL' },
  ADAMTSL4: { full: 'ADAMTSL4 / ADAMTS-Like Protein 4 / 1212aa', locus: '2q36.1', size: '1212 aa / 138 kDa', inh: 'AR', disease: 'Isolated ectopia lentis (IREL) / Ectopia lentis et pupillae (ELP) — PURE OCULAR, NO SYSTEMIC; ELP: lens + pupil displaced in OPPOSITE directions (pathognomonic for ADAMTSL4 most common cause); normal homocysteine + normal aortic root + normal stature; fibrillin-1 microfibril assembly defect in ciliary body' },
  ADAMTS10: { full: 'ADAMTS10 / ADAMTS Metalloprotease 10 / 1103aa', locus: '19p13.2', size: '1103 aa / 125 kDa', inh: 'AR', disease: 'Weill-Marchesani syndrome type 2 (WMS2) — MICROSPHEROPHAKIA + ANTERIOR SUBLUXATION + SHORT STATURE + BRACHYDACTYLY = INVERSE MARFAN; AVOID MIOTICS ABSOLUTELY (pupillary block); LASER IRIDOTOMY PROPHYLACTIC; joint stiffness (vs Marfan hypermobility); acute angle closure glaucoma emergency' },
  ADAMTS17: { full: 'ADAMTS17 / ADAMTS Metalloprotease 17 / 1221aa', locus: '15q26.3', size: '1221 aa / 139 kDa', inh: 'AR', disease: 'Weill-Marchesani syndrome type 4 (WMS4) — MILDER WMS2; microspherophakia + short stature + brachydactyly + OPTIONAL MILD SKIN LAXITY; AVOID miotics (same CI as WMS2); genetic panel (ADAMTS10 + ADAMTS17) required to distinguish; milder brachydactyly vs WMS2' },
  LTBP2:    { full: 'LTBP2 / Latent TGF-β Binding Protein 2 / 1821aa', locus: '14q24.3', size: '1821 aa / 200 kDa', inh: 'AR', disease: 'Microspherophakia with secondary glaucoma — SPHERICAL SUBLUXATED LENS + PUPILLARY BLOCK GLAUCOMA PATHOGNOMONIC; AVOID miotics ABSOLUTELY; LASER PI EMERGENCY; GULF ARAB FOUNDER pArg299Cys; Pakistani consanguineous; PCG variant (GLC3F) in subset; CYP1B1+LTBP2 digenic synergistic' },
  SUOX:     { full: 'SUOX / Sulfite Oxidase / 545aa', locus: '12q13.2', size: '545 aa / 60 kDa', inh: 'AR', disease: 'Isolated sulfite oxidase deficiency — ECTOPIA LENTIS + NEONATAL REFRACTORY SEIZURES + URINE SULFITE DIPSTICK POSITIVE (FRESH urine only) PATHOGNOMONIC; KEY DDx HCU: NORMAL plasma homocysteine; KEY DDx MoCoD: normal xanthine + normal uric acid; NO proven treatment; neonatal onset; cystic leukomalacia' },
  FBN2:     { full: 'FBN2 / Fibrillin-2 / 2832aa', locus: '5q23.3', size: '2832 aa / 330 kDa', inh: 'AD', disease: 'Congenital contractural arachnodactyly (CCA/Beals-Hecht) — CRUMPLED PINNA + CONGENITAL JOINT CONTRACTURES PATHOGNOMONIC; KEY DDx Marfan (FBN1): NO aortic root dilation; STIFF contractural joints (vs Marfan hypermobile); contractures improve with physiotherapy; EL in ~20%; fetal fibrillin-2 paralogue' },
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

export default function ELAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = tab === 'Overview' || tab === 'Gene Table' ? 'overview'
             : tab === 'Clinical Atlas' ? 'breakdown'
             : 'definitions';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const dark = { background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '24px 32px' };
  const tabBar = { display: 'flex', gap: 8, marginBottom: 24, borderBottom: '1px solid #334155', paddingBottom: 8 };
  const tabBtn = (active) => ({
    background: active ? '#3b82f6' : 'transparent',
    color: active ? '#fff' : '#94a3b8',
    border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontWeight: active ? 700 : 400,
  });

  return (
    <div style={dark}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary Ectopia Lentis Atlas
        </h1>
        <p style={{ color: '#94a3b8', fontSize: 14, margin: 0 }}>
          Complete 8-Gene Reference — FBN1 · CBS · ADAMTSL4 · ADAMTS10 · ADAMTS17 · LTBP2 · SUOX · FBN2<br />
          320-Patient Aggregate Cohort (8×40) · Seeds 2398–2405 · Marfan · Homocystinuria · WMS · Isolated EL · Sulfite Oxidase
        </p>
      </div>

      {/* Gene chips */}
      <div style={{ marginBottom: 20, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
      </div>

      <div style={tabBar}>
        {TABS.map(t => (
          <button key={t} style={tabBtn(tab === t)} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#94a3b8' }}>Loading…</div>}
      {error && <div style={{ color: '#f87171' }}>Error: {error}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
            <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
            <MetricCard label="EL Present" value={`${overview.aggregate_metrics?.el_present_pct ?? '--'}%`} sub="ectopia lentis rate" />
            <MetricCard label="Microspherophakia" value={`${overview.aggregate_metrics?.microspherophakia_pct ?? '--'}%`} sub="WMS/LTBP2 subset" />
            <MetricCard label="Glaucoma" value={`${overview.aggregate_metrics?.glaucoma_pct ?? '--'}%`} sub="across atlas" />
            <MetricCard label="Surgery Rate" value={`${overview.aggregate_metrics?.surgery_performed_pct ?? '--'}%`} sub="lensectomy / LPI" />
            <MetricCard label="Thromboembolism" value={`${overview.aggregate_metrics?.thromboembolism_pct ?? '--'}%`} sub="CBS HCU only" />
            <MetricCard label="VA < 6/18" value={`${overview.aggregate_metrics?.va_worse_than_6_18_pct ?? '--'}%`} sub="visual impairment" />
          </div>

          <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Summary</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
            {overview.gene_summary && Object.values(overview.gene_summary).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, minWidth: 260, flex: '1 1 260px', borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ color: '#94a3b8', fontSize: 11 }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
                </div>
                <div style={{ color: '#e2e8f0', fontSize: 12, marginBottom: 6, lineHeight: 1.4 }}>
                  {g.disease_category}
                </div>
                <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>{g.onset_age}</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                  <span style={{ color: '#60a5fa', fontSize: 11 }}>EL {g.el_pct}%</span>
                  <span style={{ color: '#a78bfa', fontSize: 11 }}>Micro {g.microspherophakia_pct}%</span>
                  <span style={{ color: '#fbbf24', fontSize: 11 }}>Glaucoma {g.glaucoma_pct}%</span>
                  <span style={{ color: '#34d399', fontSize: 11 }}>Surgery {g.surgery_pct}%</span>
                  <span style={{ color: '#f87171', fontSize: 11 }}>VA↓ {g.va_poor_pct}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && overview && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                {['Gene','Locus','Size','Inh','Disease','EL %','Micro %','Glaucoma %','Surgery %','VA↓ %','Systemic','Urgency'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', color: '#93c5fd', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.gene_summary && Object.values(overview.gene_summary).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b', borderBottom: '1px solid #334155' }}>
                  <td style={{ padding: '8px 10px' }}><GeneChip gene={g.gene} /></td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.locus}</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                  <td style={{ padding: '8px 10px', color: '#fbbf24', whiteSpace: 'nowrap' }}>{g.inheritance}</td>
                  <td style={{ padding: '8px 10px', color: '#e2e8f0', maxWidth: 200 }}>{g.disease_category}</td>
                  <td style={{ padding: '8px 10px', color: '#60a5fa', textAlign: 'center' }}>{g.el_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#a78bfa', textAlign: 'center' }}>{g.microspherophakia_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#fbbf24', textAlign: 'center' }}>{g.glaucoma_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#34d399', textAlign: 'center' }}>{g.surgery_pct}%</td>
                  <td style={{ padding: '8px 10px', color: '#f87171', textAlign: 'center' }}>{g.va_poor_pct}%</td>
                  <td style={{ padding: '8px 10px', color: g.systemic_involvement ? '#f87171' : '#94a3b8' }}>{g.systemic_involvement ? 'Yes' : 'No'}</td>
                  <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 11 }}>{g.surgical_urgency}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: 24 }}>
            <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Reference Detail</h3>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 10, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <GeneChip gene={gene} />
                  <span style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600 }}>{info.full}</span>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>· {info.locus} · {info.size} · {info.inh}</span>
                </div>
                <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{info.disease}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.gene_breakdowns?.map(g => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: 18, marginBottom: 16, borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                <GeneChip gene={g.gene} />
                <span style={{ color: '#e2e8f0', fontWeight: 700 }}>{g.disease_category}</span>
                <span style={{ color: '#94a3b8', fontSize: 12 }}>· {g.locus} · {g.protein_size} · {g.inheritance}</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div>
                  <div style={{ color: '#fbbf24', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>PATHOGNOMONIC</div>
                  <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.5 }}>{g.pathognomonic}</div>
                </div>
                <div>
                  <div style={{ color: '#60a5fa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY DDx</div>
                  <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{g.key_ddx}</div>
                </div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>TREATMENT / MANAGEMENT</div>
                <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, whiteSpace: 'pre-line' }}>{g.treatment}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ color: '#a78bfa', fontSize: 11, fontWeight: 700, marginBottom: 4 }}>KEY FEATURES</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {g.key_features?.map((f, i) => (
                    <span key={i} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#cbd5e1' }}>{f}</span>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', borderTop: '1px solid #334155', paddingTop: 10, marginTop: 8 }}>
                <span style={{ color: '#60a5fa', fontSize: 12 }}>EL {g.el_pct}%</span>
                <span style={{ color: '#a78bfa', fontSize: 12 }}>Microspherophakia {g.microspherophakia_pct}%</span>
                <span style={{ color: '#fbbf24', fontSize: 12 }}>Glaucoma {g.glaucoma_pct}%</span>
                <span style={{ color: '#34d399', fontSize: 12 }}>Surgery {g.surgery_pct}%</span>
                <span style={{ color: '#f87171', fontSize: 12 }}>VA↓ {g.va_poor_pct}%</span>
                <span style={{ color: '#94a3b8', fontSize: 12 }}>Consanguineous {g.consanguineous_pct}%</span>
                <span style={{ color: '#94a3b8', fontSize: 12 }}>Systemic: {g.systemic_involvement ? 'Yes' : 'No'}</span>
                <span style={{ color: '#64748b', fontSize: 11 }}>Urgency: {g.surgical_urgency}</span>
              </div>

              {g.sample_patients?.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Sample Patients (n=3)</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {g.sample_patients.map(p => (
                      <div key={p.id} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px', fontSize: 11, color: '#cbd5e1' }}>
                        <b>{p.id}</b> · EL: {p.el_present ? p.el_direction : 'N'} · Micro: {p.microspherophakia ? 'Y' : 'N'} · Glaucoma: {p.glaucoma ? 'Y' : 'N'} · Surg: {p.surgery_performed ? 'Y' : 'N'} · VA↓: {p.va_poor ? 'Y' : 'N'}
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
          <h3 style={{ color: '#93c5fd', marginBottom: 12 }}>Gene Definitions</h3>
          {Object.values(definitions.gene_entries || {}).map(e => (
            <div key={e.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[e.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <GeneChip gene={e.gene} />
                <span style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600 }}>{e.full_name}</span>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>· {e.locus} · {e.protein_size} · {e.inheritance}</span>
              </div>
              <div style={{ color: '#fbbf24', fontSize: 11, marginBottom: 4 }}>{e.disease_name}</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, marginBottom: 6 }}>{e.disease_pathway}</div>
              <div style={{ color: '#a78bfa', fontSize: 11, marginBottom: 2, fontWeight: 700 }}>PATHOGNOMONIC</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5, marginBottom: 6 }}>{e.pathognomonic}</div>
              <div style={{ color: '#34d399', fontSize: 11, fontWeight: 700, marginBottom: 2 }}>TREATMENT</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.5 }}>{e.treatment}</div>
            </div>
          ))}

          <h3 style={{ color: '#93c5fd', marginTop: 24, marginBottom: 12 }}>Ectopia Lentis Glossary</h3>
          {Object.entries(definitions.el_glossary || {}).map(([term, def]) => (
            <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: 14, marginBottom: 10 }}>
              <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>{term}</div>
              <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 }}>{def}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
