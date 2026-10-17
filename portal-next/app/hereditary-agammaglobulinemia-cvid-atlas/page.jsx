'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-agammaglobulinemia-cvid-atlas';
const TABS = ['Overview', 'Gene Table', 'CVID Classification Map', 'Definitions'];

const GENE_COLORS = {
  'BTK':    '#0d47a1',  // deep blue         -- XLA most common; X-linked males only; echovirus PATHOGNOMONIC
  'IGHM':   '#1b5e20',  // dark green        -- AR Agamma Type 1; female key; pre-BCR Igμ
  'IGLL1':  '#4a148c',  // deep purple       -- AR Agamma Type 2; lambda5; smallest 214aa
  'BLNK':   '#880e4f',  // deep pink         -- AR Agamma Type 4; SLP-65; functional BTK phenocopy
  'PIK3CD': '#e65100',  // burnt orange      -- APDS1 AD-GOF; leniolisib FDA2023; B cells present
  'PIK3R1': '#827717',  // dark yellow-olive -- APDS2 AR; p85α; same pathway as PIK3CD
  'LRBA':   '#004d40',  // deep teal         -- LRBA deficiency; CTLA4 surface loss; abatacept
  'NFKB1':  '#b71c1c',  // deep red          -- CVID Type 7 most common genetic; splenomegaly+IBD
};

const GENE_INFO = {
  'BTK':    { full: 'BTK-659aa / XL-XLA-85pct / B-Cells-Absent-PATHOGNOMONIC / Echovirus-Encephalitis-PATHOGNOMONIC / IVIG-Trough-GT800-Lifelong', locus: 'Xq22.1', size: '659 aa / 76 kDa', inh: 'XL (males)' },
  'IGHM':   { full: 'IGHM-576aa / AR-Agamma-Type1 / B-Cells-Absent / Female-Key-Distinction / Pre-BCR-Igu-Chain / IVIG-Lifelong', locus: '14q32.33', size: '576 aa / 65 kDa', inh: 'AR' },
  'IGLL1':  { full: 'IGLL1-214aa / AR-Agamma-Type2 / Lambda5-Surrogate-Light-Chain / Smallest-214aa / Pre-BCR-Absent / IVIG-Lifelong', locus: '22q11.23', size: '214 aa / 24 kDa', inh: 'AR' },
  'BLNK':   { full: 'BLNK-456aa / AR-Agamma-Type4 / SLP65-BTK-PLCgamma2-Linker / Functional-Phenocopy-BTK / IVIG-Lifelong', locus: '10q24.1', size: '456 aa / 50 kDa', inh: 'AR' },
  'PIK3CD': { full: 'PIK3CD-1044aa / AD-GOF-APDS1 / B-Cells-Present-Unlike-XLA / EBV-CMV-Susceptibility / Leniolisib-FDA2023 / Bronchiectasis', locus: '1p36.22', size: '1044 aa / 119 kDa', inh: 'AD-GOF' },
  'PIK3R1': { full: 'PIK3R1-724aa / AR-APDS2 / p85alpha-Inhibitory-Subunit-Lost / Same-Downstream-as-APDS1 / Leniolisib-Responsive', locus: '5q13.1', size: '724 aa / 84 kDa', inh: 'AR' },
  'LRBA':   { full: 'LRBA-2863aa / AR-LRBA-Deficiency / CTLA4-Surface-Reduced / Multi-Organ-Autoimmunity / Abatacept-HIGHLY-EFFECTIVE / Misdiagnosed-CVID', locus: '4q31.3', size: '2863 aa / 319 kDa', inh: 'AR' },
  'NFKB1':  { full: 'NFKB1-968aa / AD-Haploinsufficiency-CVID7 / Most-Common-Genetic-CVID-10pct / Splenomegaly-DISTINGUISHER / IBD-Colitis-DISTINGUISHER', locus: '4q24', size: '968 aa / 106 kDa', inh: 'AD-LOF' },
};

const PHENOTYPE_ROWS = [
  { gene: 'BTK',    inh: 'XL',      b_absent: '✓✓✓', agamma: '✓✓✓', sinopulm: '✓✓✓', ebv_cmv: '−',    autoimmune: '−',   lymphoprolif: '−',  gi: '✓✓',   echovirus: '✓✓✓', unique: 'Echovirus meningoencephalitis PATHOGNOMONIC; males only; Bruton 1952' },
  { gene: 'IGHM',   inh: 'AR',      b_absent: '✓✓✓', agamma: '✓✓✓', sinopulm: '✓✓✓', ebv_cmv: '−',    autoimmune: '−',   lymphoprolif: '−',  gi: '✓✓',   echovirus: '✓✓',  unique: 'AR: FEMALES affected; first gene in female with XLA phenotype' },
  { gene: 'IGLL1',  inh: 'AR',      b_absent: '✓✓✓', agamma: '✓✓✓', sinopulm: '✓✓',  ebv_cmv: '−',    autoimmune: '−',   lymphoprolif: '−',  gi: '✓',    echovirus: '✓',   unique: 'AR; lambda5 surrogate light chain; smallest protein 214aa/24kDa' },
  { gene: 'BLNK',   inh: 'AR',      b_absent: '✓✓✓', agamma: '✓✓✓', sinopulm: '✓✓',  ebv_cmv: '−',    autoimmune: '−',   lymphoprolif: '−',  gi: '✓',    echovirus: '✓',   unique: 'AR; SLP-65 adaptor; functional phenocopy BTK; identical management' },
  { gene: 'PIK3CD', inh: 'AD-GOF',  b_absent: '−',   agamma: '✓✓',  sinopulm: '✓✓✓', ebv_cmv: '✓✓✓', autoimmune: '✓✓',  lymphoprolif: '✓✓✓',gi: '✓',    echovirus: '±',   unique: 'B CELLS PRESENT (not absent); EBV/CMV; Leniolisib FDA2023 precision' },
  { gene: 'PIK3R1', inh: 'AR',      b_absent: '−',   agamma: '✓✓',  sinopulm: '✓✓✓', ebv_cmv: '✓✓',  autoimmune: '✓✓',  lymphoprolif: '✓✓', gi: '±',    echovirus: '±',   unique: 'AR APDS2; p85α lost → p110δ disinhibited; identical to APDS1 clinically' },
  { gene: 'LRBA',   inh: 'AR',      b_absent: '−',   agamma: '✓✓',  sinopulm: '✓✓',  ebv_cmv: '±',    autoimmune: '✓✓✓', lymphoprolif: '✓✓✓',gi: '✓✓',   echovirus: '−',   unique: 'AUTOIMMUNITY DOMINANT (ITP+IBD+hepatitis); Abatacept HIGHLY EFFECTIVE' },
  { gene: 'NFKB1',  inh: 'AD-LOF',  b_absent: '−',   agamma: '✓✓',  sinopulm: '✓✓✓', ebv_cmv: '−',    autoimmune: '✓✓',  lymphoprolif: '✓✓✓',gi: '✓✓',   echovirus: '−',   unique: 'Most common genetic CVID 10%; SPLENOMEGALY+IBD DISTINGUISHERS; adult onset' },
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
      {cohort && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
          <Badge text={`n=${cohort.n}`} color={color} />
          <Badge text={`Onset ${cohort.mean_onset_months?.toFixed(0)}mo`} color='#555' />
          <Badge text={`Dx lag ${cohort.mean_dx_lag_yrs?.toFixed(1)}y`} color='#888' />
          <Badge text={`B-absent ${cohort.b_cells_absent_n}/${cohort.n}`} color='#1565c0' />
          <Badge text={`Agamma ${cohort.agammaglobulinemia_n}/${cohort.n}`} color='#283593' />
          <Badge text={`SinoPulm ${cohort.sinopulmonary_n}/${cohort.n}`} color='#2e7d32' />
          <Badge text={`EBV/CMV ${cohort.enteroviral_n}/${cohort.n}`} color='#6a1b9a' />
          <Badge text={`Autoimmune ${cohort.autoimmunity_n}/${cohort.n}`} color='#c62828' />
          <Badge text={`Lymphoprolif ${cohort.lymphoproliferation_n}/${cohort.n}`} color='#4e342e' />
          <Badge text={`Bronchiect ${cohort.bronchiectasis_n}/${cohort.n}`} color='#00695c' />
          <Badge text={`IVIG resp ${cohort.ivig_responsive_n}/${cohort.n}`} color='#1a237e' />
        </div>
      )}
      {cohort?.pathognomonic && (
        <div style={{ background: `${color}15`, borderLeft: `3px solid ${color}`, padding: '6px 10px', fontSize: 12, borderRadius: '0 4px 4px 0' }}>
          <strong>Pathognomonic / Distinguisher:</strong> {cohort.pathognomonic}
        </div>
      )}
    </div>
  );
}

export default function HeredAgammaCVIDAtlasPage() {
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
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#666' }}>Loading Hereditary-Agammaglobulinemia-CVID-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  const cohortMap = {};
  (overview?.cohorts || []).forEach(c => { cohortMap[c.gene] = c; });

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 16px', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#0d47a1,#1565c0)', color: '#fff', borderRadius: 12, padding: '28px 32px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700 }}>&#x1f9ec; Hereditary-Agammaglobulinemia-CVID-Atlas</h1>
        <div style={{ opacity: 0.9, marginTop: 6, fontSize: 14 }}>
          B-Cell &amp; Antibody Deficiency Primary Immunodeficiencies — Complete 8-Gene Reference
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {(overview?.genes || []).map(g => (
            <span key={g} style={{ background: GENE_COLORS[g] || '#555', color: '#fff', padding: '3px 10px', borderRadius: 4, fontSize: 12, fontWeight: 600 }}>{g}</span>
          ))}
        </div>
        <div style={{ marginTop: 8, fontSize: 13, opacity: 0.85 }}>
          {overview?.total_patients} patients · 8 genes · seeds {overview?.seeds}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
            background: tab === i ? '#0d47a1' : '#e3f2fd', color: tab === i ? '#fff' : '#0d47a1',
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 0 && (
        <div>
          {/* Key Facts */}
          <h2 style={{ fontSize: 18, marginBottom: 12 }}>Key Clinical Facts</h2>
          {overview?.key_facts && Object.entries(overview.key_facts).map(([k, v]) => (
            <div key={k} style={{ background: '#e3f2fd', borderLeft: '4px solid #1565c0', padding: '10px 14px', marginBottom: 10, borderRadius: '0 6px 6px 0' }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#0d47a1', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 13 }}>{v}</div>
            </div>
          ))}

          {/* Diagnostic Approach */}
          <h2 style={{ fontSize: 18, marginBottom: 12, marginTop: 24 }}>Diagnostic Approach</h2>
          {overview?.diagnostic_approach && Object.entries(overview.diagnostic_approach).map(([k, v]) => (
            <div key={k} style={{ background: '#f3e5f5', borderLeft: '4px solid #7b1fa2', padding: '10px 14px', marginBottom: 10, borderRadius: '0 6px 6px 0' }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#6a1b9a', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 13 }}>{v}</div>
            </div>
          ))}

          {/* Gene Cards */}
          <h2 style={{ fontSize: 18, marginBottom: 12, marginTop: 24 }}>Gene Cohort Summaries (n=40 each)</h2>
          {(overview?.genes || []).map(g => (
            <GeneCard key={g} gene={g} color={GENE_COLORS[g] || '#333'} info={GENE_INFO[g]} cohort={cohortMap[g]} />
          ))}
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 1 && (
        <div>
          <h2 style={{ fontSize: 18, marginBottom: 12 }}>Comparative Phenotype Matrix</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#0d47a1', color: '#fff' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Gene</th>
                  <th style={{ padding: '8px 6px' }}>Inh</th>
                  <th style={{ padding: '8px 6px' }}>B-Absent</th>
                  <th style={{ padding: '8px 6px' }}>Agamma</th>
                  <th style={{ padding: '8px 6px' }}>SinoPulm</th>
                  <th style={{ padding: '8px 6px' }}>EBV/CMV</th>
                  <th style={{ padding: '8px 6px' }}>Autoimmune</th>
                  <th style={{ padding: '8px 6px' }}>LymphoProlif</th>
                  <th style={{ padding: '8px 6px' }}>GI</th>
                  <th style={{ padding: '8px 6px' }}>Echovirus</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Unique Distinguisher</th>
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((r, i) => (
                  <tr key={r.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f5f5f5', borderBottom: '1px solid #ddd' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[r.gene] }}>{r.gene}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}><code style={{ fontSize: 11 }}>{r.inh}</code></td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.b_absent}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.agamma}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.sinopulm}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.ebv_cmv}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.autoimmune}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.lymphoprolif}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.gi}</td>
                    <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.echovirus}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#444' }}>{r.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 style={{ fontSize: 18, marginBottom: 12, marginTop: 28 }}>Full Gene Registry (n=40 each)</h2>
          {(breakdown?.genes || []).map((g, i) => (
            <div key={g.gene} style={{ border: `2px solid ${GENE_COLORS[g.gene] || '#333'}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <div style={{ background: GENE_COLORS[g.gene], color: '#fff', padding: '4px 12px', borderRadius: 4, fontWeight: 700 }}>#{i+1} {g.gene}</div>
                <div style={{ fontSize: 12, color: '#666' }}>{g.locus} · {g.aa} aa / {g.mw_kda} kDa · {g.inheritance}-{g.mechanism}</div>
              </div>
              <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}><strong>Syndrome:</strong> {g.syndrome}</div>
              <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}><strong>Distinguishing:</strong> {g.distinguishing_feature}</div>
              <div style={{ fontSize: 12, color: '#333', marginBottom: 8 }}><strong>Treatment:</strong> {g.first_line_treatment}</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {(g.critical_rules || []).map(r => <Badge key={r} text={r} color={GENE_COLORS[g.gene]} />)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* CVID Classification Map Tab */}
      {tab === 2 && (
        <div>
          <h2 style={{ fontSize: 18, marginBottom: 16 }}>B-Cell / Antibody Deficiency Classification Map</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px,1fr))', gap: 16 }}>
            {/* Agammaglobulinemia group */}
            <div style={{ border: '2px solid #0d47a1', borderRadius: 10, padding: 16 }}>
              <div style={{ background: '#0d47a1', color: '#fff', padding: '6px 12px', borderRadius: 4, fontWeight: 700, marginBottom: 12 }}>
                TRUE AGAMMAGLOBULINEMIA — B Cells Absent
              </div>
              <div style={{ fontSize: 12, color: '#444', marginBottom: 10 }}>
                Pro-B cell arrest · All Ig undetectable · Onset 6-9 months · Echovirus risk
              </div>
              {['BTK','IGHM','IGLL1','BLNK'].map(g => (
                <div key={g} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, padding: '6px 10px', background: '#e3f2fd', borderRadius: 4 }}>
                  <div style={{ width: 48, fontWeight: 700, color: GENE_COLORS[g] }}>{g}</div>
                  <div style={{ fontSize: 11 }}>{GENE_INFO[g]?.inh} · {GENE_INFO[g]?.locus} · {GENE_INFO[g]?.size}</div>
                </div>
              ))}
            </div>

            {/* APDS group */}
            <div style={{ border: '2px solid #e65100', borderRadius: 10, padding: 16 }}>
              <div style={{ background: '#e65100', color: '#fff', padding: '6px 12px', borderRadius: 4, fontWeight: 700, marginBottom: 12 }}>
                APDS — Activated PI3K-delta · B Cells Present
              </div>
              <div style={{ fontSize: 12, color: '#444', marginBottom: 10 }}>
                PI3Kδ constitutive · EBV/CMV · Lymphoproliferation · Leniolisib FDA2023
              </div>
              {['PIK3CD','PIK3R1'].map(g => (
                <div key={g} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, padding: '6px 10px', background: '#fff3e0', borderRadius: 4 }}>
                  <div style={{ width: 64, fontWeight: 700, color: GENE_COLORS[g] }}>{g}</div>
                  <div style={{ fontSize: 11 }}>{GENE_INFO[g]?.inh} · {GENE_INFO[g]?.locus} · {GENE_INFO[g]?.size}</div>
                </div>
              ))}
            </div>

            {/* Immune Dysregulation group */}
            <div style={{ border: '2px solid #004d40', borderRadius: 10, padding: 16 }}>
              <div style={{ background: '#004d40', color: '#fff', padding: '6px 12px', borderRadius: 4, fontWeight: 700, marginBottom: 12 }}>
                IMMUNE DYSREGULATION + Antibody Deficiency
              </div>
              <div style={{ fontSize: 12, color: '#444', marginBottom: 10 }}>
                CTLA4/NF-κB pathway · Autoimmunity + hypogammaglobulinaemia · Lymphoproliferation
              </div>
              {['LRBA','NFKB1'].map(g => (
                <div key={g} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, padding: '6px 10px', background: '#e0f2f1', borderRadius: 4 }}>
                  <div style={{ width: 64, fontWeight: 700, color: GENE_COLORS[g] }}>{g}</div>
                  <div style={{ fontSize: 11 }}>{GENE_INFO[g]?.inh} · {GENE_INFO[g]?.locus} · {GENE_INFO[g]?.size}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Key Decision Points */}
          <h2 style={{ fontSize: 18, marginBottom: 12, marginTop: 28 }}>Key Diagnostic Decision Points</h2>
          {[
            { q: 'Are B cells absent? (<1% CD19+ of lymphocytes)', a: 'YES → True agammaglobulinemia: BTK first (if MALE); IGHM (if FEMALE or BTK negative); then IGLL1, BLNK. NO → CVID-spectrum: PIK3CD, PIK3R1, LRBA, NFKB1.' },
            { q: 'Is the patient MALE or FEMALE?', a: 'MALE with absent B cells → BTK (X-linked, 85% of agammaglobulinemia). FEMALE with absent B cells → IGHM first (AR, most common AR agammaglobulinemia); BTK-XLA cannot cause disease in females.' },
            { q: 'EBV/CMV susceptibility + lymphoproliferation present?', a: 'YES → APDS (PIK3CD APDS1 AD-GOF, or PIK3R1 APDS2 AR-LOF). B cells will be PRESENT in APDS. Leniolisib (FDA2023) is the precision therapy. Sequence PIK3CD (AD hotspots E1021K, E525K) first.' },
            { q: 'Multi-organ autoimmunity DOMINANT (ITP + IBD + hepatitis + ILD)?', a: 'YES → LRBA deficiency. Check CTLA4 surface expression on stimulated T cells (reduced). Also check CTLA4 gene itself. Abatacept (CTLA4-Ig) is HIGHLY EFFECTIVE. Also: PI3K disorders (PIK3CD/PIK3R1) have autoimmunity but less dominant.' },
            { q: 'SPLENOMEGALY + IBD-like colitis + adult-onset CVID + family history?', a: 'YES → NFKB1 haploinsufficiency (CVID Type 7 — most common defined genetic CVID at ~10%). AD with variable expressivity. Cascade family testing important. Anti-TNF/anti-integrin for IBD; IVIG for antibody deficiency.' },
            { q: 'Echovirus / enterovirus neurological symptoms in agammaglobulinemia?', a: 'URGENT: CSF PCR + stool PCR for enteroviruses. High-dose IVIG (consider ≥1000 mg/dL trough). Intraventricular IVIG for CNS infection. This is the PATHOGNOMONIC dangerous complication of BTK-XLA (and other agammaglobulinemias) — screen proactively in all neurological symptoms.' },
          ].map((item, i) => (
            <div key={i} style={{ background: '#f5f5f5', borderRadius: 8, padding: 14, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, color: '#0d47a1', marginBottom: 6, fontSize: 13 }}>Q: {item.q}</div>
              <div style={{ fontSize: 13, color: '#333' }}>→ {item.a}</div>
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 3 && (
        <div>
          <h2 style={{ fontSize: 18, marginBottom: 16 }}>Definitions &amp; Key Concepts</h2>
          {defs?.definitions && Object.entries(defs.definitions).map(([term, text]) => (
            <div key={term} style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#0d47a1', marginBottom: 6 }}>{term}</h3>
              <div style={{ fontSize: 13, lineHeight: 1.7, color: '#333', background: '#f8f9fa', padding: 12, borderRadius: 6 }}>{text}</div>
            </div>
          ))}

          <h2 style={{ fontSize: 18, marginBottom: 12, marginTop: 28 }}>Standards &amp; Key References</h2>
          {(defs?.standards || []).map((s, i) => (
            <div key={i} style={{ fontSize: 13, padding: '6px 0', borderBottom: '1px solid #eee', color: '#555' }}>• {s}</div>
          ))}
        </div>
      )}
    </div>
  );
}
