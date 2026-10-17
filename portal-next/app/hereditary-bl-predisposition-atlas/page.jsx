'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-bl-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':       '#006064',  // dark cyan          — biallelic A-T BL 5-15x; cranial RT CI; ceralasertib ATRi
  'TP53':      '#bf360c',  // deep orange-red     — LFS BL 2-4x; AVOID RADIATION ABSOLUTELY; venetoclax backbone
  'BRCA2':     '#1b5e20',  // forest green        — FA-D1 childhood BL PATHOGNOMONIC; AVOID ALKYLATING; olaparib
  'PIK3CD':    '#4a148c',  // deep purple         — APDS1; EBV-driven BL 10-15%; leniolisib FDA2023; live vaccines CI
  'CARD11':    '#0d47a1',  // deep blue           — BENTA GOF venetoclax; CID LOF HSCT urgent; MALT1 inhibitor
  'TNFRSF13B': '#880e4f',  // deep pink           — CVID2; NHL 10-12x HIGHEST; IVIG NEVER STOP; C104R 1 in 500
  'CD27':      '#e65100',  // deep orange         — CD27 deficiency; EBV-BL 30-50x HIGHEST; HSCT MANDATORY
  'NF1':       '#33691e',  // olive green         — NF1; BL 2-3x; cafe-au-lait PATHOGNOMONIC; selumetinib FDA2020
};

const GENE_INFO = {
  'ATM':       { full: 'RADIOSENSITIVITY-ABSOLUTE / biallelic-A-T-BL-5-15x / monoallelic-BL-2-3x / cranial-RT-CI / ceralasertib-ATRi / intrathecal-NOT-whole-brain-RT / EBV-Q3M', locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF (monoallelic); AR biallelic = A-T' },
  'TP53':      { full: 'LFS / BL-2-4x / AVOID-RADIATION-ABSOLUTELY / WB-MRI-Toronto-annually / venetoclax-backbone / APR-246-investigational / 17p-del-20-30pct', locus: '17p13.1', size: '393 aa / 43 kDa', inh: 'AD LOF heterozygous' },
  'BRCA2':     { full: 'FA-D1-biallelic-childhood-BL-PATHOGNOMONIC / AVOID-ALKYLATING-ABSOLUTELY / MMC-fragility-MANDATORY / sibling-donors-EXCLUDE-FA / olaparib-monoallelic / cisplatin-HRD-60pct', locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF monoallelic HBOC; AR biallelic = FA-D1' },
  'PIK3CD':    { full: 'APDS1 / PI3Kdelta-GOF-E1021K / EBV-driven-BL-10-15pct / leniolisib-FDA2023 / acyclovir-MANDATORY / sirolimus-bridge / live-vaccines-ABSOLUTELY-CI / EBV-CMV-Q3M', locus: '1p36.22', size: '1044 aa / 119 kDa', inh: 'AD GOF heterozygous' },
  'CARD11':    { full: 'BENTA-GOF-venetoclax-preferred / MALT1-inhibitor-JNJ-67856633 / CID-LOF-HSCT-URGENT-MANDATORY / NF-kB-dysregulation-BL / rapamycin-adjunct-BENTA', locus: '7p22.2', size: '1154 aa / 133 kDa', inh: 'AD GOF (BENTA); AD LOF (CID)' },
  'TNFRSF13B': { full: 'CVID2 / NHL-10-12x-HIGHEST / IVIG-MANDATORY-NEVER-STOP-IgG-7gL / rituximab-IVIG-CO-MANDATORY / C104R-European-1in500 / EBV-Q3M / BL-via-humoral-immunity-impaired', locus: '17p11.2', size: '293 aa / 31 kDa', inh: 'AD LOF heterozygous (CVID2)' },
  'CD27':      { full: 'CD27-Deficiency / EBV-driven-B-cell-LPD-BL-30-50x-HIGHEST / HLH-trigger / HSCT-MANDATORY / EBV-Q3M-MANDATORY / rituximab-IVIG / HLH-2004-etoposide', locus: '12p13.31', size: '260 aa / 29 kDa', inh: 'AR biallelic LOF' },
  'NF1':       { full: 'NF1 / NHL-BL-2-3x / cafe-au-lait-PATHOGNOMONIC-6-spots / Lisch-nodules / MPNST-8-13pct / selumetinib-FDA2020 / RT-NOT-CI-unlike-ATM-TP53 / RAS-MAPK', locus: '17q11.2', size: '2839 aa / 327 kDa', inh: 'AD LOF heterozygous' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}18`, color, border: `1px solid ${color}55`,
      borderRadius: 4, fontSize: 10, padding: '2px 7px', marginLeft: 6, fontWeight: 600,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 10,
      padding: '14px 18px', minWidth: 130, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#aaa' }}>{sub}</div>
    </div>
  );
}

export default function HereditaryBLPredispositionAtlasPage() {
  const [tab, setTab]                  = useState('Overview');
  const [overview, setOverview]        = useState(null);
  const [breakdown, setBreakdown]      = useState(null);
  const [definitions, setDefinitions]  = useState(null);
  const [loading, setLoading]          = useState(false);
  const [error, setError]              = useState('');
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true); setError('');
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '8px 18px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600,
    fontSize: 13, background: tab === t ? '#006064' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary BL Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#006064 0%,#bf360c 100%)', borderRadius: 12,
        padding: '22px 28px', marginBottom: 22, color: '#fff' }}>
        <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 4 }}>🧬 Hereditary Predisposition Atlas Series · seeds 3630–3637</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>
          Hereditary Burkitt Lymphoma (BL) Predisposition Atlas
        </h1>
        <div style={{ fontSize: 13, opacity: 0.85, marginTop: 6 }}>
          Complete 8-Gene Reference · ATM · TP53 · BRCA2 · PIK3CD · CARD11 · TNFRSF13B · CD27 · NF1
        </div>
        <div style={{ fontSize: 11, opacity: 0.75, marginTop: 4 }}>
          320-Patient Aggregate (8 × 40) · Seeds 3630–3637 · Germline BL Predisposition Reference
        </div>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 20, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="Total Patients" value={overview.total_patients} sub="8 × 40 cohorts" color="#006064" />
            <StatCard label="CR Rate" value={`${overview.cr_pct}%`} sub={`${overview.cr_n} / ${overview.total_patients}`} color="#2e7d32" />
            <StatCard label="2yr OS" value={`${overview.os2_pct}%`} sub={`${overview.os2_n} / ${overview.total_patients}`} color="#1565c0" />
            <StatCard label="EBV+" value={`${overview.ebv_pos_pct}%`} sub={`${overview.ebv_pos_n} EBV-positive`} color="#6a1b9a" />
            <StatCard label="CNS Involvement" value={`${overview.cns_inv_pct}%`} sub={`${overview.cns_inv_n} CNS+`} color="#bf360c" />
            <StatCard label="HSCT" value={`${overview.hsct_pct}%`} sub={`${overview.hsct_n} transplanted`} color="#e65100" />
          </div>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="Immunodeficiency BL" value={`${overview.immunodef_pct}%`} sub={`${overview.immunodef_n} ID-BL`} color="#880e4f" />
            <StatCard label="HLH Trigger" value={`${overview.hlh_pct}%`} sub={`${overview.hlh_n} HLH events`} color="#0d47a1" />
            <StatCard label="R-CODOX-M/IVAC" value={`${overview.rcodox_pct}%`} sub={`${overview.rcodox_n} treated`} color="#33691e" />
            <StatCard label="Venetoclax" value={`${overview.venetoclax_pct}%`} sub={`${overview.venetoclax_n} venetoclax`} color="#004d40" />
            <StatCard label="Targeted Rx" value={`${overview.targeted_pct}%`} sub={`${overview.targeted_n} targeted`} color="#c62828" />
          </div>

          {/* Gene summary cards */}
          <h3 style={{ fontSize: 16, fontWeight: 700, color: '#333', marginBottom: 12 }}>Per-Gene BL Predisposition Summary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 12 }}>
            {Object.entries(GENE_INFO).map(([gene, info]) => {
              const gc = overview.gene_counts?.[gene] || {};
              const col = GENE_COLORS[gene] || '#555';
              return (
                <div key={gene} style={{ background: '#fff', border: `2px solid ${col}33`, borderRadius: 10, padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 800, fontSize: 15, color: col }}>{gene}</span>
                    <Badge text={info.inh.split(';')[0].trim()} color={col} />
                    <span style={{ marginLeft: 'auto', fontSize: 10, color: '#aaa' }}>{info.locus}</span>
                  </div>
                  <div style={{ fontSize: 10, color: '#777', marginBottom: 8, lineHeight: 1.5 }}>{info.full}</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11 }}>
                    <span style={{ color: '#2e7d32' }}>CR: {gc.cr_n ?? '–'}/40</span>
                    <span style={{ color: '#1565c0' }}>2yr-OS: {gc.os2_n ?? '–'}/40</span>
                    <span style={{ color: '#6a1b9a' }}>EBV+: {gc.ebv_pos_n ?? '–'}</span>
                    <span style={{ color: '#bf360c' }}>CNS: {gc.cns_inv_n ?? '–'}</span>
                    <span style={{ color: '#e65100' }}>HSCT: {gc.hsct_n ?? '–'}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <h3 style={{ fontSize: 16, fontWeight: 700, color: '#333', marginBottom: 14 }}>8-Gene BL Predisposition Reference Table</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  {['Gene','Locus','Inheritance','Protein','BL Risk','CR/40','2yr-OS/40','EBV+','CNS','HSCT','HLH','Key Rule'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 700, color: '#555', borderBottom: '2px solid #ddd', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ATLAS_GENES_CLIENT.map((g, i) => {
                  const d = breakdown.gene_data?.[g.gene] || {};
                  const col = GENE_COLORS[g.gene] || '#555';
                  return (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f9f9f9', cursor: 'pointer' }}
                        onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}>
                      <td style={{ padding: '8px 10px', fontWeight: 800, color: col }}>{g.gene}</td>
                      <td style={{ padding: '8px 10px', color: '#555' }}>{g.locus}</td>
                      <td style={{ padding: '8px 10px', fontSize: 10, color: '#777', maxWidth: 120 }}>{g.inh}</td>
                      <td style={{ padding: '8px 10px', fontSize: 10, color: '#777' }}>{g.size}</td>
                      <td style={{ padding: '8px 10px', fontSize: 10, color: col, maxWidth: 180 }}>{g.blRisk}</td>
                      <td style={{ padding: '8px 10px', fontWeight: 700, color: '#2e7d32' }}>{d.cr_n ?? '–'}</td>
                      <td style={{ padding: '8px 10px', fontWeight: 700, color: '#1565c0' }}>{d.os2_n ?? '–'}</td>
                      <td style={{ padding: '8px 10px', color: '#6a1b9a' }}>{d.ebv_pos_n ?? '–'}</td>
                      <td style={{ padding: '8px 10px', color: '#bf360c' }}>{d.cns_inv_n ?? '–'}</td>
                      <td style={{ padding: '8px 10px', color: '#e65100' }}>{d.hsct_n ?? '–'}</td>
                      <td style={{ padding: '8px 10px', color: '#880e4f' }}>{d.hlh_n ?? '–'}</td>
                      <td style={{ padding: '8px 10px', fontSize: 10, color: '#555', maxWidth: 200 }}>{g.keyRule}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {expandedGene && breakdown.gene_data?.[expandedGene] && (
            <div style={{ marginTop: 16, background: '#f5f5f5', borderRadius: 10, padding: 16 }}>
              <h4 style={{ margin: '0 0 8px', color: GENE_COLORS[expandedGene] }}>{expandedGene} — Expanded Detail</h4>
              <pre style={{ fontSize: 11, color: '#333', whiteSpace: 'pre-wrap', maxHeight: 300, overflow: 'auto' }}>
                {JSON.stringify(breakdown.gene_data[expandedGene], null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <h3 style={{ fontSize: 16, fontWeight: 700, color: '#333', marginBottom: 14 }}>Clinical Rules · Key Avoids · Targeted Rx</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(500px,1fr))', gap: 14 }}>
            {ATLAS_GENES_CLIENT.map(g => {
              const d = breakdown.gene_data?.[g.gene] || {};
              const col = GENE_COLORS[g.gene] || '#555';
              return (
                <div key={g.gene} style={{ background: '#fff', border: `2px solid ${col}33`, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontWeight: 800, fontSize: 15, color: col, marginBottom: 6 }}>
                    {g.gene} <span style={{ fontSize: 11, color: '#aaa', fontWeight: 400 }}>{g.locus} · {g.syndrome}</span>
                  </div>
                  <div style={{ fontSize: 11, marginBottom: 6 }}>
                    <span style={{ fontWeight: 700, color: '#555' }}>BL Risk: </span>
                    <span style={{ color: '#666' }}>{d.bl_risk || g.blRisk}</span>
                  </div>
                  <div style={{ fontSize: 11, marginBottom: 6 }}>
                    <span style={{ fontWeight: 700, color: '#c62828' }}>⚠ Key Avoid: </span>
                    <span style={{ color: '#c62828' }}>{d.key_avoid || g.keyAvoid}</span>
                  </div>
                  <div style={{ fontSize: 11, marginBottom: 6 }}>
                    <span style={{ fontWeight: 700, color: '#2e7d32' }}>✔ Key Rule: </span>
                    <span style={{ color: '#2e7d32' }}>{d.key_rule || g.keyRule}</span>
                  </div>
                  <div style={{ fontSize: 11, marginBottom: 6 }}>
                    <span style={{ fontWeight: 700, color: '#1565c0' }}>💊 Targeted Rx: </span>
                    <span style={{ color: '#555' }}>{d.targeted_rx || g.targetedRx}</span>
                  </div>
                  <div style={{ fontSize: 11 }}>
                    <span style={{ fontWeight: 700, color: '#555' }}>🔬 Surveillance: </span>
                    <span style={{ color: '#777' }}>{d.surveillance || g.surveillance}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ fontSize: 16, fontWeight: 700, color: '#333', marginBottom: 14 }}>Clinical Definitions & Key Distinctions</h3>
          {definitions.key_clinical_distinctions && (
            <div style={{ background: '#fff8e1', border: '2px solid #f9a825', borderRadius: 10, padding: 16, marginBottom: 18 }}>
              <div style={{ fontWeight: 700, color: '#f57f17', marginBottom: 8 }}>⚡ Key Clinical Distinctions — BL</div>
              {definitions.key_clinical_distinctions.map((d, i) => (
                <div key={i} style={{ fontSize: 11, color: '#555', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #f9a825' }}>{d}</div>
              ))}
            </div>
          )}
          <div>
            {(definitions.definitions || []).map((d, i) => (
              <div key={i} style={{ marginBottom: 10, background: '#fff', borderRadius: 8, border: '1px solid #e0e0e0' }}>
                <div style={{ padding: '10px 14px', fontWeight: 700, fontSize: 13, color: '#333', cursor: 'pointer',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                  onClick={() => setExpandedDef(expandedDef === i ? null : i)}>
                  {d.term}
                  <span style={{ color: '#aaa', fontSize: 16 }}>{expandedDef === i ? '▲' : '▼'}</span>
                </div>
                {expandedDef === i && (
                  <div style={{ padding: '0 14px 12px', fontSize: 11, color: '#555', whiteSpace: 'pre-wrap', borderTop: '1px solid #f0f0f0', paddingTop: 10 }}>
                    {d.definition}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Client-side static gene data (mirrors backend ATLAS_GENES, no import needed)
const ATLAS_GENES_CLIENT = [
  { gene: 'ATM',       locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF monoallelic; AR biallelic = A-T',         syndrome: 'Ataxia-Telangiectasia / ATM-BL',           blRisk: 'Biallelic A-T BL 5-15x HIGHEST; monoallelic BL 2-3x', keyRule: 'RADIOSENSITIVITY ABSOLUTE — cranial RT CI; ceralasertib ATRi; intrathecal NOT whole-brain RT', keyAvoid: 'High-dose cranial RT (ABSOLUTELY CI); unmonitored cyclophosphamide', surveillance: 'Annual CBC + lymph node; EBV Q3M; IgA baseline; biallelic: neuro + pulmonary', targetedRx: 'Ceralasertib ATRi; R-CODOX-M/IVAC RT dose-reduction; rituximab mandatory' },
  { gene: 'TP53',      locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF heterozygous',                             syndrome: 'Li-Fraumeni Syndrome / TP53-BL',           blRisk: 'LFS BL 2-4x; 17p del 20-30%; MYC+TP53 co-loss aggressive', keyRule: 'AVOID RADIATION ABSOLUTELY; WB-MRI Toronto annually; venetoclax backbone', keyAvoid: 'ANY radiation field (ABSOLUTELY CI); intrathecal OK NOT whole-brain RT', surveillance: 'WB-MRI annually Toronto; annual CBC; first-degree family TP53 testing', targetedRx: 'Venetoclax BCL-2i backbone; R-CODOX-M/IVAC ± venetoclax; APR-246 investigational' },
  { gene: 'BRCA2',     locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF monoallelic HBOC; AR biallelic = FA-D1',    syndrome: 'HBOC / Fanconi D1 (FA-D1) / BRCA2-BL',    blRisk: 'FA-D1 biallelic: childhood BL PATHOGNOMONIC; monoallelic HBOC: BL 2-3x; HRD 60%', keyRule: 'BIALLELIC: AVOID ALKYLATING ABSOLUTELY — MMC fragility MANDATORY; sibling donors EXCLUDE FA testing', keyAvoid: 'Cyclophosphamide/alkylating FA-D1 biallelic (ABSOLUTELY CI); sibling donor without FA screen', surveillance: 'MMC fragility test before treatment FA-D1; annual breast MRI monoallelic; CBC from infancy FA-D1', targetedRx: 'Olaparib PARP inhibitor monoallelic; cisplatin-preferred HRD-60pct; FA-D1 HSCT FA-negative sibling' },
  { gene: 'PIK3CD',    locus: '1p36.22', size: '1044 aa / 119 kDa', inh: 'AD GOF heterozygous',                             syndrome: 'APDS1 / PASLI / PIK3CD-BL',               blRisk: 'EBV-driven BL 10-15%; APDS1 nodular lymphoid hyperplasia → EBV+BL', keyRule: 'Leniolisib FDA2023 first targeted therapy; acyclovir MANDATORY prophylaxis; EBV/CMV Q3M; live vaccines CI', keyAvoid: 'Live vaccines ABSOLUTELY CI; delay leniolisib without sirolimus bridge', surveillance: 'EBV/CMV Q3M MANDATORY; CBC quarterly; IgG q3M; annual CT; cascade PIK3CD testing', targetedRx: 'Leniolisib PI3Kδi FDA2023; sirolimus mTORi bridge; acyclovir MANDATORY; IVIG replacement' },
  { gene: 'CARD11',    locus: '7p22.2',  size: '1154 aa / 133 kDa', inh: 'AD GOF (BENTA); AD LOF (CID)',                    syndrome: 'BENTA Disease (GOF) / CID (LOF) / CARD11-BL', blRisk: 'BENTA GOF: B-cell lymphoma/BL constitutive NF-kB; CID LOF: lymphoma via immunodeficiency', keyRule: 'GOF BENTA: venetoclax preferred + MALT1 inhibitor; LOF CID: HSCT URGENT MANDATORY before decompensation', keyAvoid: 'Delaying HSCT in CID-LOF until lymphoma occurs; rituximab monotherapy without venetoclax BENTA', surveillance: 'CBC 6-monthly; lymphocyte subsets annually; annual CT BENTA; HSCT evaluation CID-LOF diagnosis', targetedRx: 'Venetoclax BCL-2i BENTA GOF; MALT1 inhibitor JNJ-67856633; rapamycin adjunct; CID-LOF HSCT urgent' },
  { gene: 'TNFRSF13B', locus: '17p11.2', size: '293 aa / 31 kDa',   inh: 'AD LOF heterozygous (CVID2)',                    syndrome: 'CVID type 2 (CVID2) / TACI Deficiency / TNFRSF13B-BL', blRisk: 'NHL 10-12x HIGHEST; BL via EBV IVIG-insufficient CVID2; C104R European 1 in 500', keyRule: 'IVIG MANDATORY NEVER STOP (IgG ≥7g/L trough); rituximab + IVIG CO-MANDATORY; EBV Q3M surveillance', keyAvoid: 'Stopping IVIG during rituximab therapy (IgG falls → fatal hypogammaglobulinaemia)', surveillance: 'IgG trough monthly IVIG; EBV Q3M; annual CT lymph nodes; PFT annual; C104R cascade family', targetedRx: 'IVIG never stop; rituximab + IVIG co-mandatory NHL; R-CODOX-M/IVAC + IVIG active BL' },
  { gene: 'CD27',      locus: '12p13.31', size: '260 aa / 29 kDa',  inh: 'AR biallelic LOF',                               syndrome: 'CD27 Deficiency / TNFRSF7 Deficiency / CD27-BL', blRisk: 'EBV-driven B-cell LPD/BL 30-50x HIGHEST; HLH trigger; hypogammaglobulinaemia; lymphoma age 5-20yr', keyRule: 'HSCT MANDATORY curative; EBV viral load Q3M MANDATORY; rituximab+IVIG; HLH-2004 etoposide if triggered', keyAvoid: 'Delaying HSCT after B-cell LPD established; missing HLH trigger requiring etoposide', surveillance: 'EBV viral load Q3M MANDATORY; CBC quarterly; IgG q3M; annual imaging; HSCT evaluation at diagnosis', targetedRx: 'HSCT mandatory curative; rituximab+IVIG EBV+BL; HLH-2004 etoposide+dex; IVIG replacement hypogamma' },
  { gene: 'NF1',       locus: '17q11.2', size: '2839 aa / 327 kDa', inh: 'AD LOF heterozygous',                            syndrome: 'Neurofibromatosis Type 1 (NF1) / NF1-BL',  blRisk: 'NHL/BL 2-3x; RAS-MAPK lymphoproliferation; MPNST 8-13%; MYC translocation susceptibility', keyRule: 'Selumetinib MEKi FDA2020 plexiform neurofibromas; standard R-CODOX-M/IVAC for NF1-BL; RT NOT CI unlike ATM/TP53', keyAvoid: 'Confusing MPNST AVOID with BL AVOID RT — NF1 RT NOT CI (unlike ATM/TP53); delaying selumetinib plexiform', surveillance: 'Annual neuro + MRI spine/brain; annual CBC; annual skin exam; BP monitoring; cascade NF1 family', targetedRx: 'Selumetinib MEKi FDA2020 plexiform NF; standard R-CODOX-M/IVAC BL; MPNST surgery + chemoRT' },
];
