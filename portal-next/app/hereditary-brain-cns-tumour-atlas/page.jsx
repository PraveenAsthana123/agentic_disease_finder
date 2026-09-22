'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-brain-cns-tumour-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TP53':    '#b71c1c',  // deep red       — LFS; CPC PATHOGNOMONIC; AVOID RADIATION ABSOLUTELY
  'PTCH1':   '#e65100',  // deep orange    — Gorlin; desmoplastic MB; calcified falx; AVOID RADIATION; Vismodegib
  'SUFU':    '#f57f17',  // amber-orange   — BCNS2; infant MB highest penetrance; MBEN PATHOGNOMONIC
  'PTEN':    '#2e7d32',  // forest green   — Cowden/PHTS; Lhermitte-Duclos PATHOGNOMONIC; macrocephaly
  'APC':     '#1565c0',  // deep blue      — FAP/Turcot2; WNT-MB best prognosis; CHRPE PATHOGNOMONIC
  'VHL':     '#6a1b9a',  // deep purple    — VHL disease; CNS hemangioblastoma PATHOGNOMONIC; Belzutifan
  'SMARCB1': '#4a148c',  // dark violet    — RTPS2; AT/RT <3yr PATHOGNOMONIC; INI1 IHC loss; tazemetostat
  'PMS2':    '#004d40',  // dark teal      — CMMRD; GBM TMB>100 PATHOGNOMONIC; CALMs; PD-1 immunotherapy
};

const GENE_INFO = {
  'TP53':    { full: 'LFS / CPC Age <5yr PATHOGNOMONIC / GBM / MB / AVOID RADIATION ABSOLUTELY / WBMRI Toronto',         locus: '17p13.1',  size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'PTCH1':   { full: 'Gorlin/NBCCS / Desmoplastic MB SHH / Calcified Falx PATHOGNOMONIC / BCCs / AVOID RADIATION',      locus: '9q22.32',  size: '1447 aa / 160 kDa', inh: 'AD LOF' },
  'SUFU':    { full: 'BCNS Type 2 / Infant MB 15-20% / MBEN PATHOGNOMONIC / Fewer BCCs / AVOID RADIATION',              locus: '10q24.32', size: '484 aa / 60 kDa',   inh: 'AD LOF' },
  'PTEN':    { full: 'Cowden/PHTS / Lhermitte-Duclos Cerebellar Gangliocytoma PATHOGNOMONIC / Macrocephaly / Everolimus', locus: '10q23.31', size: '403 aa / 47 kDa',   inh: 'AD LOF' },
  'APC':     { full: 'FAP/Turcot Type 2 / WNT-MB Best Prognosis / CHRPE PATHOGNOMONIC / Gardner / Desmoid',             locus: '5q22.2',   size: '2843 aa / 310 kDa', inh: 'AD LOF' },
  'VHL':     { full: 'VHL Disease / CNS Hemangioblastoma PATHOGNOMONIC / Retinal Annual Age 1 / Belzutifan FDA 2021',    locus: '3p25.3',   size: '213 aa / 24 kDa',   inh: 'AD LOF' },
  'SMARCB1': { full: 'RTPS2 / AT/RT <3yr PATHOGNOMONIC / INI1 IHC Loss PATHOGNOMONIC / Sibling MANDATORY / Tazemetostat', locus: '22q11.23', size: '385 aa / 45 kDa',   inh: 'AD LOF' },
  'PMS2':    { full: 'CMMRD Biallelic / GBM TMB >100 PATHOGNOMONIC / CALMs + Brain Tumour / PD-1 Active / Lynch Mono',  locus: '7p22.2',   size: '862 aa / 96 kDa',   inh: 'AR/AD LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color,
      border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px',
      fontSize: 11, fontWeight: 600, marginRight: 4,
    }}>{text}</span>
  );
}

export default function HereditaryBrainCNSTumourAtlas() {
  const [tab, setTab]                   = useState('Overview');
  const [overview, setOverview]         = useState(null);
  const [breakdown, setBreakdown]       = useState(null);
  const [definitions, setDefinitions]   = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ];
    Promise.all(endpoints)
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const geneColor = g => GENE_COLORS[g] || '#607d8b';

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#0a0a0a', minHeight: '100vh', color: '#e8e8e8', padding: 24 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a0000 0%,#0d0d2b 60%,#001a0d 100%)', borderRadius: 12, padding: '28px 32px', marginBottom: 24 }}>
        <div style={{ fontSize: 11, color: '#ef9a9a', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 8 }}>
          Hereditary Disease Atlas · Brain &amp; CNS Tumour Predisposition · 8-Gene Reference
        </div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, color: '#fff' }}>
          🧬 Hereditary Brain &amp; CNS Tumour Predisposition Atlas
        </h1>
        <div style={{ marginTop: 10, color: '#b0bec5', fontSize: 13 }}>
          Complete 8-Gene Predisposition Reference · TP53 · PTCH1 · SUFU · PTEN · APC · VHL · SMARCB1 · PMS2
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(GENE_COLORS).map(([g, c]) => (
            <Badge key={g} text={g} color={c} />
          ))}
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: '#ef9a9a', fontWeight: 600 }}>
          ⚠ CRITICAL: TP53 LFS → AVOID RADIATION ABSOLUTELY · PTCH1/SUFU Gorlin → AVOID RADIATION (field cancerisation) · SMARCB1 AT/RT → SIBLING TESTING MANDATORY · PMS2 CMMRD → TMB &gt;100 check + PD-1
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: '#90caf9' }}>
          320-patient aggregate · 8 × 40 seeds · seeds 3150-3157 · CPC &lt;5yr = PATHOGNOMONIC LFS · Calcified falx = PATHOGNOMONIC Gorlin · LDD striated MRI = PATHOGNOMONIC PTEN · AT/RT &lt;3yr = PATHOGNOMONIC SMARCB1
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#b71c1c' : '#1e1e1e',
            color: tab === t ? '#fff' : '#aaa', fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#90caf9', padding: 40, textAlign: 'center' }}>Loading atlas data…</div>}
      {error   && <div style={{ color: '#ef9a9a', padding: 20, background: '#1a0000', borderRadius: 8 }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Genes',    val: overview.total_genes },
              { label: 'Total Patients', val: overview.total_patients },
              { label: 'Seed Range',     val: overview.seed_range },
              { label: 'Patients/Gene',  val: 40 },
            ].map(({ label, val }) => (
              <div key={label} style={{ background: '#1e1e1e', borderRadius: 8, padding: '18px 20px', textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: '#ef9a9a' }}>{val}</div>
                <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Gene cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(340px,1fr))', gap: 16, marginBottom: 24 }}>
            {overview.genes.map(g => {
              const color = geneColor(g);
              const info  = GENE_INFO[g] || {};
              const inh   = (overview.inheritance_modes || {})[g] || '';
              return (
                <div key={g} style={{ background: '#1e1e1e', borderRadius: 8, padding: 18, borderLeft: `4px solid ${color}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span style={{ fontSize: 20, fontWeight: 800, color }}>{g}</span>
                      <span style={{ fontSize: 11, color: '#888', marginLeft: 8 }}>{info.locus}</span>
                    </div>
                    <Badge text={info.inh || 'AD LOF'} color={color} />
                  </div>
                  <div style={{ fontSize: 12, color: '#ccc', marginTop: 6 }}>{info.full}</div>
                  <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{info.size}</div>
                  {inh && (
                    <div style={{ fontSize: 11, color: '#b0bec5', marginTop: 8, background: '#111', borderRadius: 4, padding: '6px 8px' }}>
                      {inh.substring(0, 220)}{inh.length > 220 ? '…' : ''}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Critical warnings */}
          <div style={{ background: '#1a0000', border: '2px solid #b71c1c', borderRadius: 8, padding: '14px 20px', marginBottom: 16 }}>
            <div style={{ color: '#ef9a9a', fontWeight: 700, fontSize: 14, marginBottom: 8 }}>
              ⚠ CRITICAL RADIATION AVOIDANCE + MANDATORY ACTION RULES
            </div>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              <li style={{ fontSize: 12, color: '#ffcdd2', marginBottom: 5 }}>
                <strong>TP53 germline (LFS):</strong> AVOID ALL IONISING RADIATION ABSOLUTELY — standard RT induces radiation-field sarcoma in LFS carriers; chemotherapy-only protocols for CNS tumours
              </li>
              <li style={{ fontSize: 12, color: '#ffcdd2', marginBottom: 5 }}>
                <strong>PTCH1/SUFU (Gorlin):</strong> AVOID ALL RADIATION ABSOLUTELY — WBRT in Gorlin child → thousands of BCCs in irradiated skin; use Baby Brain / chemo-only protocol exclusively for medulloblastoma
              </li>
              <li style={{ fontSize: 12, color: '#ffcdd2', marginBottom: 5 }}>
                <strong>SMARCB1 (RTPS2 / AT/RT):</strong> Sibling testing MANDATORY immediately after AT/RT diagnosis in any child — 50% sibling risk; brain+spine MRI 3-6 monthly in SMARCB1-positive siblings from birth
              </li>
              <li style={{ fontSize: 12, color: '#ffcdd2' }}>
                <strong>PMS2 biallelic (CMMRD):</strong> Childhood GBM TMB &gt;100 mut/Mb = PATHOGNOMONIC CMMRD — check MMR IHC and TMB on ALL paediatric GBM; PD-1 immunotherapy may achieve durable remission
              </li>
            </ul>
          </div>

          {/* Key clinical rules */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20, marginBottom: 16 }}>
            <h3 style={{ margin: '0 0 14px', color: '#ef9a9a', fontSize: 15 }}>⚠ Key Clinical Rules</h3>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              {(overview.key_clinical_rules || []).map((rule, i) => (
                <li key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 7, lineHeight: 1.5 }}>
                  {rule}
                </li>
              ))}
            </ul>
          </div>

          {/* Gene panel note */}
          {overview.gene_panel_note && (
            <div style={{ background: '#0d1b2a', borderRadius: 8, padding: 16, fontSize: 11, color: '#90caf9', lineHeight: 1.7 }}>
              <strong style={{ color: '#80cbc4' }}>Gene Panel &amp; Decision Tree:</strong>{' '}
              {overview.gene_panel_note}
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          {breakdown.genes.map(g => {
            const color  = geneColor(g.gene);
            const isOpen = expandedGene === g.gene;
            const info   = GENE_INFO[g.gene] || {};
            return (
              <div key={g.gene} style={{ background: '#1e1e1e', borderRadius: 8, marginBottom: 12, overflow: 'hidden', borderLeft: `4px solid ${color}` }}>
                <div
                  onClick={() => setExpandedGene(isOpen ? null : g.gene)}
                  style={{ padding: '14px 18px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                >
                  <div>
                    <span style={{ fontSize: 17, fontWeight: 700, color }}>{g.gene}</span>
                    <span style={{ fontSize: 12, color: '#888', marginLeft: 10 }}>{g.locus} · {info.full}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center', fontSize: 12 }}>
                    <span style={{ color: '#aaa' }}>n={g.n}</span>
                    <span style={{ color: '#ef9a9a' }}>Age {g.mean_age_diagnosis}yr</span>
                    <span style={{ color: isOpen ? '#fff' : '#666', fontSize: 16 }}>{isOpen ? '▲' : '▼'}</span>
                  </div>
                </div>
                {isOpen && (
                  <div style={{ padding: '0 18px 18px', borderTop: '1px solid #333' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 10, marginTop: 14 }}>
                      {Object.entries(g)
                        .filter(([k]) => k.endsWith('_pct'))
                        .map(([k, v]) => (
                          <div key={k} style={{ background: '#111', borderRadius: 6, padding: '10px 12px' }}>
                            <div style={{ fontSize: 18, fontWeight: 700, color }}>{v}%</div>
                            <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
                              {k.replace(/_pct$/, '').replace(/_/g, ' ')}
                            </div>
                          </div>
                        ))}
                    </div>
                    {g.surveillance_key && (
                      <div style={{ marginTop: 12, fontSize: 11, color: '#80cbc4', background: '#0d1b2a', borderRadius: 4, padding: '8px 10px' }}>
                        <strong>Surveillance:</strong> {g.surveillance_key}
                      </div>
                    )}
                    {g.pathognomonic && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#ffcc80', background: '#1a1000', borderRadius: 4, padding: '8px 10px' }}>
                        <strong>Pathognomonic:</strong> {g.pathognomonic}
                      </div>
                    )}
                    {g.inheritance && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#b0bec5', background: '#111', borderRadius: 4, padding: '8px 10px', lineHeight: 1.6 }}>
                        <strong>Inheritance:</strong> {g.inheritance}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 16px', color: '#ef9a9a', fontSize: 15 }}>Syndrome Summary — Hereditary Brain &amp; CNS Tumour Predisposition</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#111' }}>
                    {['Gene', 'Syndrome', 'Locus', 'Size', 'Inheritance', 'Pathognomonic', 'Surveillance Key', 'n', 'Dx Age'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#ef9a9a', borderBottom: '1px solid #333', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes.map((g, idx) => {
                    const color = geneColor(g.gene);
                    const info  = GENE_INFO[g.gene] || {};
                    return (
                      <tr key={g.gene} style={{ background: idx % 2 === 0 ? '#181818' : '#1e1e1e' }}>
                        <td style={{ padding: '8px 10px', color, fontWeight: 700 }}>{g.gene}</td>
                        <td style={{ padding: '8px 10px', color: '#ccc', maxWidth: 200 }}>{info.full}</td>
                        <td style={{ padding: '8px 10px', color: '#aaa' }}>{g.locus}</td>
                        <td style={{ padding: '8px 10px', color: '#aaa' }}>{info.size}</td>
                        <td style={{ padding: '8px 10px', color: '#b0bec5' }}>{info.inh}</td>
                        <td style={{ padding: '8px 10px', color: '#ffcc80', fontSize: 11 }}>{g.pathognomonic}</td>
                        <td style={{ padding: '8px 10px', color: '#80cbc4', fontSize: 11 }}>{g.surveillance_key ? g.surveillance_key.split(';')[0] : '—'}</td>
                        <td style={{ padding: '8px 10px', color: '#e0e0e0' }}>{g.n}</td>
                        <td style={{ padding: '8px 10px', color: '#ef9a9a' }}>{g.mean_age_diagnosis}yr</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pathway classification */}
          <div style={{ background: '#0d1b2a', border: '1px solid #1565c0', borderRadius: 8, padding: '14px 20px', marginBottom: 16 }}>
            <div style={{ color: '#90caf9', fontWeight: 700, fontSize: 14, marginBottom: 8 }}>
              Hereditary Brain &amp; CNS Tumour — Molecular Pathway Classification
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 10 }}>
              {[
                { label: 'p53 Pathway (LFS — CPC/GBM/MB PATHOGNOMONIC)', desc: 'TP53 LOF · CPC <5yr PATHOGNOMONIC · AVOID RADIATION ABSOLUTELY · WBMRI Toronto', color: '#b71c1c' },
                { label: 'Hedgehog/PTCH1 (Gorlin — Desmoplastic MB)', desc: 'PTCH1 LOF · desmoplastic MB SHH · calcified falx · BCCs · AVOID RADIATION · Vismodegib', color: '#e65100' },
                { label: 'Hedgehog/SUFU (BCNS2 — Infant MB)', desc: 'SUFU LOF · MB penetrance 15-20% · MBEN infant PATHOGNOMONIC · AVOID RADIATION', color: '#f57f17' },
                { label: 'PI3K/mTOR (Cowden — Lhermitte-Duclos)', desc: 'PTEN LOF · LDD striated MRI PATHOGNOMONIC · macrocephaly · breast/thyroid · Everolimus', color: '#2e7d32' },
                { label: 'WNT (FAP/Turcot2 — WNT-MB)', desc: 'APC LOF · WNT-MB best prognosis · CHRPE PATHOGNOMONIC · Gardner desmoids', color: '#1565c0' },
                { label: 'HIF-2α (VHL — Hemangioblastoma)', desc: 'VHL LOF · CNS/retinal hemangioblastoma PATHOGNOMONIC · annual retinal age 1 · Belzutifan FDA2021', color: '#6a1b9a' },
                { label: 'SWI/SNF (SMARCB1 — AT/RT PATHOGNOMONIC)', desc: 'SMARCB1 LOF · AT/RT <3yr PATHOGNOMONIC · INI1 IHC loss · sibling MANDATORY · tazemetostat', color: '#4a148c' },
                { label: 'MMR (CMMRD/PMS2 — Hypermutation)', desc: 'PMS2 biallelic CMMRD · GBM TMB>100 PATHOGNOMONIC · CALMs · PD-1 active', color: '#004d40' },
              ].map(({ label, desc, color }) => (
                <div key={label} style={{ background: '#111', borderRadius: 6, padding: '10px 12px', borderTop: `3px solid ${color}` }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color }}>{label}</div>
                  <div style={{ fontSize: 11, color: '#aaa', marginTop: 4 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Protein size reference */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20 }}>
            <h3 style={{ margin: '0 0 16px', color: '#ef9a9a', fontSize: 15 }}>Protein Size Reference</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 10 }}>
              {Object.entries(GENE_INFO).map(([g, info]) => {
                const color = geneColor(g);
                return (
                  <div key={g} style={{ background: '#111', borderRadius: 6, padding: '12px 14px', borderTop: `3px solid ${color}` }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color }}>{g}</div>
                    <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{info.size}</div>
                    <div style={{ fontSize: 11, color: '#aaa', marginTop: 2 }}>{info.locus}</div>
                    <div style={{ fontSize: 10, color: '#666', marginTop: 4 }}>{info.inh}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ marginBottom: 12, fontSize: 12, color: '#888' }}>
            {definitions.definitions ? definitions.definitions.length : 0} clinical definitions · TP53 CPC LFS AVOID RADIATION · PTCH1 Gorlin calcified falx Vismodegib · SUFU BCNS2 MBEN infant · PTEN Lhermitte-Duclos macrocephaly · APC WNT-MB CHRPE Gardner · VHL hemangioblastoma Belzutifan · SMARCB1 AT/RT INI1 sibling · PMS2 CMMRD TMB100 PD-1
          </div>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={{ background: '#1e1e1e', borderRadius: 8, marginBottom: 12, overflow: 'hidden' }}>
              <div style={{ background: '#b71c1c', padding: '10px 16px', fontSize: 13, fontWeight: 700, color: '#fff' }}>
                {d.term.replace(/-/g, ' ')}
              </div>
              <div style={{ padding: '14px 16px', fontSize: 12, color: '#ccc', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
