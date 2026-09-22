'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-lung-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'EGFR':  '#00695c',  // deep teal        — Hereditary NSCLC; germline T790M; osimertinib
  'STK11': '#e65100',  // deep orange       — Peutz-Jeghers; lung adeno 7-17×; KRAS immunotherapy resistance
  'TP53':  '#b71c1c',  // deep red          — LFS; SCLC; AVOID RADIATION ABSOLUTELY
  'RB1':   '#1565c0',  // deep blue         — Retinoblastoma; secondary SCLC 15-20×; CDK4/6i inactive
  'BRCA2': '#6a1b9a',  // deep purple       — HBOC; lung Ca 2-3×; platinum + PARP inhibitor
  'FLCN':  '#2e7d32',  // forest green      — Birt-Hogg-Dubé; pulmonary cysts; pneumothorax; pleurodesis
  'ATM':   '#827717',  // dark yellow-olive — A-T; radiosensitivity ABSOLUTE; lung 2-4×; ceralasertib
  'BAP1':  '#283593',  // dark indigo       — BAP1 TPDS; mesothelioma PATHOGNOMONIC; AVOID ASBESTOS
};

const GENE_INFO = {
  'EGFR':  { full: 'Hereditary NSCLC / Germline T790M / Osimertinib 3rd-gen TKI / Bilateral GGOs',           locus: '7p11.2',   size: '1210 aa / 134 kDa', inh: 'AD GOF' },
  'STK11': { full: 'Peutz-Jeghers / Lung Adeno 7-17× RR / KRAS-co-mut / Immunotherapy Resistance',          locus: '19p13.3',  size: '433 aa / 48 kDa',  inh: 'AD LOF' },
  'TP53':  { full: 'LFS / SCLC + Lung / AVOID RADIATION ABSOLUTELY / WBMRI Toronto',                        locus: '17p13.1',  size: '393 aa / 43 kDa',  inh: 'AD LOF' },
  'RB1':   { full: 'Hereditary Retinoblastoma / Secondary SCLC 15-20× / CDK4/6i Inactive / Bilateral Rb',   locus: '13q14.2',  size: '928 aa / 110 kDa', inh: 'AD LOF' },
  'BRCA2': { full: 'HBOC / Lung Ca 2-3× / Platinum + PARP Inhibitor (Olaparib) / Male Breast PATHOGNOMONIC', locus: '13q12.3',  size: '3418 aa / 384 kDa',inh: 'AD LOF' },
  'FLCN':  { full: 'Birt-Hogg-Dubé / Bilateral Cysts PATHOGNOMONIC / Pneumothorax 24-38% / Pleurodesis',    locus: '17p11.2',  size: '579 aa / 64 kDa',  inh: 'AD LOF' },
  'ATM':   { full: 'A-T / Radiosensitivity ABSOLUTE (biallelic) / Lung 2-4× (monoallelic) / Ceralasertib',   locus: '11q22.3',  size: '3056 aa / 350 kDa',inh: 'AR/AD LOF' },
  'BAP1':  { full: 'BAP1 TPDS / Mesothelioma PATHOGNOMONIC 30-60× / AVOID ASBESTOS / Tazemetostat',         locus: '3p21.1',   size: '729 aa / 80 kDa',  inh: 'AD LOF' },
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

export default function HereditaryLungCancerAtlas() {
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
      <div style={{ background: 'linear-gradient(135deg,#001a0d 0%,#001a2e 60%,#1a0033 100%)', borderRadius: 12, padding: '28px 32px', marginBottom: 24 }}>
        <div style={{ fontSize: 11, color: '#80cbc4', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 8 }}>
          Hereditary Disease Atlas · Lung Cancer Predisposition · 8-Gene Reference
        </div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, color: '#fff' }}>
          🧬 Hereditary Lung Cancer Atlas
        </h1>
        <div style={{ marginTop: 10, color: '#b0bec5', fontSize: 13 }}>
          Complete 8-Gene Predisposition Reference · EGFR · STK11 · TP53 · RB1 · BRCA2 · FLCN · ATM · BAP1
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(GENE_COLORS).map(([g, c]) => (
            <Badge key={g} text={g} color={c} />
          ))}
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: '#ef9a9a', fontWeight: 600 }}>
          ⚠ CRITICAL: TP53 LFS → AVOID RADIATION ABSOLUTELY · ATM biallelic (A-T) → standard RT LETHAL · BAP1 TPDS → AVOID ALL ASBESTOS · RB1 → CDK4/6i inactive in RB1-null SCLC
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: '#80cbc4' }}>
          320-patient aggregate · 8 × 40 seeds · seeds 3142-3149 · EGFR germline T790M: never-smoker lung adeno + family history → germline panel · STK11 lung adeno 7-17× RR (highest hereditary lung risk) · FLCN bilateral cysts PATHOGNOMONIC · BAP1 mesothelioma 30-60× RR
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#00695c' : '#1e1e1e',
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
                <div style={{ fontSize: 28, fontWeight: 800, color: '#80cbc4' }}>{val}</div>
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
                    <Badge text={info.inh || 'AD'} color={color} />
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

          {/* Critical warnings box */}
          <div style={{ background: '#1a0000', border: '2px solid #b71c1c', borderRadius: 8, padding: '14px 20px', marginBottom: 16 }}>
            <div style={{ color: '#ef9a9a', fontWeight: 700, fontSize: 14, marginBottom: 8 }}>
              ⚠ CRITICAL RADIATION + ASBESTOS AVOIDANCE RULES
            </div>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              <li style={{ fontSize: 12, color: '#ffcdd2', marginBottom: 5 }}>
                <strong>TP53 germline (LFS):</strong> AVOID ALL THORACIC RADIATION ABSOLUTELY — standard-dose RT causes radiation-field sarcoma; use systemic therapy (chemotherapy + immunotherapy + TKI) exclusively
              </li>
              <li style={{ fontSize: 12, color: '#ffcdd2', marginBottom: 5 }}>
                <strong>ATM biallelic (A-T):</strong> Standard-dose RT (60 Gy NSCLC) = POTENTIALLY LETHAL — 3-5× radiation hypersensitivity; systemic-only management; PARP inhibitor + ATR inhibitor (ceralasertib)
              </li>
              <li style={{ fontSize: 12, color: '#ffcdd2', marginBottom: 5 }}>
                <strong>BAP1 TPDS:</strong> AVOID ALL ASBESTOS EXPOSURE — mesothelioma 30-60× RR; BAP1 + asbestos latency only 15-25yr (vs 40yr sporadic); even low-level exposure is high-risk; occupational history mandatory every visit
              </li>
              <li style={{ fontSize: 12, color: '#ffcdd2' }}>
                <strong>RB1 germline (Rb survivors):</strong> AVOID radiation-field treatments — secondary SCLC risk elevated; CDK4/6 inhibitors (palbociclib/ribociclib/abemaciclib) are INACTIVE in RB1-null SCLC
              </li>
            </ul>
          </div>

          {/* Key clinical rules */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20, marginBottom: 16 }}>
            <h3 style={{ margin: '0 0 14px', color: '#80cbc4', fontSize: 15 }}>⚠ Key Clinical Rules</h3>
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
            <div style={{ background: '#0d1b2a', borderRadius: 8, padding: 16, fontSize: 11, color: '#80cbc4', lineHeight: 1.7 }}>
              <strong style={{ color: '#90caf9' }}>Gene Panel &amp; Decision Tree:</strong>{' '}
              {overview.gene_panel_note}
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          {breakdown.genes.map(g => {
            const color   = geneColor(g.gene);
            const isOpen  = expandedGene === g.gene;
            const info    = GENE_INFO[g.gene] || {};
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
                    <span style={{ color: '#80cbc4' }}>Age {g.mean_age_diagnosis}yr</span>
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
            <h3 style={{ margin: '0 0 16px', color: '#80cbc4', fontSize: 15 }}>Syndrome Summary — Hereditary Lung Cancer Predisposition</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#111' }}>
                    {['Gene', 'Syndrome', 'Locus', 'Size', 'Inheritance', 'Pathognomonic', 'Surveillance Key', 'n', 'Dx Age'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#80cbc4', borderBottom: '1px solid #333', whiteSpace: 'nowrap' }}>{h}</th>
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
                        <td style={{ padding: '8px 10px', color: '#80cbc4' }}>{g.mean_age_diagnosis}yr</td>
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
              Hereditary Lung Cancer — Molecular Pathway Classification
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 10 }}>
              {[
                { label: 'EGFR RTK GOF (Hereditary NSCLC)', desc: 'Germline T790M → never-smoker adeno · osimertinib 3rd-gen TKI', color: '#00695c' },
                { label: 'LKB1/AMPK (PJS — Lung Adeno)', desc: 'STK11 LOF · 7-17× lung RR · KRAS co-mut · immunotherapy resistance', color: '#e65100' },
                { label: 'p53 Pathway (LFS — SCLC + Lung)', desc: 'TP53 LOF · AVOID RADIATION · WBMRI Toronto · SCLC + lung LFS', color: '#b71c1c' },
                { label: 'pRB/E2F (Rb — Secondary SCLC)', desc: 'RB1 LOF · bilateral Rb germline · SCLC 15-20× · CDK4/6i inactive', color: '#1565c0' },
                { label: 'HRD/HR Repair (BRCA2 — HRD Lung)', desc: 'BRCA2 LOF · lung 2-3× · olaparib · platinum sensitivity', color: '#6a1b9a' },
                { label: 'mTOR/FLCN (BHD — Cysts)', desc: 'FLCN LOF · bilateral cysts PATHOGNOMONIC · pneumothorax 24-38% · pleurodesis', color: '#2e7d32' },
                { label: 'DDR/ATM (A-T — Radiosensitivity)', desc: 'ATM LOF · biallelic: RT lethal · monoallelic: lung 2-4× · ceralasertib', color: '#827717' },
                { label: 'BAP1/Polycomb (TPDS — Mesothelioma)', desc: 'BAP1 LOF · mesothelioma 30-60× · AVOID ASBESTOS · tazemetostat', color: '#283593' },
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
            <h3 style={{ margin: '0 0 16px', color: '#80cbc4', fontSize: 15 }}>Protein Size Reference</h3>
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
            {definitions.definitions ? definitions.definitions.length : 0} clinical definitions · EGFR germline T790M osimertinib · STK11 PJS immunotherapy resistance · BAP1 mesothelioma asbestos · ATM radiosensitivity · FLCN BHD pneumothorax pleurodesis
          </div>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={{ background: '#1e1e1e', borderRadius: 8, marginBottom: 12, overflow: 'hidden' }}>
              <div style={{ background: '#00695c', padding: '10px 16px', fontSize: 13, fontWeight: 700, color: '#fff' }}>
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
