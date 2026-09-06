'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MEN1:    '#7b1fa2',  // deep purple — menin, parathyroid + pituitary + pNET
  RET:     '#b71c1c',  // deep red — tyrosine kinase, MTC + pheo (life-threatening if missed)
  VHL:     '#1565c0',  // deep blue — HIF pathway, ccRCC + hemangioblastoma
  SDHB:    '#e65100',  // deep orange — highest malignant potential 30-50%
  SDHD:    '#bf360c',  // dark ember — head/neck PGL, paternal imprinting
  CDKN1B:  '#2e7d32',  // deep green — p27, MEN4 MEN1-like
  CDC73:   '#4e342e',  // dark brown — parafibromin, parathyroid carcinoma
  PRKAR1A: '#37474f',  // dark slate — PKA, Carney complex cardiac myxoma
};

const GENE_DISEASE = {
  MEN1:    'MEN1 AD — Parathyroid 95% + Pituitary 40% + Pancreatic NET 70% — Sunitinib/Everolimus pNETs — Annual Ca/PTH/CgA — Multiglandular Parathyroidectomy',
  RET:     'MEN2A/MEN2B/FMTC AD GOF — MTC 100% + Pheo 50% + HPT 25% — PROPHYLACTIC THYROIDECTOMY Codon-Stratified — M918T <6m Life — C634 Age 5 — PHEO BEFORE SURGERY',
  VHL:     'VHL AD — ccRCC + Hemangioblastoma + Retinal Angioma + Pheo + PNET + ELST — Belzutifan FDA 2021 HIF-2α — Annual Multi-Organ Surveillance — Type 2B Highest Risk',
  SDHB:    'PGL4 AD — Highest Malignant 30-50% Metastatic — Extra-Adrenal Predominant — SDHB IHC ABSENT — Ga-DOTATATE PET — 177Lu-DOTATATE PRRT — Sporadic Presentation 30-40%',
  SDHD:    'PGL1 AD Paternal Imprinting — Head/Neck PGL Predominant — MATERNAL CARRIERS SILENT — Multifocal 40% — Non-Secreting 70% — Carotid Body Embolise Pre-Op',
  CDKN1B:  'MEN4 AD — p27 Kip1 — MEN1-Like PHPT + Pituitary — Sequence MEN1 FIRST — Treat as MEN1 Protocol — Annual Ca/PTH + Pituitary MRI Every 3 Years',
  CDC73:   'HPT-JT AD — Parathyroid Carcinoma 15% HIGHEST — Jaw Ossifying Fibroma PATHOGNOMONIC — Ca ≥3.0 mmol/L — Parafibromin IHC ABSENT — En Bloc Resection Mandatory',
  PRKAR1A: 'Carney Complex AD — Cardiac Myxoma EMBOLIC STROKE URGENT — Annual Echo MANDATORY — Spotty Pigmentation — PPNAD Paradoxical Dexamethasone PATHOGNOMONIC — LCCSCT Males',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /URGENT|ABSOLUTE|NEVER|STOP|FATAL|LETHAL|PATHOGNOMONIC|MANDATORY|CURATIVE|CONTRAINDICATED|AVOID|FIRST\s+LINE|HIGHEST|EMBOLIC|PROPHYLACTIC|EN\s+BLOC/i.test(text);
  const isWarning = /MONITOR|SCREEN|ANNUAL|REQUIRED|PROTOCOL|CONTINUOUS|LIFELONG|IMMEDIATELY|CASCADE|DISTINGUISH|START|ASSAY|PARTIAL|BEFORE\s+SURGERY|CODON-STRATIFIED|IMPRINTING/i.test(text);
  const bg = isCI ? '#b71c1c' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, genes } = data;
  return (
    <div>
      <h2 style={{ color: '#7b1fa2', marginBottom: 8 }}>Hereditary Endocrine Tumour Atlas — 8-Gene Reference</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>
        320 patients (8 × 40, seeds 1566–1573) · MEN1 / RET / VHL / SDHB / SDHD / CDKN1B / CDC73 / PRKAR1A
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 12, marginBottom: 24 }}>
        {[
          ['Total Patients', s.total_patients],
          ['Mean Dx Age (y)', s.mean_dx_age_years],
          ['Mean Dx Delay (m)', s.mean_dx_delay_months],
          ['Cascade Tested %', s.cascade_tested_pct + '%'],
          ['MEN1 PHPT %', s.men1_phpt_pct + '%'],
          ['RET Prophylactic Thyroidectomy %', s.ret_prophylactic_thyroidectomy_pct + '%'],
          ['VHL Retinal Angioma %', s.vhl_retinal_angioma_pct + '%'],
          ['SDHB Malignant %', s.sdhb_malignant_pct + '%'],
          ['SDHD Multifocal %', s.sdhd_multifocal_pct + '%'],
          ['CDC73 Parathyroid Ca %', s.cdc73_parathyroid_carcinoma_pct + '%'],
          ['PRKAR1A Cardiac Myxoma %', s.prkar1a_cardiac_myxoma_pct + '%'],
        ].map(([label, value]) => (
          <div key={label} style={{
            background: '#f5f7fa', borderRadius: 8, padding: '12px 16px',
            borderLeft: '4px solid #7b1fa2',
          }}>
            <div style={{ fontSize: 12, color: '#888' }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#7b1fa2' }}>{value}</div>
          </div>
        ))}
      </div>

      <h3 style={{ marginBottom: 12 }}>Top Clinical Alerts</h3>
      {top_alerts?.slice(0, 10).map((a, i) => <AlertBadge key={i} text={a} />)}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Gene Summary</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#ede7f6' }}>
              {['Gene', 'Protein (short)', 'Locus', 'Inheritance', 'OMIM Disease', 'Mean Dx Age', 'N'].map(h => (
                <th key={h} style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #7b1fa2' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes?.map(g => (
              <tr key={g.gene} style={{ borderBottom: '1px solid #f0f0f0' }}>
                <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.protein_short}</td>
                <td style={{ padding: '6px 10px' }}>{g.locus}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.inheritance}</td>
                <td style={{ padding: '6px 10px' }}>{g.omim_disease}</td>
                <td style={{ padding: '6px 10px' }}>{g.mean_dx_age}y</td>
                <td style={{ padding: '6px 10px' }}>{g.n_patients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { genes } = data;
  return (
    <div>
      <h3 style={{ marginBottom: 16 }}>Gene Disease Associations</h3>
      <div style={{ display: 'grid', gap: 12 }}>
        {genes?.map(g => (
          <div key={g.gene} style={{
            padding: 14, borderRadius: 8, background: '#f5f7fa',
            borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#333'}`,
          }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#333', fontSize: 15 }}>
              {g.gene} — {g.locus}
            </div>
            <div style={{ fontSize: 13, color: '#444', marginTop: 4 }}>
              {GENE_DISEASE[g.gene]}
            </div>
            <div style={{ fontSize: 12, color: '#777', marginTop: 4 }}>
              OMIM: {g.omim_disease} · {g.inheritance} · Mean Dx Age: {g.mean_dx_age}y · N={g.n_patients}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  if (!selected && genes.length) setTimeout(() => setSelected(genes[0]), 0);
  const info = selected ? data[selected] : null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 16 }}>
      <div>
        {genes.map(g => (
          <button key={g} onClick={() => setSelected(g)} style={{
            display: 'block', width: '100%', padding: '8px 12px', marginBottom: 4,
            borderRadius: 6, border: 'none', cursor: 'pointer', textAlign: 'left',
            background: selected === g ? (GENE_COLORS[g] || '#7b1fa2') : '#f0f0f0',
            color: selected === g ? '#fff' : '#333',
            fontWeight: selected === g ? 700 : 400, fontSize: 13,
          }}>{g}</button>
        ))}
      </div>
      {info && (
        <div>
          <h3 style={{ color: GENE_COLORS[selected] || '#7b1fa2', marginBottom: 4 }}>
            {info.gene} — {info.aa} · {info.kDa} · {info.locus}
          </h3>
          <p style={{ fontSize: 13, color: '#666', marginBottom: 12 }}>
            OMIM Gene: {info.omim_gene} · Disease: {info.omim_disease} · {info.inheritance}
          </p>
          <div style={{ background: '#f5f7fa', borderRadius: 8, padding: 12, marginBottom: 16, fontSize: 13, lineHeight: 1.6 }}>
            {info.alias}
          </div>
          <div style={{ background: '#ede7f6', borderRadius: 8, padding: 12, marginBottom: 16, fontSize: 12, lineHeight: 1.5 }}>
            <strong>Molecular Class: </strong>{info.gene_class}
          </div>

          <h4 style={{ marginBottom: 8 }}>Variant Distribution</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
            {Object.entries(info.etiologies || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
              <span key={k} style={{
                background: GENE_COLORS[selected] || '#7b1fa2', color: '#fff',
                borderRadius: 20, padding: '4px 12px', fontSize: 12,
              }}>{k}: {v}</span>
            ))}
          </div>

          <h4 style={{ marginBottom: 8 }}>Clinical Feature Rates</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(180px,1fr))', gap: 8, marginBottom: 16 }}>
            {Object.entries(info.stats || {}).filter(([k]) => !['mean_dx_age','mean_dx_delay_months'].includes(k)).map(([k, v]) => (
              <div key={k} style={{ background: '#f5f7fa', padding: '8px 12px', borderRadius: 6, borderLeft: `3px solid ${GENE_COLORS[selected] || '#7b1fa2'}` }}>
                <div style={{ fontSize: 11, color: '#888' }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[selected] || '#7b1fa2' }}>{typeof v === 'number' ? (v > 10 ? v + '%' : v) : v}</div>
              </div>
            ))}
          </div>

          <h4 style={{ marginBottom: 8 }}>Diagnosis Delay Distribution</h4>
          <div style={{ marginBottom: 16 }}>
            {Object.entries(info.dx_delay_distribution || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <span style={{ width: 50, fontSize: 12, color: '#555' }}>{k}</span>
                <div style={{ flex: 1, background: '#e0e0e0', borderRadius: 4, height: 14 }}>
                  <div style={{ width: `${(v / info.n_patients) * 100}%`, background: GENE_COLORS[selected] || '#7b1fa2', height: 14, borderRadius: 4 }} />
                </div>
                <span style={{ fontSize: 12, width: 24 }}>{v}</span>
              </div>
            ))}
          </div>

          <h4 style={{ marginBottom: 8 }}>Key Clinical Alerts</h4>
          {info.key_alerts?.map((a, i) => <AlertBadge key={i} text={a} />)}

          <h4 style={{ marginBottom: 8, marginTop: 12 }}>Sample Patients (first 10)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#e0e0e0' }}>
                  {Object.keys(info.patients?.[0] || {}).map(k => (
                    <th key={k} style={{ padding: '4px 8px', textAlign: 'left' }}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {info.patients?.map(p => (
                  <tr key={p.id} style={{ borderBottom: '1px solid #eee' }}>
                    {Object.values(p).map((v, i) => (
                      <td key={i} style={{ padding: '4px 8px' }}>
                        {typeof v === 'boolean' ? (v ? '✓' : '✗') : v?.toString() ?? '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ marginBottom: 16 }}>Hereditary Endocrine Tumour — Key Concepts</h3>
      {Object.entries(data.concepts || {}).map(([k, v]) => (
        <div key={k} style={{ marginBottom: 16, padding: 14, background: '#f5f7fa', borderRadius: 8, borderLeft: '4px solid #7b1fa2' }}>
          <div style={{ fontWeight: 700, color: '#7b1fa2', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
          <div style={{ fontSize: 13, lineHeight: 1.6, color: '#333' }}>{v}</div>
        </div>
      ))}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Pharmacological Distinctions</h3>
      {data.pharmacological_distinctions?.map((d, i) => (
        <div key={i} style={{ marginBottom: 8, padding: '10px 14px', background: '#fff3e0', borderRadius: 8, borderLeft: '4px solid #e65100', fontSize: 13 }}>
          {d}
        </div>
      ))}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Key Standards & References</h3>
      {data.key_standards?.map((s, i) => (
        <div key={i} style={{ marginBottom: 6, fontSize: 13, color: '#444', paddingLeft: 12, borderLeft: '3px solid #7b1fa2' }}>
          {s}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryEndocrineTumourAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-endocrine-tumour-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-endocrine-tumour-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-endocrine-tumour-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error: {error}</div>;

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#7b1fa2', marginBottom: 4 }}>
          🧬 Hereditary Endocrine Tumour Atlas
        </h1>
        <p style={{ color: '#666', fontSize: 14 }}>
          Complete 8-Gene Hereditary Endocrine Tumour Syndrome Reference — MEN1 (Parathyroid/Pituitary/pNET) ·
          RET (MEN2A/MEN2B/MTC) · VHL (ccRCC/Hemangioblastoma) · SDHB (Metastatic PGL) ·
          SDHD (Head/Neck PGL, Paternal Imprinting) · CDKN1B (MEN4) · CDC73 (HPT-JT/Parathyroid Carcinoma) ·
          PRKAR1A (Carney Complex) · 320 patients (8×40, seeds 1566–1573)
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: '8px 8px 0 0', border: 'none', cursor: 'pointer',
            background: tab === t ? '#7b1fa2' : '#f0f0f0',
            color: tab === t ? '#fff' : '#444', fontWeight: tab === t ? 700 : 400,
            fontSize: 14, borderBottom: tab === t ? '2px solid #7b1fa2' : 'none',
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={overview} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
