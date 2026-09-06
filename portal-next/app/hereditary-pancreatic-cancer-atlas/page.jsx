'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BRCA2:  '#1565c0',  // deep blue — POLO olaparib FDA approved, highest PDAC RR among HR genes
  CDKN2A: '#7b1fa2',  // deep purple — FAMMM, highest absolute PDAC risk, CDK4/6 pathway
  ATM:    '#4e342e',  // dark brown — PI3K-DDR kinase, platinum-sensitive, PARPi emerging
  STK11:  '#e65100',  // deep orange — Peutz-Jeghers, 36% lifetime risk, mTOR pathway
  BRCA1:  '#2e7d32',  // dark green — weaker PDAC risk, HBOC priority
  PALB2:  '#00695c',  // dark teal — emerging data, PARPi off-label, BRCA2 bridge
  PRSS1:  '#b71c1c',  // dark red — hereditary pancreatitis, 40–57× RR, NOT DNA repair
  MLH1:   '#37474f',  // dark slate — Lynch syndrome, dMMR, pembrolizumab
};

const GENE_DISEASE = {
  BRCA2:  'HPANCA-1 AD — 5–10× Pancreatic RR — POLO Olaparib FDA 2019 — Cisplatin Preferred Upfront — Germline Testing All Metastatic PDAC',
  CDKN2A: 'HPANCA-2 AD — FAMMM Syndrome — 25–58% Lifetime Pancreatic Risk — Atypical Moles + Melanoma — No Approved Targeted Therapy',
  ATM:    'HPANCA-3 AD-Heterozygous — ~6× Pancreatic RR — Platinum-Sensitive — PARPi Emerging Not Labelled — Radiation Sensitivity Intermediate',
  STK11:  'HPANCA-4 AD — Peutz-Jeghers Syndrome — 36% Lifetime Pancreatic Risk — Mucocutaneous Pigmentation Pathognomonic — Surveillance Age 25–30',
  BRCA1:  'HPANCA-5 AD — ~2% Lifetime Pancreatic Risk — POLO Covers gBRCA1 (Limited Data) — HBOC Breast/Ovarian Priority — RRSO Mandatory',
  PALB2:  'HPANCA-6 AD — 2–4% Pancreatic Risk — POLO Excluded PALB2 — PARPi Off-Label Rational — TBCRC048 82% ORR Breast',
  PRSS1:  'HPANCA-7 AD-GOF — Hereditary Pancreatitis → 40–57× PDAC RR — R122H + N29I Founders — TPIAT Severe Disease — Smoking ABSOLUTE-CI',
  MLH1:   'HPANCA-8 AD — Lynch Syndrome — 3–4% Lifetime Pancreatic Risk — dMMR → Pembrolizumab FDA 2017 — IHC Mandatory All PDAC',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#94a3b8' }}>Loading…</div>;
}

function ErrorBox({ msg }) {
  return (
    <div style={{ padding: '1rem', background: '#450a0a', borderRadius: 8, color: '#fca5a5', margin: '1rem 0' }}>
      Error: {msg}
    </div>
  );
}

function KPI({ label, value, color }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem',
      borderLeft: `4px solid ${color || '#6366f1'}`, minWidth: 160,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#a5b4fc' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function AlertBadge({ text }) {
  const [key, ...rest] = text.split(': ');
  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: '0.6rem 0.9rem',
      borderLeft: '3px solid #f59e0b', marginBottom: 6, fontSize: 12,
    }}>
      <span style={{ color: '#fcd34d', fontWeight: 700 }}>{key}</span>
      {rest.length > 0 && <span style={{ color: '#cbd5e1' }}>: {rest.join(': ')}</span>}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats || {};
  return (
    <div>
      <p style={{ color: '#94a3b8', marginBottom: '1.5rem', fontSize: 13 }}>{data.subtitle}</p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '2rem' }}>
        <KPI label="Total Patients" value={s.total_patients} color="#6366f1" />
        <KPI label="Mean Dx Age (yr)" value={s.mean_dx_age_years} color="#06b6d4" />
        <KPI label="Mean Dx Delay (mo)" value={s.mean_dx_delay_months} color="#f59e0b" />
        <KPI label="BRCA2 POLO PFS HR" value={s.brca2_polo_pfs_hr} color="#1565c0" />
        <KPI label="CDKN2A Lifetime PDAC Risk" value={`${s.cdkn2a_highest_absolute_pdac_risk_pct}%`} color="#7b1fa2" />
        <KPI label="STK11 Lifetime PDAC Risk" value={`${s.stk11_lifetime_pdac_risk_pct}%`} color="#e65100" />
        <KPI label="PRSS1 PDAC RR" value={s.prss1_pdac_rr} color="#b71c1c" />
        <KPI label="MLH1 KEYNOTE-158 ORR" value={`${s.mlh1_keynote158_pdac_orr_pct}%`} color="#37474f" />
        <KPI label="EUS Surveillance %" value={`${s.eus_surveillance_performed_pct}%`} color="#10b981" />
      </div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '0.8rem', fontSize: 14 }}>Clinical Alerts</h3>
      {(data.top_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
      <h3 style={{ color: '#f1f5f9', margin: '1.5rem 0 0.8rem', fontSize: 14 }}>Gene Summary</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Gene', 'Locus', 'Inheritance', 'OMIM Disease', 'Mean Dx Age', 'N Patients'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data.genes || []).map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#0f172a' }}>
                <td style={{ padding: '6px 10px', color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700 }}>{g.gene}</td>
                <td style={{ padding: '6px 10px', color: '#cbd5e1' }}>{g.locus}</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{g.omim_disease}</td>
                <td style={{ padding: '6px 10px', color: '#e2e8f0', textAlign: 'center' }}>{g.mean_dx_age}</td>
                <td style={{ padding: '6px 10px', color: '#e2e8f0', textAlign: 'center' }}>{g.n_patients}</td>
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
  return (
    <div>
      {Object.entries(data).map(([gene, info]) => (
        <div key={gene} style={{
          background: '#1e293b', borderRadius: 10, padding: '1.2rem', marginBottom: '1rem',
          borderLeft: `4px solid ${GENE_COLORS[gene] || '#6366f1'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap', marginBottom: 6 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc' }}>{gene}</span>
            <span style={{ fontSize: 11, color: '#94a3b8' }}>{info.locus} · {info.aa} aa · {info.kDa} kDa · {info.inheritance?.split(';')[0]}</span>
          </div>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{info.omim_disease}</div>
          <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 8 }}>{GENE_DISEASE[gene]}</div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>
            <strong style={{ color: '#f1f5f9' }}>Gene Class:</strong> {info.gene_class}
          </div>
          <div style={{ marginBottom: 6 }}>
            {(info.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 11 }}>
            <span style={{ background: '#0f172a', borderRadius: 4, padding: '2px 8px', color: '#94a3b8' }}>
              Mean Dx Age: {info.stats?.mean_dx_age}
            </span>
            <span style={{ background: '#0f172a', borderRadius: 4, padding: '2px 8px', color: '#94a3b8' }}>
              Lifetime PDAC Risk: {info.stats?.lifetime_risk_pct}%
            </span>
            <span style={{ background: '#0f172a', borderRadius: 4, padding: '2px 8px', color: '#94a3b8' }}>
              Delay: {info.stats?.mean_dx_delay_months} mo
            </span>
            <span style={{ background: '#0f172a', borderRadius: 4, padding: '2px 8px', color: '#94a3b8' }}>
              N=40 (seed {info.patients?.[0]?.seed})
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      {Object.entries(data).map(([gene, info]) => (
        <div key={gene} style={{
          background: '#0f172a', borderRadius: 10, padding: '1.2rem', marginBottom: '1.2rem',
          border: `1px solid ${GENE_COLORS[gene] || '#334155'}22`,
        }}>
          <h3 style={{ color: GENE_COLORS[gene] || '#a5b4fc', marginBottom: '0.5rem', fontSize: 14 }}>
            {gene} — {info.protein?.split('—')[1]?.trim() || info.gene_class}
          </h3>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>OMIM Gene: {info.omim_gene} · {info.omim_disease}</div>
          <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.7, marginBottom: 8 }}>
            {info.alias?.substring(0, 600)}{info.alias?.length > 600 ? '…' : ''}
          </p>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
            <strong style={{ color: '#94a3b8' }}>Key Etiologies:</strong>
          </div>
          {(info.etiologies || []).map((e, i) => (
            <div key={i} style={{
              fontSize: 11, color: '#94a3b8', padding: '3px 0 3px 12px',
              borderLeft: `2px solid ${GENE_COLORS[gene] || '#334155'}`,
              marginBottom: 3,
            }}>{e}</div>
          ))}
          <div style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>
            <strong style={{ color: '#94a3b8' }}>Dx Delay Pattern:</strong> {info.dx_delay_distribution}
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem', fontSize: 14 }}>Clinical Concepts</h3>
      {Object.entries(data.concepts || {}).map(([title, body]) => (
        <div key={title} style={{
          background: '#1e293b', borderRadius: 10, padding: '1.2rem', marginBottom: '1rem',
          borderLeft: '3px solid #6366f1',
        }}>
          <h4 style={{ color: '#a5b4fc', marginBottom: '0.6rem', fontSize: 13 }}>{title}</h4>
          <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.75 }}>{body}</p>
        </div>
      ))}
      <h3 style={{ color: '#f1f5f9', margin: '1.5rem 0 0.8rem', fontSize: 14 }}>Pharmacological Distinctions</h3>
      {(data.pharmacological_distinctions || []).map((d, i) => (
        <div key={i} style={{
          background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem', marginBottom: 8,
          borderLeft: '3px solid #f59e0b', fontSize: 12, color: '#94a3b8', lineHeight: 1.65,
        }}>{d}</div>
      ))}
      <h3 style={{ color: '#f1f5f9', margin: '1.5rem 0 0.8rem', fontSize: 14 }}>Key Clinical Standards</h3>
      {(data.key_standards || []).map((s, i) => (
        <div key={i} style={{
          background: '#1e293b', borderRadius: 8, padding: '0.6rem 0.9rem', marginBottom: 6,
          borderLeft: '3px solid #10b981', fontSize: 12, color: '#94a3b8',
        }}>{s}</div>
      ))}
    </div>
  );
}

export default function HereditaryPancreaticCancerAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-pancreatic-cancer-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-pancreatic-cancer-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-pancreatic-cancer-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  const style = {
    container: { background: '#0f172a', minHeight: '100vh', padding: '2rem', color: '#f1f5f9', fontFamily: 'system-ui, sans-serif' },
    header: { marginBottom: '1.5rem' },
    title: { fontSize: 22, fontWeight: 700, color: '#f1f5f9', margin: 0 },
    subtitle: { fontSize: 13, color: '#64748b', marginTop: 4 },
    tabBar: { display: 'flex', gap: 8, marginBottom: '1.5rem', flexWrap: 'wrap' },
    tab: (active) => ({
      padding: '6px 16px', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 500,
      background: active ? '#6366f1' : '#1e293b',
      color: active ? '#fff' : '#94a3b8',
      border: 'none',
    }),
  };

  return (
    <div style={style.container}>
      <div style={style.header}>
        <h1 style={style.title}>🧬 Hereditary-Pancreatic-Cancer-Atlas</h1>
        <div style={style.subtitle}>
          Complete 8-Gene Hereditary Pancreatic Cancer Reference · BRCA2 · CDKN2A · ATM · STK11 · BRCA1 · PALB2 · PRSS1 · MLH1 · 320 patients (8×40, seeds 1646–1653)
        </div>
      </div>
      {error && <ErrorBox msg={error} />}
      <div style={style.tabBar}>
        {TABS.map(t => (
          <button key={t} style={style.tab(tab === t)} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>
      {tab === 'Overview'      && <OverviewTab data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
    </div>
  );
}
