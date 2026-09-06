'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  VHL:   '#1565c0',  // deep blue — Belzutifan FDA 2021, most common hereditary RCC, 95% penetrance
  SDHB:  '#7b1fa2',  // deep purple — SDH4, aggressive PGL malignancy 40%, sunitinib preferred
  FH:    '#b71c1c',  // dark red — HLRCC most aggressive, any-size surgery, bevacizumab+erlotinib
  FLCN:  '#e65100',  // deep orange — Birt-Hogg-Dubé, fibrofolliculomas PATHOGNOMONIC, pneumothorax
  MET:   '#2e7d32',  // dark green — HPRCC type 1, pure renal, GOF, cabozantinib
  BAP1:  '#00695c',  // dark teal — BAP1-TPDS, uveal melanoma mandatory ophthalmology, tebentafusp
  SDHA:  '#4e342e',  // dark brown — SDH5, pituitary adenoma UNIQUE, rarest SDH-RCC
  PTEN:  '#37474f',  // dark slate — Cowden syndrome, 34% renal risk, everolimus FDA RECORD-1
};

const GENE_DISEASE = {
  VHL:   'HRCC-VHL1 AD — 95% Clear Cell RCC Penetrance — Belzutifan HIF-2α FDA 2021 — 3 cm Threshold — Annual MRI Brain+Spine+Abdomen+Ophthalmology',
  SDHB:  'HRCC-SDH4 AD — SDH-Deficient RCC Type 4 — 40% Malignant PGL — SDHB-IHC Loss Diagnostic — DOTATATE-PET Mandatory — Sunitinib Preferred',
  FH:    'HRCC-HLRCC AD — Type 2C RCC Most Aggressive — ANY SIZE Warrants Surgery — Bevacizumab+Erlotinib 65% ORR — Uterine Leiomyoma 95% Women — 2SC-IHC Surrogate',
  FLCN:  'HRCC-BHD AD — Birt-Hogg-Dubé — Hybrid Chromophobe/Oncocytoma 55–67% — Fibrofolliculomas PATHOGNOMONIC — Spontaneous Pneumothorax 25–33% — mTORC1',
  MET:   'HRCC-HPRCC1 AD-GOF — Hereditary Papillary RCC Type 1 — Pure Renal Phenotype — GOF Exons 16–19 — Cabozantinib/Savolitinib — Trisomy 7',
  BAP1:  'HRCC-BAP1-TPDS AD — Clear Cell RCC + Uveal Melanoma + Mesothelioma — ANNUAL OPHTHALMOLOGY MANDATORY — Tebentafusp FDA 2022 HLA-A*02:01+ UM',
  SDHA:  'HRCC-SDH5 AD/AR — Pituitary Adenoma UNIQUE — 1% Population Carrier — SDHA+SDHB Dual-IHC Specific — GIST Imatinib-Resistant — Rarest SDH-RCC',
  PTEN:  'HRCC-CS AD — Cowden Syndrome / PHTS — 34% Lifetime Renal Cancer — 85% Breast — Everolimus FDA RECORD-1 — Macrocephaly+Trichilemmomas PATHOGNOMONIC',
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
        <KPI label="Total Patients" value={data.total_patients} color="#6366f1" />
        <KPI label="Mean Dx Age (yr)" value={s.mean_dx_age} color="#06b6d4" />
        <KPI label="Mean Dx Delay (mo)" value={s.mean_dx_delay_months} color="#f59e0b" />
        <KPI label="Genes Covered" value={s.genes_covered} color="#1565c0" />
        <KPI label="VHL Lifetime Risk" value="95%" color="#1565c0" />
        <KPI label="HLRCC ORR Bev+Erl" value="65%" color="#b71c1c" />
        <KPI label="MET HPRCC Penetrance" value="~85%" color="#2e7d32" />
        <KPI label="PTEN Lifetime Renal" value="34%" color="#37474f" />
        <KPI label="SDHB Malignant PGL" value="40%" color="#7b1fa2" />
      </div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '0.8rem', fontSize: 14 }}>Top Clinical Alerts</h3>
      {(data.top_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
      <h3 style={{ color: '#f1f5f9', margin: '1.5rem 0 0.8rem', fontSize: 14 }}>Gene Summary</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Gene', 'Locus', 'aa / kDa', 'Inheritance', 'Lifetime RCC Risk', 'Mean Dx Age', 'N Patients'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data.genes || []).map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#0f172a' }}>
                <td style={{ padding: '6px 10px', color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700 }}>{g.gene}</td>
                <td style={{ padding: '6px 10px', color: '#cbd5e1' }}>{g.locus}</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.aa} aa / {g.kDa} kDa</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.inheritance?.split(';')[0]}</td>
                <td style={{ padding: '6px 10px', color: '#fcd34d', fontWeight: 600 }}>{g.lifetime_risk_pct}%</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.mean_dx_age} yr</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.n_patients}</td>
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
      {(data || []).map((g, i) => (
        <div key={g.gene} style={{
          background: '#1e293b', borderRadius: 10, marginBottom: 16, padding: '1.2rem',
          borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{g.locus} · {g.aa} aa · {g.kDa} kDa · OMIM {g.omim_gene}</div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{g.gene_class}</div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>
                <strong style={{ color: GENE_DISEASE[g.gene] ? GENE_COLORS[g.gene] : '#a5b4fc' }}>
                  {GENE_DISEASE[g.gene] || g.inheritance}
                </strong>
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>Lifetime RCC Risk: <strong style={{ color: '#fcd34d' }}>{g.stats?.lifetime_risk_pct}%</strong> · Renal RR: {g.stats?.renal_rr} · Mean Dx Age: {g.computed?.mean_dx_age} yr · Seed: {g.computed?.seed}</div>
            </div>
          </div>
          <div style={{ marginTop: '0.8rem' }}>
            <div style={{ fontSize: 12, color: '#f1f5f9', fontWeight: 600, marginBottom: 4 }}>Key Alerts</div>
            {(g.key_alerts || []).map((a, j) => <AlertBadge key={j} text={a} />)}
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
      {(data || []).map((g, i) => (
        <div key={g.gene} style={{
          background: '#0f172a', borderRadius: 10, marginBottom: 20, padding: '1.4rem',
          border: `1px solid ${GENE_COLORS[g.gene] || '#334155'}30`,
        }}>
          <div style={{ fontSize: 18, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc', marginBottom: 6 }}>
            {g.gene} — {g.omim_disease}
          </div>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 10 }}>{g.locus} · {g.aa} aa · {g.gene_class}</div>
          <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.65, marginBottom: 12 }}>
            {(g.alias || '').slice(0, 1200)}{g.alias?.length > 1200 ? '…' : ''}
          </div>
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 12, color: '#f1f5f9', fontWeight: 600, marginBottom: 4 }}>Molecular Etiologies</div>
            {(g.etiologies || []).map((e, j) => (
              <div key={j} style={{ fontSize: 12, color: '#94a3b8', padding: '3px 0 3px 10px', borderLeft: '2px solid #334155' }}>
                {e}
              </div>
            ))}
          </div>
          <div style={{ fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>
            Dx Delay Distribution: {g.dx_delay_distribution}
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
      <h3 style={{ color: '#f1f5f9', fontSize: 14, marginBottom: '1rem' }}>Clinical Concepts</h3>
      {Object.entries(data.concepts || {}).map(([title, body], i) => (
        <div key={i} style={{
          background: '#1e293b', borderRadius: 8, marginBottom: 14,
          padding: '1rem', borderLeft: '3px solid #6366f1',
        }}>
          <div style={{ color: '#a5b4fc', fontWeight: 700, fontSize: 13, marginBottom: 8 }}>{title}</div>
          <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.65 }}>{body}</div>
        </div>
      ))}
      <h3 style={{ color: '#f1f5f9', fontSize: 14, margin: '1.5rem 0 0.8rem' }}>Pharmacological Distinctions</h3>
      {(data.pharmacological_distinctions || []).map((d, i) => {
        const [first, ...rest] = d.split(':');
        return (
          <div key={i} style={{
            background: '#0f172a', borderRadius: 8, marginBottom: 10,
            padding: '0.8rem', borderLeft: '3px solid #06b6d4',
          }}>
            <div style={{ color: '#67e8f9', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>{first}</div>
            <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{rest.join(':')}</div>
          </div>
        );
      })}
      <h3 style={{ color: '#f1f5f9', fontSize: 14, margin: '1.5rem 0 0.8rem' }}>Key Standards</h3>
      {(data.key_standards || []).map((s, i) => (
        <div key={i} style={{
          background: '#1e293b', borderRadius: 8, marginBottom: 8,
          padding: '0.7rem', borderLeft: '3px solid #10b981',
          fontSize: 12, color: '#94a3b8', lineHeight: 1.6,
        }}>
          {s}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryRenalCancerAtlas() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-renal-cancer-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-renal-cancer-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-renal-cancer-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'monospace', padding: '1.5rem' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#a5b4fc', marginBottom: 4 }}>
          &#x1f9ec; Hereditary-Renal-Cancer-Atlas
        </h1>
        <p style={{ fontSize: 12, color: '#64748b', marginBottom: '1.5rem' }}>
          Complete 8-Gene Hereditary Renal Cell Carcinoma Atlas · VHL · SDHB · FH · FLCN · MET · BAP1 · SDHA · PTEN · 320 patients (8×40, seeds 1654–1661)
        </p>
        {error && <ErrorBox msg={error} />}
        <div style={{ display: 'flex', gap: 8, marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          {TABS.map((t, i) => (
            <button key={t} onClick={() => setTab(i)} style={{
              padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === i ? '#6366f1' : '#1e293b',
              color: tab === i ? '#fff' : '#94a3b8',
              fontWeight: tab === i ? 700 : 400,
            }}>{t}</button>
          ))}
        </div>
        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <GeneTableTab data={breakdown} />}
        {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
        {tab === 3 && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
