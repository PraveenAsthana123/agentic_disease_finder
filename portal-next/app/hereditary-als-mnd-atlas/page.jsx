'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SOD1:    '#b71c1c',  // deep red — A4V rapid, tofersen FDA2023
  TARDBP:  '#1565c0',  // deep blue — TDP-43 proteinopathy 97% sporadic
  FUS:     '#4a148c',  // deep purple — juvenile ALS P525L
  C9ORF72: '#e65100',  // deep orange — most common familial ALS/FTD
  UBQLN2:  '#1b5e20',  // dark green — X-linked dominant proteasomal
  VCP:     '#880e4f',  // dark pink — IBMPFD multisystem IBM+Paget+FTD
  OPTN:    '#006064',  // dark teal — autophagy receptor TBK1 pair
  TBK1:    '#37474f',  // dark slate — ALS+FTD kinase inhibitors CI
};

const GENE_DISEASE = {
  SOD1:    'AD/AR ALS1 — Cu/Zn Superoxide Dismutase — Toxic GOF Misfolded Aggregates — A4V Rapid <1y — D90A-hom Slow (AR) — Tofersen FDA Apr 2023 — ATLAS Pre-symptomatic Trial — ~20% Familial ALS',
  TARDBP:  'AD ALS10 — TDP-43 RNA-binding Protein — Cytoplasmic Inclusions PATHOGNOMONIC 97% Sporadic ALS — Nuclear Depletion Spliceopathy — A382T Sardinian FTD — No Approved TDP-43-Targeted Therapy',
  FUS:     'AD/AR ALS6 — Fused-in-Sarcoma NLS Mutations — Young Onset <40y — P525L Juvenile Lethal <20y — Basophilic Inclusions (NOT TDP-43) — De Novo in Juvenile — R521C/H Most Common Adult',
  C9ORF72: 'AD ALS-FTD — GGGGCC Repeat Expansion — MOST COMMON Familial ALS 40% + FTD 25% — Repeat-Primed PCR Mandatory — RNA Foci + DPR Proteins — Incomplete Penetrance 50% by 60y',
  UBQLN2:  'X-linked Dominant ALS15 — Ubiquilin-2 Proteasomal Shuttle — PXX Domain Mutations — Males Hemizygous More Severe — FTD 20% — Ubiquilin-2 Inclusions PATHOGNOMONIC — First X-linked ALS',
  VCP:     'AD ALS14/IBMPFD — AAA+ Segregase — IBM 90% + Paget 50% + FTD 30% + ALS 10% — R155H Most Common — TDP-43 Pathology — STEROIDS CI in IBM — Zoledronic Acid for Paget',
  OPTN:    'AD/AR ALS12 — Optineurin Autophagy Receptor — TBK1 Substrate (S177) — E478G Most Common — Q398X-AR Slowly Progressive — UBAN Ubiquitin Binding — MLPA for Deletions',
  TBK1:    'AD ALS-FTD — TANK-Binding Kinase 1 Haploinsufficiency — FTD 40% Highest After C9ORF72 — KINASE INHIBITORS ABSOLUTELY CI — Phosphorylates OPTN+p62 — Deletions Need MLPA — Innate Immunity',
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

function Alert({ text, color }) {
  return (
    <div style={{
      background: '#0f172a', border: `1px solid ${color || '#334155'}`,
      borderRadius: 8, padding: '0.65rem 1rem', marginBottom: 8,
      fontSize: 12, color: '#e2e8f0', lineHeight: 1.5,
    }}>
      <span style={{ color: color || '#f59e0b', fontWeight: 700, marginRight: 6 }}>⚠</span>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { atlas, subtitle, total_patients, seed_range, aggregate_stats, genes, top_alerts } = data;
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 4 }}>{atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 20 }}>{subtitle}</p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
        <KPI label="Total Patients" value={total_patients} color="#6366f1" />
        <KPI label="Genes Covered" value={aggregate_stats.genes_covered} color="#0ea5e9" />
        <KPI label="Patients / Gene" value={aggregate_stats.patients_per_gene} color="#10b981" />
        <KPI label="Mean Onset Age (yrs)" value={aggregate_stats.mean_dx_age} color="#f59e0b" />
        <KPI label="Mean Dx Delay (mo)" value={aggregate_stats.mean_dx_delay_months} color="#ef4444" />
        <KPI label="Seed Range" value={seed_range} color="#8b5cf6" />
      </div>

      <h3 style={{ color: '#f59e0b', marginBottom: 12 }}>Critical Clinical Alerts</h3>
      {top_alerts.map((a, i) => <Alert key={i} text={a} color={['#ef4444','#f59e0b','#10b981','#6366f1','#0ea5e9','#ec4899','#8b5cf6','#14b8a6'][i % 8]} />)}

      <h3 style={{ color: '#e2e8f0', marginTop: 28, marginBottom: 12 }}>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(360px,1fr))', gap: 14 }}>
        {genes.map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.aa} aa · {g.kDa} kDa</span>
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, lineHeight: 1.4 }}>{GENE_DISEASE[g.gene]}</div>
            <div style={{ display: 'flex', gap: 10, fontSize: 11, color: '#64748b' }}>
              <span>Onset Age: <b style={{ color: '#e2e8f0' }}>{g.mean_dx_age}y</b></span>
              <span>Delay: <b style={{ color: '#e2e8f0' }}>{g.mean_dx_delay_months}mo</b></span>
              <span>N: <b style={{ color: '#e2e8f0' }}>{g.n_patients}</b></span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>Gene Reference Table — All 8 ALS/MND Genes</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Gene','Protein (short)','Locus','aa','kDa','OMIM Gene','Inheritance','Gene Class',
                'Mean Onset Age','Mean Dx Delay','N Patients'].map(h => (
                <th key={h} style={{ padding: '8px 10px', color: '#94a3b8', textAlign: 'left',
                  borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#162032' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0', maxWidth: 220, fontSize: 11 }}>{g.protein.slice(0, 90)}&hellip;</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.aa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.kDa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.omim_gene}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 160, fontSize: 11 }}>{g.inheritance.slice(0, 80)}&hellip;</td>
                <td style={{ padding: '8px 10px', color: '#64748b', maxWidth: 200, fontSize: 11 }}>{g.gene_class.slice(0, 80)}&hellip;</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{g.computed.mean_dx_age}y</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{g.computed.mean_dx_delay_months}mo</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{g.computed.n_patients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const gene = selected ? data.find(g => g.gene === selected) : null;
  return (
    <div style={{ display: 'flex', gap: 16 }}>
      <div style={{ width: 160, flexShrink: 0 }}>
        {data.map(g => (
          <button key={g.gene}
            onClick={() => setSelected(g.gene)}
            style={{
              display: 'block', width: '100%', textAlign: 'left',
              padding: '8px 12px', marginBottom: 4, borderRadius: 6,
              background: selected === g.gene ? GENE_COLORS[g.gene] || '#6366f1' : '#1e293b',
              color: '#e2e8f0', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            }}>{g.gene}</button>
        ))}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        {!gene ? (
          <div style={{ color: '#64748b', padding: '2rem' }}>Select a gene to view full clinical profile.</div>
        ) : (
          <div>
            <h3 style={{ color: GENE_COLORS[gene.gene] || '#a5b4fc', marginBottom: 4 }}>{gene.gene}</h3>
            <p style={{ color: '#94a3b8', fontSize: 12, marginBottom: 12 }}>{gene.protein}</p>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
              <span style={{ background: '#0f172a', padding: '4px 10px', borderRadius: 5, fontSize: 11, color: '#94a3b8' }}>
                {gene.locus} · {gene.aa} aa · {gene.kDa} kDa
              </span>
              <span style={{ background: '#0f172a', padding: '4px 10px', borderRadius: 5, fontSize: 11, color: '#94a3b8' }}>
                OMIM: {gene.omim_gene}
              </span>
              <span style={{ background: '#0f172a', padding: '4px 10px', borderRadius: 5, fontSize: 11, color: '#94a3b8' }}>
                {gene.inheritance.slice(0, 60)}
              </span>
            </div>

            <h4 style={{ color: '#f59e0b', marginBottom: 8 }}>Key Clinical Alerts</h4>
            {gene.key_alerts.map((a, i) => <Alert key={i} text={a} color="#f59e0b" />)}

            <h4 style={{ color: '#e2e8f0', marginTop: 20, marginBottom: 8 }}>Disease Etiologies</h4>
            {gene.etiologies.map((e, i) => (
              <div key={i} style={{ background: '#0f172a', borderRadius: 6, padding: '0.65rem 1rem',
                marginBottom: 8, fontSize: 12, color: '#e2e8f0', lineHeight: 1.5 }}>
                <span style={{ color: '#6366f1', fontWeight: 700, marginRight: 6 }}>{i+1}.</span>{e}
              </div>
            ))}

            <h4 style={{ color: '#e2e8f0', marginTop: 20, marginBottom: 8 }}>Gene Function (Full)</h4>
            <div style={{ background: '#0f172a', borderRadius: 8, padding: '1rem',
              fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
              {gene.alias}
            </div>

            <h4 style={{ color: '#e2e8f0', marginTop: 20, marginBottom: 8 }}>Cohort Statistics</h4>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {Object.entries(gene.stats).map(([k, v]) => (
                <KPI key={k} label={k.replace(/_/g, ' ')} value={v} color={GENE_COLORS[gene.gene]} />
              ))}
            </div>

            <h4 style={{ color: '#e2e8f0', marginTop: 20, marginBottom: 8 }}>Sample Patients (first 10)</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#0f172a' }}>
                    {['ID','Onset Age','Dx Delay(mo)','Variant/Phenotype','FTD','Bulbar Onset'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#64748b', textAlign: 'left',
                        borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {gene.sample_patients.map((p, i) => (
                    <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#1e293b' : '#162032' }}>
                      <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0' }}>{p.onset_age}</td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0' }}>{p.dx_delay_months}</td>
                      <td style={{ padding: '6px 8px', color: '#94a3b8', fontSize: 10 }}>
                        {p.variant || p.phenotype || p.sex || '—'}
                      </td>
                      <td style={{ padding: '6px 8px', color: p.ftd || p.ftd_component ? '#ef4444' : '#64748b' }}>
                        {(p.ftd || p.ftd_component) ? 'Yes' : 'No'}
                      </td>
                      <td style={{ padding: '6px 8px', color: p.bulbar_onset ? '#f59e0b' : '#64748b' }}>
                        {p.bulbar_onset ? 'Yes' : 'No'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { concepts, pharmacological_distinctions, key_standards } = data;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 16 }}>Core Concepts</h3>
      {Object.entries(concepts).map(([title, text]) => (
        <div key={title} style={{ marginBottom: 20 }}>
          <h4 style={{ color: '#6366f1', marginBottom: 8, fontSize: 13 }}>{title}</h4>
          <div style={{ background: '#0f172a', borderRadius: 8, padding: '1rem',
            fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{text}</div>
        </div>
      ))}

      <h3 style={{ color: '#e2e8f0', marginTop: 28, marginBottom: 12 }}>Pharmacological Distinctions</h3>
      {pharmacological_distinctions.map((d, i) => (
        <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem',
          marginBottom: 10, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
          <span style={{ color: '#10b981', fontWeight: 700, marginRight: 6 }}>Rx {i+1}:</span>{d}
        </div>
      ))}

      <h3 style={{ color: '#e2e8f0', marginTop: 28, marginBottom: 12 }}>Key Standards &amp; Trials</h3>
      {key_standards.map((s, i) => (
        <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem',
          marginBottom: 10, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
          <span style={{ color: '#f59e0b', fontWeight: 700, marginRight: 6 }}>§{i+1}</span>{s}
        </div>
      ))}
    </div>
  );
}

export default function ALSMNDPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-als-mnd-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-als-mnd-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-als-mnd-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#0b1120', color: '#e2e8f0', padding: '2rem' }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
            🧬 Hereditary ALS / Motor Neuron Disease Atlas
          </h1>
          <p style={{ color: '#64748b', fontSize: 13 }}>
            Complete 8-Gene Hereditary ALS/MND Atlas — SOD1 / TARDBP / FUS / C9ORF72 / UBQLN2 / VCP / OPTN / TBK1 — 320 Patients (8×40, Seeds 1710–1717)
          </p>
        </div>

        {error && <ErrorBox msg={error} />}

        <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '1px solid #1e293b', paddingBottom: 8 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === t ? '#b71c1c' : '#1e293b',
              color: tab === t ? '#fff' : '#94a3b8', fontWeight: tab === t ? 700 : 400,
            }}>{t}</button>
          ))}
        </div>

        {tab === 'Overview'      && <OverviewTab data={overview} />}
        {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
        {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
