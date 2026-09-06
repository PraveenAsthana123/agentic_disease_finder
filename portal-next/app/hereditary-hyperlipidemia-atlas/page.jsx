'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  LDLR:    '#1565c0',  // deep blue — FH1, most common, 1:300 HeFH, LDL apheresis HoFH
  APOB:    '#6a1b9a',  // deep purple — FDB, R3527Q, milder FH, statins effective
  PCSK9:   '#b71c1c',  // dark red — FH3 GOF, most severe; LOF 88% CVD reduction
  LDLRAP1: '#e65100',  // deep orange — ARH biallelic, HoFH-equivalent, PCSK9i paradox
  ABCG5:   '#2e7d32',  // dark green — Sitosterolemia1, plant sterols, ezetimibe curative
  ABCG8:   '#00695c',  // dark teal — Sitosterolemia2, more common STSL, D19H gallstones
  APOE:    '#4e342e',  // dark brown — FD Type III, palmar xanthomas PATHOGNOMONIC, fibrates
  LPA:     '#37474f',  // dark slate — Lp(a) elevation, no FDA Rx yet, pelacarsen Phase 3
};

const GENE_DISEASE = {
  LDLR:    'Familial Hypercholesterolemia type 1 (FH1) AD — LDL Receptor — 1:300 HeFH Most Common — LDL 190–500 mg/dL — Statins+Ezetimibe+PCSK9i — LDL Apheresis HoFH — Cascade Test Age 2–5',
  APOB:    'Familial Defective ApoB-100 (FDB/FH2) AD — R3527Q LDLR-Binding Impaired — LDL 200–350 mg/dL Milder — Statins MORE Effective Than LDLR FH — Statins CI Pregnancy (Category X)',
  PCSK9:   'FH3 GOF D374Y Most Severe AD — LOF R46L/Y142X Protective 88% CVD Reduction — Evolocumab/Alirocumab 50–60% LDL Reduction — FOURIER HR 0.85 — Very Low LDL Safe',
  LDLRAP1: 'Autosomal Recessive Hypercholesterolemia (ARH) — Clathrin Adaptor Biallelic — LDL 400–900 mg/dL HoFH-Equivalent — Parents Unaffected (KEY DDx HoFH) — PCSK9i Paradox Effective',
  ABCG5:   'Sitosterolemia type 1 (STSL1) AR Biallelic — Sterolin-1 Plant-Sterol Pump — Xanthomas + NORMAL LDL (KEY DDx FH) — Haemolysis Stomatocytes — Ezetimibe CURATIVE — Statins Ineffective',
  ABCG8:   'Sitosterolemia type 2 (STSL2) AR Biallelic — Sterolin-2 More Common STSL Gene — D19H Heterozygous Gallstones NOT Sitosterolemia — Ezetimibe CURATIVE — Low Plant-Sterol Diet',
  APOE:    'Familial Dysbetalipoproteinemia (FD) / Type III AR-like — APOE2/E2 + Second Hit — BOTH TG AND LDL Elevated — Palmar Xanthomas PATHOGNOMONIC — Fibrates First-Line — Treat Second Hit First',
  LPA:     'Lp(a) Elevation Co-Dominant — >50 mg/dL 20–25% Population — Independent CVD Risk — No FDA-Approved Rx Yet — Pelacarsen HORIZON Phase 3 80% Reduction — Measure Once All Adults',
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
  const parts = text.split(': ');
  const key = parts[0];
  const rest = parts.slice(1).join(': ');
  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: '0.6rem 0.9rem',
      borderLeft: '3px solid #f59e0b', marginBottom: 6, fontSize: 12,
    }}>
      <span style={{ color: '#fbbf24', fontWeight: 700 }}>{key}</span>
      {rest && <span style={{ color: '#cbd5e1' }}>: {rest}</span>}
    </div>
  );
}

function GeneCard({ gene, expanded, onToggle }) {
  const color = GENE_COLORS[gene.gene] || '#6366f1';
  return (
    <div style={{ background: '#0f172a', borderRadius: 10, marginBottom: 10, border: `1px solid ${color}40` }}>
      <button
        onClick={onToggle}
        style={{
          width: '100%', textAlign: 'left', padding: '0.9rem 1.1rem',
          background: 'none', border: 'none', cursor: 'pointer', color: '#e2e8f0',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span style={{
            background: color, color: '#fff', borderRadius: 6,
            padding: '2px 10px', fontWeight: 700, fontSize: 13, minWidth: 80,
          }}>{gene.gene}</span>
          <span style={{ color: '#94a3b8', fontSize: 11 }}>{gene.locus} · {gene.aa} aa · {gene.inheritance?.split('—')[0]?.trim()}</span>
          <span style={{ marginLeft: 'auto', color: '#64748b', fontSize: 11 }}>
            {expanded ? '▲' : '▼'} {gene.n_patients} patients
          </span>
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 6 }}>
          {GENE_DISEASE[gene.gene] || gene.protein}
        </div>
      </button>
      {expanded && (
        <div style={{ padding: '0 1.1rem 1rem' }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
            <KPI label="Mean Dx Age" value={`${gene.mean_dx_age}y`} color={color} />
            <KPI label="Mean Dx Delay" value={`${gene.mean_dx_delay_months}m`} color={color} />
            <KPI label="OMIM Gene" value={gene.omim_gene} color={color} />
            <KPI label="Patients" value={gene.n_patients} color={color} />
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>
            <strong style={{ color: '#e2e8f0' }}>Class:</strong> {gene.gene_class}
          </div>
          <div>
            {(gene.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
          </div>
        </div>
      )}
    </div>
  );
}

function OverviewTab({ data }) {
  const [expanded, setExpanded] = useState({});
  if (!data) return <Loading />;
  const toggle = g => setExpanded(e => ({ ...e, [g]: !e[g] }));
  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ color: '#e2e8f0', fontSize: 18, marginBottom: 4 }}>{data.atlas}</h2>
        <div style={{ color: '#94a3b8', fontSize: 13 }}>{data.subtitle}</div>
        <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>Seeds {data.seed_range} · {data.total_patients} patients</div>
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
        <KPI label="Total Patients" value={data.total_patients} color="#6366f1" />
        <KPI label="Genes Covered" value={data.aggregate_stats?.genes_covered} color="#10b981" />
        <KPI label="Patients / Gene" value={data.aggregate_stats?.patients_per_gene} color="#f59e0b" />
        <KPI label="Mean Dx Age" value={`${data.aggregate_stats?.mean_dx_age}y`} color="#06b6d4" />
        <KPI label="Mean Dx Delay" value={`${data.aggregate_stats?.mean_dx_delay_months}m`} color="#ec4899" />
      </div>
      {(data.top_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
      <h3 style={{ color: '#cbd5e1', fontSize: 14, marginTop: 18, marginBottom: 10 }}>Gene Atlas (click to expand)</h3>
      {(data.genes || []).map(g => (
        <GeneCard key={g.gene} gene={g} expanded={!!expanded[g.gene]} onToggle={() => toggle(g.gene)} />
      ))}
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#1e293b' }}>
            {['Gene', 'Locus', 'aa', 'Inheritance', 'Disease', 'Mean Dx Age', 'Mean Dx Delay', 'N Patients'].map(h => (
              <th key={h} style={{ padding: '8px 10px', color: '#94a3b8', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(data.genes || []).map((g, i) => (
            <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
              <td style={{ padding: '7px 10px' }}>
                <span style={{
                  background: GENE_COLORS[g.gene] || '#6366f1',
                  color: '#fff', borderRadius: 4, padding: '1px 8px', fontWeight: 700, fontSize: 11,
                }}>{g.gene}</span>
              </td>
              <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
              <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.aa}</td>
              <td style={{ padding: '7px 10px', color: '#94a3b8', maxWidth: 180, fontSize: 11 }}>{g.inheritance}</td>
              <td style={{ padding: '7px 10px', color: '#e2e8f0', maxWidth: 260, fontSize: 11 }}>{GENE_DISEASE[g.gene]?.split('—')[0]?.trim()}</td>
              <td style={{ padding: '7px 10px', color: '#a5b4fc', textAlign: 'center' }}>{g.mean_dx_age}y</td>
              <td style={{ padding: '7px 10px', color: '#fbbf24', textAlign: 'center' }}>{g.mean_dx_delay_months}m</td>
              <td style={{ padding: '7px 10px', color: '#6ee7b7', textAlign: 'center' }}>{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [sel, setSel] = useState(null);
  if (!data) return <Loading />;
  const genes = data || [];
  const gene = sel ? genes.find(g => g.gene === sel) : null;
  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
      <div style={{ minWidth: 160 }}>
        {genes.map(g => (
          <button
            key={g.gene}
            onClick={() => setSel(g.gene === sel ? null : g.gene)}
            style={{
              display: 'block', width: '100%', textAlign: 'left',
              padding: '6px 12px', marginBottom: 4, borderRadius: 6, border: 'none',
              background: sel === g.gene ? (GENE_COLORS[g.gene] || '#6366f1') : '#1e293b',
              color: sel === g.gene ? '#fff' : '#94a3b8', cursor: 'pointer', fontWeight: 700, fontSize: 13,
            }}
          >{g.gene}</button>
        ))}
      </div>
      <div style={{ flex: 1, minWidth: 280 }}>
        {!gene && (
          <div style={{ color: '#64748b', fontSize: 13, padding: '2rem' }}>
            Select a gene to view its full clinical atlas entry.
          </div>
        )}
        {gene && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
              <span style={{
                background: GENE_COLORS[gene.gene] || '#6366f1',
                color: '#fff', borderRadius: 6, padding: '3px 14px', fontWeight: 700, fontSize: 15,
              }}>{gene.gene}</span>
              <span style={{ color: '#94a3b8', fontSize: 12 }}>{gene.locus} · {gene.aa} aa · {gene.kDa} kDa</span>
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>
              <strong style={{ color: '#e2e8f0' }}>OMIM Gene:</strong> {gene.omim_gene} &nbsp;|&nbsp;
              <strong style={{ color: '#e2e8f0' }}>OMIM Disease:</strong> {gene.omim_disease}
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>
              <strong style={{ color: '#e2e8f0' }}>Inheritance:</strong> {gene.inheritance}
            </div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10 }}>
              <strong style={{ color: '#e2e8f0' }}>Gene Class:</strong> {gene.gene_class}
            </div>
            <div style={{ marginBottom: 12 }}>
              <strong style={{ color: '#e2e8f0', fontSize: 12 }}>Key Alerts:</strong>
              {(gene.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
            </div>
            <div style={{ marginBottom: 12 }}>
              <strong style={{ color: '#e2e8f0', fontSize: 12 }}>Etiologies:</strong>
              {(gene.etiologies || []).map((e, i) => (
                <div key={i} style={{
                  background: '#1e293b', borderRadius: 6, padding: '5px 10px', marginBottom: 4,
                  fontSize: 12, color: '#cbd5e1', borderLeft: '3px solid #6366f1',
                }}>{e}</div>
              ))}
            </div>
            {gene.stats && (
              <div style={{ marginBottom: 12 }}>
                <strong style={{ color: '#e2e8f0', fontSize: 12 }}>Statistics:</strong>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 6 }}>
                  {Object.entries(gene.stats).map(([k, v]) => (
                    <div key={k} style={{
                      background: '#1e293b', borderRadius: 6, padding: '4px 10px', fontSize: 11,
                    }}>
                      <span style={{ color: '#94a3b8' }}>{k.replace(/_/g, ' ')}: </span>
                      <span style={{ color: '#a5b4fc', fontWeight: 600 }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div style={{ marginBottom: 12 }}>
              <strong style={{ color: '#e2e8f0', fontSize: 12 }}>Dx Delay Distribution:</strong>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>{gene.dx_delay_distribution}</div>
            </div>
            {gene.alias && (
              <details style={{ marginTop: 10 }}>
                <summary style={{ color: '#6366f1', cursor: 'pointer', fontSize: 12 }}>Full Molecular Reference</summary>
                <div style={{
                  background: '#0f172a', borderRadius: 8, padding: '0.8rem', marginTop: 6,
                  fontSize: 11, color: '#94a3b8', maxHeight: 400, overflowY: 'auto', whiteSpace: 'pre-wrap',
                }}>{gene.alias}</div>
              </details>
            )}
            {gene.sample_patients && gene.sample_patients.length > 0 && (
              <div style={{ marginTop: 12 }}>
                <strong style={{ color: '#e2e8f0', fontSize: 12 }}>Sample Patients (first 10):</strong>
                <div style={{ overflowX: 'auto', marginTop: 6 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ background: '#1e293b' }}>
                        {['ID', 'Age Dx', 'Delay(m)', 'LDL-C', 'TG', 'Lp(a)', 'Statin', 'PCSK9i', 'EZE', 'Xanthomas', 'ASCVD'].map(h => (
                          <th key={h} style={{ padding: '5px 8px', color: '#64748b', textAlign: 'center' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {gene.sample_patients.map((p, i) => (
                        <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                          <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                          <td style={{ padding: '4px 8px', color: '#a5b4fc', textAlign: 'center' }}>{p.age_at_dx}</td>
                          <td style={{ padding: '4px 8px', color: '#fbbf24', textAlign: 'center' }}>{p.dx_delay_months}</td>
                          <td style={{ padding: '4px 8px', color: '#f87171', textAlign: 'center' }}>{p.ldl_c_untreated_mgdL}</td>
                          <td style={{ padding: '4px 8px', color: '#fb923c', textAlign: 'center' }}>{p.tg_mgdL}</td>
                          <td style={{ padding: '4px 8px', color: '#e879f9', textAlign: 'center' }}>{p.lpa_mgdL}</td>
                          <td style={{ padding: '4px 8px', textAlign: 'center', color: p.on_statin ? '#4ade80' : '#f87171' }}>{p.on_statin ? '✓' : '✗'}</td>
                          <td style={{ padding: '4px 8px', textAlign: 'center', color: p.on_pcsk9i ? '#4ade80' : '#f87171' }}>{p.on_pcsk9i ? '✓' : '✗'}</td>
                          <td style={{ padding: '4px 8px', textAlign: 'center', color: p.on_ezetimibe ? '#4ade80' : '#f87171' }}>{p.on_ezetimibe ? '✓' : '✗'}</td>
                          <td style={{ padding: '4px 8px', textAlign: 'center', color: p.xanthomas_present ? '#fb923c' : '#64748b' }}>{p.xanthomas_present ? '✓' : '–'}</td>
                          <td style={{ padding: '4px 8px', textAlign: 'center', color: p.ascvd_event_prior ? '#f87171' : '#64748b' }}>{p.ascvd_event_prior ? '✓' : '–'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  const [openConcept, setOpenConcept] = useState(null);
  if (!data) return <Loading />;
  const concepts = data.concepts || {};
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', fontSize: 14, marginBottom: 12 }}>Core Concepts</h3>
      {Object.entries(concepts).map(([title, text]) => (
        <div key={title} style={{ marginBottom: 8 }}>
          <button
            onClick={() => setOpenConcept(openConcept === title ? null : title)}
            style={{
              width: '100%', textAlign: 'left', padding: '8px 12px',
              background: '#1e293b', border: 'none', borderRadius: 8, color: '#a5b4fc',
              cursor: 'pointer', fontWeight: 600, fontSize: 12,
            }}
          >
            {openConcept === title ? '▼' : '▶'} {title}
          </button>
          {openConcept === title && (
            <div style={{
              background: '#0f172a', padding: '0.8rem 1rem', borderRadius: '0 0 8px 8px',
              fontSize: 12, color: '#94a3b8', whiteSpace: 'pre-wrap', lineHeight: 1.7,
            }}>{text}</div>
          )}
        </div>
      ))}
      {data.pharmacological_distinctions && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ color: '#e2e8f0', fontSize: 14, marginBottom: 10 }}>Pharmacological Distinctions</h3>
          {data.pharmacological_distinctions.map((d, i) => (
            <div key={i} style={{
              background: '#1e293b', borderRadius: 8, padding: '8px 12px', marginBottom: 6,
              borderLeft: '3px solid #10b981', fontSize: 12, color: '#cbd5e1',
            }}>{d}</div>
          ))}
        </div>
      )}
      {data.key_standards && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ color: '#e2e8f0', fontSize: 14, marginBottom: 10 }}>Key Standards & References</h3>
          {data.key_standards.map((s, i) => (
            <div key={i} style={{
              background: '#1e293b', borderRadius: 8, padding: '7px 12px', marginBottom: 5,
              borderLeft: '3px solid #6366f1', fontSize: 12, color: '#94a3b8',
            }}>{s}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function HereditaryHyperlipidemiaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-hyperlipidemia-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setErr(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Gene Table' || tab === 'Clinical Atlas') {
      if (!breakdown) {
        fetch(`${API}/api/hereditary-hyperlipidemia-atlas/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setErr(e.message));
      }
    }
    if (tab === 'Definitions') {
      if (!definitions) {
        fetch(`${API}/api/hereditary-hyperlipidemia-atlas/definitions`)
          .then(r => r.json()).then(setDefinitions).catch(e => setErr(e.message));
      }
    }
  }, [tab]);

  return (
    <div style={{
      minHeight: '100vh', background: '#020617', color: '#e2e8f0',
      fontFamily: 'Inter, system-ui, sans-serif', padding: '1.5rem',
    }}>
      <div style={{
        background: 'linear-gradient(135deg, #1565c0 0%, #6a1b9a 50%, #b71c1c 100%)',
        borderRadius: 12, padding: '1.2rem 1.5rem', marginBottom: 20,
      }}>
        <div style={{ fontSize: 20, fontWeight: 800, color: '#fff' }}>
          🧬 Hereditary-Hyperlipidemia-Atlas
        </div>
        <div style={{ fontSize: 12, color: '#ddd6fe', marginTop: 4 }}>
          Complete 8-Gene Hereditary Hyperlipidemia Atlas — LDLR · APOB · PCSK9 · LDLRAP1 · ABCG5 · ABCG8 · APOE · LPA
        </div>
        <div style={{ fontSize: 11, color: '#bfdbfe', marginTop: 2 }}>
          Monogenic Dyslipidaemia (FH/FDB/FH3/ARH/Sitosterolemia/FD/Lp(a)) · 320 Patients · Seeds 1670–1677
        </div>
      </div>
      {err && <ErrorBox msg={err} />}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '6px 16px', borderRadius: 20, border: 'none', cursor: 'pointer',
              background: tab === t ? '#6366f1' : '#1e293b',
              color: tab === t ? '#fff' : '#94a3b8',
              fontWeight: tab === t ? 700 : 400, fontSize: 13,
            }}
          >{t}</button>
        ))}
      </div>
      <div>
        {tab === 'Overview' && <OverviewTab data={overview} />}
        {tab === 'Gene Table' && <GeneTableTab data={breakdown ? { genes: breakdown.map ? breakdown : [] } : null} />}
        {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
