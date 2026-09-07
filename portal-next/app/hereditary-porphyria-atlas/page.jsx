'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  HMBS:  '#7c3aed',  // violet — AIP most common acute porphyria / port-wine urine
  CPOX:  '#1d4ed8',  // indigo — HCP dual acute+cutaneous / faecal coproporphyrin
  PPOX:  '#b45309',  // amber — VP R59W South Africa founder / plasma 626nm
  UROS:  '#dc2626',  // red — CEP Günther disease / erythrodontia / HSCT curative
  FECH:  '#c2410c',  // deep orange — EPP burning pain in sunlight / afamelanotide FDA2019
  ALAS2: '#0369a1',  // sky blue — XLP GOF / XLSA LOF / dual phenotype
  UROD:  '#166534',  // dark green — PCT most common porphyria / phlebotomy + HCQ
  ALAD:  '#065f46',  // deep teal — ADP rarest porphyria / AR / lead DDx
};

const GENE_DISEASE = {
  HMBS:  'AD AIP-Acute-Intermittent-Porphyria — Hydroxymethylbilane-Synthase — 361aa — 11q23.3 — Most-Common-Acute-Porphyria — Neurovisceral-Attacks — Port-Wine-Urine — >300-Drug-Triggers-Check-EPNET — Hemin-IV-First-Line-Acute — Givosiran-siRNA-FDA2019-Monthly-Prevention',
  CPOX:  'AD HCP-Hereditary-Coproporphyria — Coproporphyrinogen-Oxidase — 454aa — 3q11.2 — Acute-AND-Cutaneous-Blistering-30pct — Faecal-Coproporphyrin-III-KEY-DDx-from-AIP — Plasma-Fluorescence-619nm — Same-Drug-Triggers-AIP',
  PPOX:  'AD VP-Variegate-Porphyria — Protoporphyrinogen-Oxidase — 477aa — 1q23.3 — R59W-South-Africa-1in300-Afrikaner-Founder — Acute-PLUS-Skin-Prominent — Plasma-626nm-Stable-Inter-Attack-PATHOGNOMONIC — Etomidate-Thiopental-CONTRAINDICATED',
  UROS:  'AR CEP-Congenital-Erythropoietic-Porphyria-Günther — Uroporphyrinogen-III-Synthase — 265aa — 10q25.2 — Pink-Red-Urine-Nappies-PATHOGNOMONIC — Erythrodontia-UV-Fluorescent-PATHOGNOMONIC — Severe-Mutilating-Blistering — HSCT-ONLY-Curative-Treatment — NO-Neurovisceral-Attacks',
  FECH:  'AR/QTL EPP-Erythropoietic-Protoporphyria — Ferrochelatase — 423aa — 18q21.31 — Burning-Pain-Within-Minutes-Sunlight-PATHOGNOMONIC-NOT-Blistering — Urine-Porphyrins-NORMAL-KEY-DDx — Afamelanotide-SC-FDA2019 — Liver-Failure-2-5pct-Annual-LFTs-Mandatory',
  ALAS2: 'X-linked XLP-GOF-XLinkProtoporphyria / XLSA-LOF-SideroblasticAnaemia — ALA-Synthase-2 — 587aa — Xp11.21 — GOF-Exon11-Identical-EPP-Burning-Pain — LOF-Ring-Sideroblasts-Elevated-Serum-Iron-Pyridoxine-B6 — Afamelanotide-Effective-XLP — Annual-Liver-LFTs-XLP',
  UROD:  'AD/Acquired PCT-Porphyria-Cutanea-Tarda — Uroporphyrinogen-Decarboxylase — 367aa — 1p34.1 — MOST-COMMON-Porphyria-Worldwide — Blistering-Dorsal-Hands-Classic — HFE-C282Y-Strong-Cofactor — HCV-HIV-Alcohol-Oestrogen-Triggers — Phlebotomy-LowDose-HCQ-First-Line — NO-Neurovisceral-Attacks',
  ALAD:  'AR ADP-ALA-Dehydratase-Deficiency — ALA-Dehydratase — 330aa — 9q32 — RAREST-Human-Porphyria-<10-Cases-Worldwide — AR-Unique-Among-Acute-Porphyrias — Elevated-ALA-NORMAL-PBG-KEY-DDx — Lead-Inhibits-Enzyme-Exclude-Lead-FIRST — Hemin-IV-Acute',
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
        <KPI label="Genes Covered" value={aggregate_stats?.genes_covered ?? 8} color="#0ea5e9" />
        <KPI label="Patients / Gene" value={aggregate_stats?.patients_per_gene ?? 40} color="#10b981" />
        <KPI label="Mean Dx Age (yrs)" value={aggregate_stats?.mean_dx_age ?? '—'} color="#f59e0b" />
        <KPI label="Mean Dx Delay (mo)" value={aggregate_stats?.mean_dx_delay_months ?? '—'} color="#ef4444" />
        <KPI label="Seed Range" value={seed_range} color="#8b5cf6" />
      </div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>8 Genes — Locus · Size · Cohort</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Locus', 'aa', 'kDa', 'n', 'Mean Dx Age (y)', 'Mean Dx Delay (mo)'].map(h => (
                <th key={h} style={{ padding: '8px 12px', color: '#94a3b8', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(genes || []).map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 ? '#0f172a' : '#111827' }}>
                <td style={{ padding: '8px 12px', color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700 }}>{g.gene}</td>
                <td style={{ padding: '8px 12px', color: '#e2e8f0' }}>{g.locus}</td>
                <td style={{ padding: '8px 12px', color: '#e2e8f0' }}>{(g.aa || 0).toLocaleString()}</td>
                <td style={{ padding: '8px 12px', color: '#e2e8f0' }}>{g.kDa ?? '—'}</td>
                <td style={{ padding: '8px 12px', color: '#e2e8f0' }}>{g.n_patients ?? '40'}</td>
                <td style={{ padding: '8px 12px', color: '#fbbf24' }}>{g.mean_dx_age ?? '—'}</td>
                <td style={{ padding: '8px 12px', color: '#f87171' }}>{g.mean_dx_delay_months ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {top_alerts && top_alerts.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>Top Clinical Alerts (1 per gene)</h3>
          {top_alerts.map((a, i) => (
            <Alert key={i} text={typeof a === 'string' ? a : (a.alert || a.text || JSON.stringify(a))}
              color={Object.values(GENE_COLORS)[i % Object.keys(GENE_COLORS).length]} />
          ))}
        </div>
      )}
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 16 }}>Gene × Disease Matrix</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {Object.entries(GENE_DISEASE).map(([gene, desc]) => (
          <div key={gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderLeft: `4px solid ${GENE_COLORS[gene] || '#6366f1'}`,
          }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc', fontSize: 15, marginBottom: 6 }}>{gene}</div>
            <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 }}>{desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [expanded, setExpanded] = useState(null);
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 16 }}>Clinical Atlas — Per-Gene Detail</h3>
      {data.map((g) => (
        <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, marginBottom: 16, overflow: 'hidden' }}>
          <div
            onClick={() => setExpanded(expanded === g.gene ? null : g.gene)}
            style={{
              padding: '1rem 1.2rem', cursor: 'pointer', display: 'flex', alignItems: 'center',
              borderBottom: expanded === g.gene ? '1px solid #334155' : 'none',
            }}
          >
            <span style={{ color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700, fontSize: 16, minWidth: 100 }}>{g.gene}</span>
            <span style={{ color: '#94a3b8', fontSize: 12, flex: 1, marginLeft: 12 }}>
              {g.locus} · {(g.aa || 0).toLocaleString()} aa · {g.kDa ?? '—'} kDa · {(g.inheritance || '').split(';')[0]}
            </span>
            <span style={{ color: '#475569', fontSize: 18 }}>{expanded === g.gene ? '▲' : '▼'}</span>
          </div>
          {expanded === g.gene && (
            <div style={{ padding: '1rem 1.2rem' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 12, lineHeight: 1.5 }}>{g.alias}</div>
              <div style={{ marginBottom: 12 }}>
                <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 6, fontSize: 13 }}>Key Clinical Alerts</div>
                {(g.key_alerts || []).map((a, i) => <Alert key={i} text={a} color={GENE_COLORS[g.gene]} />)}
              </div>
              <div style={{ marginBottom: 12 }}>
                <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 6, fontSize: 13 }}>Etiological Subtypes</div>
                {Array.isArray(g.etiologies) ? (
                  <ul style={{ margin: 0, paddingLeft: 20, color: '#cbd5e1', fontSize: 12, lineHeight: 1.7 }}>
                    {g.etiologies.map((e, i) => <li key={i}>{typeof e === 'string' ? e : JSON.stringify(e)}</li>)}
                  </ul>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {Object.entries(g.etiologies || {}).map(([key, val]) => (
                      <div key={key} style={{ background: '#0f172a', borderRadius: 6, padding: '0.5rem 0.75rem' }}>
                        <span style={{ color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 600, fontSize: 11, marginRight: 8 }}>
                          {key.replace(/_/g, ' ')}
                        </span>
                        {typeof val === 'object' ? (
                          <span style={{ color: '#94a3b8', fontSize: 11 }}>
                            {val.pct != null ? `${val.pct}%` : ''}{val.phenotype ? ` · ${val.phenotype}` : ''}{val.notes ? ` — ${val.notes}` : ''}
                          </span>
                        ) : (
                          <span style={{ color: '#94a3b8', fontSize: 11 }}>{val}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {g.sample_patients && g.sample_patients.length > 0 && (
                <div>
                  <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 6, fontSize: 13 }}>Sample Patients (10 of 40)</div>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                      <thead>
                        <tr style={{ background: '#0f172a' }}>
                          <th style={{ padding: '6px 10px', color: '#64748b', textAlign: 'left' }}>ID</th>
                          <th style={{ padding: '6px 10px', color: '#64748b', textAlign: 'left' }}>Onset Age (y)</th>
                          <th style={{ padding: '6px 10px', color: '#64748b', textAlign: 'left' }}>Dx Delay (mo)</th>
                          <th style={{ padding: '6px 10px', color: '#64748b', textAlign: 'left' }}>Phenotype</th>
                        </tr>
                      </thead>
                      <tbody>
                        {g.sample_patients.map((p) => (
                          <tr key={p.patient_id || p.id} style={{ borderBottom: '1px solid #1e293b' }}>
                            <td style={{ padding: '5px 10px', color: '#94a3b8', fontFamily: 'monospace' }}>{p.patient_id || p.id}</td>
                            <td style={{ padding: '5px 10px', color: '#fbbf24' }}>{p.onset_age ?? '—'}</td>
                            <td style={{ padding: '5px 10px', color: '#f87171' }}>{p.dx_delay_months ?? '—'}</td>
                            <td style={{ padding: '5px 10px', color: '#34d399' }}>{p.phenotype ?? p.gene ?? '—'}</td>
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
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { concepts, pharmacological_distinctions, key_standards } = data;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 16 }}>Core Concepts</h3>
      {Object.entries(concepts || {}).map(([title, body]) => (
        <div key={title} style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: 12 }}>
          <div style={{ color: '#a5b4fc', fontWeight: 700, marginBottom: 6, fontSize: 13 }}>{title}</div>
          <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6 }}>{body}</div>
        </div>
      ))}
      <h3 style={{ color: '#e2e8f0', marginTop: 24, marginBottom: 12 }}>Pharmacological Distinctions</h3>
      <ul style={{ margin: 0, paddingLeft: 20, color: '#cbd5e1', fontSize: 12, lineHeight: 1.8 }}>
        {(pharmacological_distinctions || []).map((p, i) => <li key={i}>{p}</li>)}
      </ul>
      <h3 style={{ color: '#e2e8f0', marginTop: 24, marginBottom: 12 }}>Key Standards</h3>
      <ul style={{ margin: 0, paddingLeft: 20, color: '#cbd5e1', fontSize: 12, lineHeight: 1.8 }}>
        {(key_standards || []).map((s, i) => <li key={i}>{s}</li>)}
      </ul>
    </div>
  );
}

export default function HPorphyriaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-porphyria-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 800, margin: 0 }}>
            🧪 Hereditary Porphyria Atlas
          </h1>
          <p style={{ color: '#64748b', fontSize: 13, margin: '4px 0 0' }}>
            HMBS · CPOX · PPOX · UROS · FECH · ALAS2 · UROD · ALAD — 320 Patients (8×40, Seeds 1758–1765)
          </p>
        </div>

        {error && <ErrorBox msg={error} />}

        <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
          {TABS.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
                background: tab === t ? '#7c3aed' : '#1e293b',
                color: tab === t ? '#fff' : '#94a3b8', fontWeight: tab === t ? 700 : 400,
                fontSize: 13,
              }}
            >{t}</button>
          ))}
        </div>

        <div style={{ background: '#111827', borderRadius: 12, padding: '1.5rem' }}>
          {tab === 'Overview' && <OverviewTab data={overview} />}
          {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
          {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
          {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
        </div>
      </div>
    </div>
  );
}
