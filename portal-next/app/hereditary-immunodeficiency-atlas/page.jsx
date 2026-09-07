'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BTK:       '#1565c0',  // deep blue  — XLA, X-linked agammaglobulinemia
  ADA:       '#e65100',  // deep orange — ADA-SCID, dATP toxicity, Strimvelis
  IL2RG:     '#4a148c',  // deep purple — XSCID, gamma-c, OTL-101 FDA2024
  RAG1:      '#b71c1c',  // deep red   — Omenn/SCID, V(D)J recombination
  WAS:       '#006064',  // dark cyan  — Wiskott-Aldrich, microplatelets
  DOCK8:     '#2e7d32',  // deep green — DOCK8 HIES2, Molluscum/HPV, STAT3 DDx
  TNFRSF13B: '#f57f17',  // amber      — TACI/CVID2, GLILD, lymphoma surveillance
  LRBA:      '#4e342e',  // deep brown — LRBA deficiency, abatacept PATHOGNOMONIC
};

const GENE_DISEASE = {
  BTK:       'XLR XLA — 638aa Xq22.1 — No-Mature-B-Cells — Absent-ALL-Ig-Classes — Live-Vaccines-ABSOLUTELY-CI — IVIG-Lifelong — Enteroviral-Encephalitis-Fatal',
  ADA:       'AR ADA-SCID — 363aa 20q13.12 — dATP-Lymphotoxic — T-B-NK-Pan-Lymphopenia — Strimvelis-EMA2016 — PEG-ADA-Bridge — Skeletal-Dysplasia-Pathognomonic',
  IL2RG:     'XLR XSCID — 369aa Xq13.1 — Common-gamma-c-Shared-6-Cytokines — T-B+NK- — Irradiated-Blood-MANDATORY — OTL-101-FDA2024 — HSCT-before-3m-Best',
  RAG1:      'AR Omenn/RAG1-SCID — 1043aa 11p13 — VDJ-Recombination — Hypomorphic=Omenn — Complete-LOF=SCID — Ciclosporin-Before-HSCT — BCG-ABSOLUTELY-CI',
  WAS:       'XLR Wiskott-Aldrich — 502aa Xp11.23 — Eczema+Microplatelets+Immunodeficiency-TRIAD — MPV<7fL-PATHOGNOMONIC — Splenectomy-ABSOLUTELY-CI — HSCT-Curative',
  DOCK8:     'AR DOCK8-HIES2 — 2099aa 9p24.3 — Molluscum-Contagiosum+HPV-PATHOGNOMONIC — Severe-Eczema — ElevatedIgE — STAT3-HIES-DDx — HSCT-Curative',
  TNFRSF13B: 'AD/AR CVID2-TACI — 293aa 17p11.2 — Hypogammaglobulinaemia-IgG+IgA — GLILD-10-20pct — Lymphoma-5x-Risk — IVIG/SCIG-Lifelong — No-Live-Vaccines',
  LRBA:      'AR CVID8-LRBA — 2863aa 4q31.3 — CTLA4-Recycling — Immune-Dysregulation-AIHA+IBD+GLILD — Abatacept-PATHOGNOMONIC-Response — HSCT-Curative',
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
function Alert({ text }) {
  const lv = /ABSOLUTELY.CI|PATHOGNOMONIC|MANDATORY|ABSOLUTELY|FATAL|NEVER|LETHAL/i.test(text) ? 'critical'
           : /\bCI\b|RISK|MONITOR|WARNING|AVOID/i.test(text) ? 'warning' : 'info';
  const colors = { critical: '#fca5a5', warning: '#fcd34d', info: '#93c5fd' };
  const bg     = { critical: '#450a0a', warning: '#451a03', info: '#0c1a3a' };
  return (
    <div style={{
      background: bg[lv], border: `1px solid ${colors[lv]}33`,
      borderLeft: `3px solid ${colors[lv]}`, borderRadius: 6,
      padding: '0.4rem 0.7rem', fontSize: 12, color: colors[lv], marginBottom: 4,
    }}>{text}</div>
  );
}

/* ── OVERVIEW TAB ─────────────────────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats || {};
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 4 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: '1.5rem', lineHeight: 1.5 }}>
        {data.subtitle}
      </p>

      {/* KPIs */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
        <KPI label="Total Patients"         value={data.total_patients}                    color="#6366f1" />
        <KPI label="Genes Covered"          value={s.genes_covered ?? 8}                  color="#10b981" />
        <KPI label="Seeds"                  value={data.seed_range}                        color="#f59e0b" />
        <KPI label="XLR Genes"             value={s.xlr_genes ?? 3}                       color="#3b82f6" />
        <KPI label="AR Genes"              value={s.ar_genes ?? 4}                        color="#8b5cf6" />
        <KPI label="SCID Genes"           value={s.scid_genes ?? 3}                       color="#ef4444" />
        <KPI label="Antibody Deficiency"   value={s.antibody_deficiency_genes ?? 2}       color="#0d9488" />
        <KPI label="Combined ID Genes"     value={s.combined_immunodeficiency_genes ?? 3} color="#f472b6" />
      </div>

      {/* Inheritance breakdown bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Inheritance Pattern (8 genes)</div>
        <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          {[
            { label: 'XLR (3 genes: BTK, IL2RG, WAS)', val: 3, color: '#3b82f6' },
            { label: 'AR (4 genes: ADA, RAG1, DOCK8, LRBA)', val: 4, color: '#10b981' },
            { label: 'AD/AR (1: TNFRSF13B)', val: 1, color: '#f59e0b' },
          ].map(b => (
            <div key={b.label} title={b.label} style={{ flex: b.val, background: b.color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#fff', fontWeight: 700, overflow: 'hidden' }}>
              {b.val}
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 16, marginTop: 6, fontSize: 11, color: '#64748b', flexWrap: 'wrap' }}>
          <span style={{ color: '#3b82f6' }}>■ XLR (3)</span>
          <span style={{ color: '#10b981' }}>■ AR (4)</span>
          <span style={{ color: '#f59e0b' }}>■ AD/AR (1)</span>
        </div>
      </div>

      {/* Top alerts */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Key Clinical Alerts</div>
        {(data.top_alerts || []).map((a, i) => <Alert key={i} text={a} />)}
      </div>

      {/* Gene summary table */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Gene Summary (320 patients, 8 × 40)</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
              {['Gene', 'Locus', 'aa', 'Inheritance', 'Disease Class', 'Patients'].map(h => (
                <th key={h} style={{ padding: '6px 8px', textAlign: 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data.genes || []).map(g => (
              <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b55' }}>
                <td style={{ padding: '5px 8px', color: GENE_COLORS[g.gene] || '#f1f5f9', fontWeight: 700 }}>{g.gene}</td>
                <td style={{ padding: '5px 8px', color: '#cbd5e1', fontFamily: 'monospace' }}>{g.locus}</td>
                <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{g.aa}</td>
                <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{g.inheritance || '—'}</td>
                <td style={{ padding: '5px 8px', color: '#64748b', fontSize: 11 }}>{GENE_DISEASE[g.gene]?.split('—')[1]?.trim() || '—'}</td>
                <td style={{ padding: '5px 8px', color: '#f1f5f9', fontWeight: 600 }}>{g.n_patients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ───────────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      {data.map(g => (
        <div key={g.gene} style={{
          background: '#1e293b', borderRadius: 10, padding: '1.2rem', marginBottom: '1rem',
          borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
        }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', flexWrap: 'wrap', marginBottom: 6 }}>
            <span style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
            <span style={{ fontSize: 12, color: '#64748b' }}>{g.locus} · {g.aa} aa · {g.inheritance}</span>
            <span style={{ fontSize: 11, color: '#475569' }}>OMIM gene {g.omim_gene} / disease {g.omim_disease}</span>
          </div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8, lineHeight: 1.6 }}>
            {g.alias?.slice(0, 400)}…
          </div>

          {/* Etiologies */}
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Mutation / Etiology Classes</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {(g.etiologies || []).map(e => (
                <span key={e.label} style={{
                  background: `${GENE_COLORS[g.gene]}22`, color: GENE_COLORS[g.gene] || '#a5b4fc',
                  border: `1px solid ${GENE_COLORS[g.gene]}55`, borderRadius: 4,
                  padding: '2px 8px', fontSize: 11,
                }}>
                  {e.label} {e.pct != null ? `${e.pct}%` : ''}
                </span>
              ))}
            </div>
          </div>

          {/* Key alerts */}
          <div>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Key Clinical Alerts</div>
            {(g.key_alerts || []).map((a, i) => <Alert key={i} text={a} />)}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ───────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(data[0]?.gene || 'BTK');
  const gene = data.find(g => g.gene === selected);
  if (!gene) return <ErrorBox msg="Gene not found" />;

  const stats = gene.stats || {};
  const patients = gene.sample_patients || [];
  const statKeys = Object.entries(stats).filter(([, v]) => v != null);

  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
      {/* Sidebar */}
      <div style={{ minWidth: 160, flex: '0 0 160px' }}>
        {data.map(g => (
          <div key={g.gene}
            onClick={() => setSelected(g.gene)}
            style={{
              padding: '8px 14px', borderRadius: 8, marginBottom: 4, cursor: 'pointer',
              background: selected === g.gene ? `${GENE_COLORS[g.gene]}33` : '#1e293b',
              borderLeft: `3px solid ${selected === g.gene ? GENE_COLORS[g.gene] : 'transparent'}`,
              color: selected === g.gene ? GENE_COLORS[g.gene] : '#94a3b8',
              fontSize: 13, fontWeight: selected === g.gene ? 700 : 400,
            }}>
            {g.gene}
          </div>
        ))}
      </div>

      {/* Main content */}
      <div style={{ flex: 1, minWidth: 300 }}>
        <div style={{ color: GENE_COLORS[gene.gene] || '#a5b4fc', fontSize: 16, fontWeight: 700, marginBottom: 4 }}>
          {gene.gene} — {GENE_DISEASE[gene.gene] || gene.gene_class}
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginBottom: '1rem' }}>
          {gene.locus} · {gene.aa} aa · {gene.inheritance} · OMIM gene {gene.omim_gene}
        </div>

        {/* Stats grid */}
        {statKeys.length > 0 && (
          <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>40-Patient Cohort Statistics</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 8 }}>
              {statKeys.map(([k, v]) => (
                <div key={k} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px' }}>
                  <div style={{ fontSize: 10, color: '#64748b' }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e8f0' }}>
                    {typeof v === 'number' && v >= 0 && v <= 100 ? `${v}%` : v}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Sample patients table */}
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>
            Sample Patients ({patients.length} shown)
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                  {['Age', 'Sex', 'Etiology', 'Dx Delay', 'HSCT', 'IVIG', 'GT', 'Prophylaxis'].map(h => (
                    <th key={h} style={{ padding: '4px 6px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {patients.slice(0, 10).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #1e293b55' }}>
                    <td style={{ padding: '4px 6px', color: '#cbd5e1' }}>{p.age_at_diagnosis ?? p.age ?? '—'}</td>
                    <td style={{ padding: '4px 6px', color: '#94a3b8' }}>{p.sex}</td>
                    <td style={{ padding: '4px 6px', color: GENE_COLORS[gene.gene] || '#a5b4fc', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                    <td style={{ padding: '4px 6px', color: '#94a3b8' }}>{p.dx_delay_months != null ? `${p.dx_delay_months}m` : '—'}</td>
                    <td style={{ padding: '4px 6px', color: p.hsct ? '#34d399' : '#64748b' }}>{p.hsct ? 'Yes' : 'No'}</td>
                    <td style={{ padding: '4px 6px', color: p.ivig ? '#60a5fa' : '#64748b' }}>{p.ivig ? 'Yes' : 'No'}</td>
                    <td style={{ padding: '4px 6px', color: p.gene_therapy ? '#f472b6' : '#64748b' }}>{p.gene_therapy ? 'Yes' : 'No'}</td>
                    <td style={{ padding: '4px 6px', color: p.prophylaxis ? '#fbbf24' : '#64748b' }}>{p.prophylaxis ? 'Yes' : 'No'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ──────────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const concepts = data.concepts || {};
  const pharma = data.pharmacological_distinctions || [];
  const standards = data.key_standards || [];

  return (
    <div>
      {/* Pharmacological Distinctions */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9', marginBottom: 8 }}>
          Pharmacological Distinctions ({pharma.length})
        </div>
        {pharma.map((p, i) => (
          <div key={i} style={{
            background: '#0f172a', borderRadius: 6, padding: '8px 12px', marginBottom: 6,
            fontSize: 12, color: '#cbd5e1', lineHeight: 1.5,
            borderLeft: '3px solid #6366f1',
          }}>{p}</div>
        ))}
      </div>

      {/* Key Standards */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9', marginBottom: 8 }}>
          Key Standards &amp; Protocols ({standards.length})
        </div>
        {standards.map((s, i) => (
          <div key={i} style={{
            background: '#0f172a', borderRadius: 6, padding: '8px 12px', marginBottom: 6,
            fontSize: 12, color: '#cbd5e1', lineHeight: 1.5,
            borderLeft: '3px solid #10b981',
          }}>{s}</div>
        ))}
      </div>

      {/* Concepts */}
      {Object.keys(concepts).length > 0 && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9', marginBottom: 8 }}>
            Clinical Concepts ({Object.keys(concepts).length})
          </div>
          {Object.entries(concepts).map(([k, v]) => (
            <div key={k} style={{ marginBottom: '1rem' }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#a5b4fc', marginBottom: 4 }}>{k}</div>
              <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>
                {String(v).slice(0, 600)}{String(v).length > 600 ? '…' : ''}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── ROOT PAGE ────────────────────────────────────────────────────────────── */
export default function HImmunodeficiencyAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-immunodeficiency-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) {
    return (
      <div style={{ padding: '2rem', background: '#0f172a', minHeight: '100vh' }}>
        <ErrorBox msg={error} />
      </div>
    );
  }

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', padding: '1.5rem', color: '#f1f5f9' }}>
      {/* Header */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 11, color: '#475569', marginBottom: 4 }}>
          Expert Dashboards › Hereditary Disease Atlases › Primary Immunodeficiency
        </div>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', margin: 0 }}>
          Hereditary Immunodeficiency Atlas
        </h1>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          Complete 8-Gene Primary Immunodeficiency Atlas — BTK · ADA · IL2RG · RAG1 · WAS · DOCK8 · TNFRSF13B · LRBA — 320 Patients (Seeds 1782–1789)
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', borderBottom: '1px solid #1e293b', paddingBottom: 0 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            background: 'none', border: 'none', cursor: 'pointer',
            padding: '8px 18px', fontSize: 13, fontWeight: tab === i ? 700 : 400,
            color: tab === i ? '#a5b4fc' : '#64748b',
            borderBottom: tab === i ? '2px solid #6366f1' : '2px solid transparent',
          }}>{t}</button>
        ))}
      </div>

      {/* Tab Content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <GeneTableTab data={breakdown} />}
      {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
