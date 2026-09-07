'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BTK:       '#1565c0',  // deep blue    — XLA, X-linked agammaglobulinemia; IVIG lifelong
  IL2RG:     '#4a148c',  // deep purple  — SCID-X1, common-gamma-c, OTL-101 FDA2024
  ADA:       '#e65100',  // deep orange  — ADA-SCID, dATP toxicity, Strimvelis EMA2016
  RAG1:      '#b71c1c',  // deep red     — Omenn/SCID, V(D)J recombination, cyclosporin
  RAG2:      '#880e4f',  // deep pink    — RAG2-SCID/Omenn, NK+ phenotype, panel testing
  DCLRE1C:   '#1b5e20',  // deep green   — Artemis-SCID, radiation-sensitive, RIC mandatory
  JAK3:      '#0d47a1',  // dark blue    — JAK3-SCID, AR phenocopy SCID-X1, females affected
  TNFRSF13B: '#f57f17',  // amber        — CVID/TACI, GLILD, lymphoma 8× risk, IVIG lifelong
};

const GENE_INFO = {
  BTK:       { aa: 659,  locus: 'Xq22.2',   inh: 'XLR', disease: 'XLA — Absent-B-Cells — Live-Vaccines-ABSOLUTELY-CI — IVIG-Lifelong — Enteroviral-Encephalitis-Fatal' },
  IL2RG:     { aa: 369,  locus: 'Xq13.1',   inh: 'XLR', disease: 'SCID-X1 — T-B+NK- — Common-gamma-c — OTL-101-FDA2024 — Irradiated-Blood-MANDATORY — TREC-Newborn-Screen' },
  ADA:       { aa: 363,  locus: '20q13.12',  inh: 'AR',  disease: 'ADA-SCID — T-B-NK- — dATP-Lymphotoxic — Strimvelis-EMA2016 — PEG-ADA-Bridge — Skeletal-Dysplasia-Pathognomonic' },
  RAG1:      { aa: 1043, locus: '11p13',     inh: 'AR',  disease: 'RAG1-SCID/Omenn — T-B-NK+ — VDJ-Recombination — Hypomorphic=Omenn — Ciclosporin-Before-HSCT — BCG-ABSOLUTELY-CI' },
  RAG2:      { aa: 527,  locus: '11p13',     inh: 'AR',  disease: 'RAG2-SCID/Omenn — T-B-NK+ — Adjacent-RAG1-Test-BOTH — IgE>10000-Not-Allergy — Cyclosporin-Before-HSCT' },
  DCLRE1C:   { aa: 692,  locus: '10p13',     inh: 'AR',  disease: 'Artemis-SCID — T-B-NK+ — Radiation-Sensitive — RIC-Conditioning-MANDATORY — Myeloablative-LETHAL — Athabascan-Founder' },
  JAK3:      { aa: 1124, locus: '19p13.11',  inh: 'AR',  disease: 'JAK3-SCID — T-B+NK- — AR-SCID-X1-Phenocopy — Females-Affected — JAK3-Not-Radiation-Sensitive — Maternal-Engraftment-Risk' },
  TNFRSF13B: { aa: 293,  locus: '17p11.2',   inh: 'AD/AR', disease: 'CVID/TACI — Hypogammaglobulinaemia — Absent-Vaccine-Responses-REQUIRED — GLILD-10-20pct — Lymphoma-8x-Risk — IVIG-Lifelong' },
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
  const geneCounts = data.gene_patient_counts || {};
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 4 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: '1.5rem', lineHeight: 1.5 }}>
        {data.subtitle}
      </p>

      {/* KPIs */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
        <KPI label="Total Patients"         value={data.total_patients}                    color="#6366f1" />
        <KPI label="Genes Covered"          value={(data.genes || []).length || 8}         color="#10b981" />
        <KPI label="Seeds"                  value={data.seeds}                             color="#f59e0b" />
        <KPI label="XLR Genes (BTK, IL2RG)"value={2}                                      color="#3b82f6" />
        <KPI label="AR Genes (5 genes)"     value={5}                                      color="#8b5cf6" />
        <KPI label="SCID Genes"             value={6}                                      color="#ef4444" />
        <KPI label="IVIG Patients"          value={data.ivig_patients}                     color="#0d9488" />
        <KPI label="Radiation-Sensitive Pts" value={data.radiation_sensitive_patients}     color="#f472b6" />
      </div>

      {/* Inheritance breakdown bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Inheritance Pattern (8 genes)</div>
        <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          {[
            { label: 'XLR (2 genes: BTK, IL2RG)', val: 2, color: '#3b82f6' },
            { label: 'AR (5 genes: ADA, RAG1, RAG2, DCLRE1C, JAK3)', val: 5, color: '#10b981' },
            { label: 'AD/AR (1: TNFRSF13B)', val: 1, color: '#f59e0b' },
          ].map(b => (
            <div key={b.label} title={b.label} style={{ flex: b.val, background: b.color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#fff', fontWeight: 700, overflow: 'hidden' }}>
              {b.val}
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 16, marginTop: 6, fontSize: 11, color: '#64748b', flexWrap: 'wrap' }}>
          <span style={{ color: '#3b82f6' }}>■ XLR (2: BTK, IL2RG)</span>
          <span style={{ color: '#10b981' }}>■ AR (5: ADA, RAG1, RAG2, DCLRE1C, JAK3)</span>
          <span style={{ color: '#f59e0b' }}>■ AD/AR (1: TNFRSF13B)</span>
        </div>
      </div>

      {/* Top pathway */}
      {data.pathway && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
          <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Molecular Pathway</div>
          <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{data.pathway}</div>
        </div>
      )}

      {/* Gene summary table — built from gene_patient_counts + GENE_INFO */}
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
            {Object.entries(geneCounts).map(([gene, cnt]) => {
              const info = GENE_INFO[gene] || {};
              return (
                <tr key={gene} style={{ borderBottom: '1px solid #1e293b55' }}>
                  <td style={{ padding: '5px 8px', color: GENE_COLORS[gene] || '#f1f5f9', fontWeight: 700 }}>{gene}</td>
                  <td style={{ padding: '5px 8px', color: '#cbd5e1', fontFamily: 'monospace' }}>{info.locus || '—'}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{info.aa || '—'}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{info.inh || '—'}</td>
                  <td style={{ padding: '5px 8px', color: '#64748b', fontSize: 11, maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{info.disease || '—'}</td>
                  <td style={{ padding: '5px 8px', color: '#f1f5f9', fontWeight: 600 }}>{cnt}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ───────────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div>
      {genes.map(g => (
        <div key={g.gene} style={{
          background: '#1e293b', borderRadius: 10, padding: '1.2rem', marginBottom: '1rem',
          borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
        }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', flexWrap: 'wrap', marginBottom: 6 }}>
            <span style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
            <span style={{ fontSize: 12, color: '#64748b' }}>{g.locus} · {g.protein_size} aa · {g.inheritance}</span>
            <span style={{ fontSize: 11, color: '#475569' }}>{g.alt_name}</span>
          </div>

          {/* Age of onset + biomarker */}
          <div style={{ display: 'flex', gap: 24, marginBottom: 8, flexWrap: 'wrap' }}>
            <div>
              <span style={{ fontSize: 10, color: '#64748b' }}>Age of Onset: </span>
              <span style={{ fontSize: 11, color: '#cbd5e1' }}>{g.age_of_onset}</span>
            </div>
            <div>
              <span style={{ fontSize: 10, color: '#64748b' }}>Key Biomarker: </span>
              <span style={{ fontSize: 11, color: '#cbd5e1' }}>{g.key_biomarker}</span>
            </div>
          </div>

          {/* Stats bar */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
            {[
              { label: 'IVIG', val: g.ivig_pct },
              { label: 'HSCT', val: g.hsct_pct },
              { label: 'Gene Tx', val: g.gene_therapy_pct },
              { label: 'PCP', val: g.pcp_pneumonia_pct },
              { label: 'Maternal Engraftment', val: g.maternal_engraftment_pct },
              { label: 'Autoimmune', val: g.autoimmune_complication_pct },
              { label: 'Lymphoma', val: g.lymphoma_pct },
              { label: 'Omenn', val: g.omenn_syndrome_pct },
            ].filter(s => s.val > 0).map(s => (
              <span key={s.label} style={{
                background: '#0f172a', borderRadius: 4, padding: '2px 8px',
                fontSize: 11, color: '#94a3b8',
              }}>{s.label}: <strong style={{ color: '#e2e8f0' }}>{s.val}%</strong></span>
            ))}
          </div>

          {/* Pathognomonic */}
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Pathognomonic / Diagnostic Pivot</div>
            <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5 }}>
              {String(g.pathognomonic || '').slice(0, 400)}{(g.pathognomonic || '').length > 400 ? '…' : ''}
            </div>
          </div>

          {/* Critical flags */}
          <div>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Critical Clinical Alerts</div>
            {(g.critical_flags || []).map((a, i) => <Alert key={i} text={a} />)}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ───────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  const [selected, setSelected] = useState(genes[0]?.gene || 'BTK');
  const gene = genes.find(g => g.gene === selected);
  if (!gene) return <ErrorBox msg="Gene not found" />;

  const preview = gene.cohort_preview || [];

  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
      {/* Sidebar */}
      <div style={{ minWidth: 160, flex: '0 0 160px' }}>
        {genes.map(g => (
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
          {gene.gene} — {GENE_INFO[gene.gene]?.disease || gene.alt_name}
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginBottom: '1rem' }}>
          {gene.locus} · {gene.protein_size} aa · {gene.inheritance} · {gene.alt_name}
        </div>

        {/* Stats grid */}
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>40-Patient Cohort Statistics</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(180px,1fr))', gap: 8 }}>
            {[
              { k: 'IVIG', v: gene.ivig_pct },
              { k: 'HSCT', v: gene.hsct_pct },
              { k: 'Gene Therapy', v: gene.gene_therapy_pct },
              { k: 'PCP Pneumonia', v: gene.pcp_pneumonia_pct },
              { k: 'Maternal Engraftment', v: gene.maternal_engraftment_pct },
              { k: 'Autoimmune Complication', v: gene.autoimmune_complication_pct },
              { k: 'Lymphoma', v: gene.lymphoma_pct },
              { k: 'Omenn Syndrome', v: gene.omenn_syndrome_pct },
              { k: 'Bronchiectasis', v: gene.bronchiectasis_pct },
              { k: 'Sinopulmonary Infections', v: gene.sinopulmonary_infections_pct },
            ].filter(s => s.v != null).map(({ k, v }) => (
              <div key={k} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px' }}>
                <div style={{ fontSize: 10, color: '#64748b' }}>{k}</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e8f0' }}>{v}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Treatment */}
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>Treatment Protocol</div>
          <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
            {String(gene.treatment || '').slice(0, 600)}{(gene.treatment || '').length > 600 ? '…' : ''}
          </div>
        </div>

        {/* Sample patients table */}
        {preview.length > 0 && (
          <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>
              Sample Patients ({preview.length} shown)
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                    {['Age (dx)', 'Sex', 'Etiology', 'Dx Delay', 'HSCT', 'IVIG', 'GT', 'Prophylaxis'].map(h => (
                      <th key={h} style={{ padding: '4px 6px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #1e293b55' }}>
                      <td style={{ padding: '4px 6px', color: '#cbd5e1' }}>{p.age_at_diagnosis ?? p.age ?? '—'}</td>
                      <td style={{ padding: '4px 6px', color: '#94a3b8' }}>{p.sex}</td>
                      <td style={{ padding: '4px 6px', color: GENE_COLORS[gene.gene] || '#a5b4fc', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                      <td style={{ padding: '4px 6px', color: '#94a3b8' }}>{p.dx_delay_months != null ? `${p.dx_delay_months}m` : '—'}</td>
                      <td style={{ padding: '4px 6px', color: p.hsct_received ? '#34d399' : '#64748b' }}>{p.hsct_received ? 'Yes' : 'No'}</td>
                      <td style={{ padding: '4px 6px', color: p.on_ivig ? '#60a5fa' : '#64748b' }}>{p.on_ivig ? 'Yes' : 'No'}</td>
                      <td style={{ padding: '4px 6px', color: (p.gene_therapy_received || p.strimvelis_received) ? '#f472b6' : '#64748b' }}>{(p.gene_therapy_received || p.strimvelis_received) ? 'Yes' : 'No'}</td>
                      <td style={{ padding: '4px 6px', color: p.prophylaxis ? '#fbbf24' : '#64748b' }}>{p.prophylaxis ? 'Yes' : 'No'}</td>
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

/* ── DEFINITIONS TAB ──────────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const glossary = data.glossary || {};
  const pearls = data.clinical_pearls || [];
  const geneDefs = data.gene_definitions || {};

  return (
    <div>
      {/* Clinical Pearls */}
      {pearls.length > 0 && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9', marginBottom: 8 }}>
            Clinical Pearls ({pearls.length})
          </div>
          {pearls.map((p, i) => (
            <div key={i} style={{
              background: '#0f172a', borderRadius: 6, padding: '8px 12px', marginBottom: 6,
              fontSize: 12, color: '#cbd5e1', lineHeight: 1.5,
              borderLeft: '3px solid #6366f1',
            }}>{p}</div>
          ))}
        </div>
      )}

      {/* Gene Definitions */}
      {Object.keys(geneDefs).length > 0 && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9', marginBottom: 8 }}>
            Gene Definitions ({Object.keys(geneDefs).length})
          </div>
          {Object.entries(geneDefs).map(([gene, info]) => (
            <div key={gene} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: '1px solid #334155' }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc', fontSize: 13 }}>{gene}</span>
              <span style={{ color: '#64748b', fontSize: 11, marginLeft: 8 }}>
                {info.protein} · {info.locus} · {info.protein_size} aa · {info.inheritance}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Glossary */}
      {Object.keys(glossary).length > 0 && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#f1f5f9', marginBottom: 8 }}>
            Clinical Glossary ({Object.keys(glossary).length} terms)
          </div>
          {Object.entries(glossary).map(([k, v]) => (
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
          Complete 8-Gene Primary Immunodeficiency Atlas — BTK · IL2RG · ADA · RAG1 · RAG2 · DCLRE1C · JAK3 · TNFRSF13B — 320 Patients (Seeds 1974–1981)
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
