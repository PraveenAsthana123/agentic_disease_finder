'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  HPRT1: '#dc2626',  // red — Lesch-Nyhan, self-injurious behaviour, no neurological cure
  ADSL:  '#b45309',  // amber — SAICAR/SAdo pathognomonic, autistic features
  ADA:   '#1d4ed8',  // blue — ADA-SCID, Strimvelis gene therapy, NBS-TREC
  PNP:   '#7c3aed',  // purple — T-cell selective, autoimmune, spastic diplegia
  XDH:   '#0f766e',  // teal — xanthinuria, radiolucent stones, allopurinol CI
  APRT:  '#065f46',  // dark green — 2,8-DHA crystals, allopurinol curative, Japanese
  AMPD1: '#c2410c',  // deep orange — forearm test ammonia absent, common polymorphism
  UMPS:  '#1e1b4b',  // dark indigo — orotic aciduria, uridine curative, DDx OTC
};

const GENE_DISEASE = {
  HPRT1: 'XLR Lesch-Nyhan — HGPRT-217aa — Xq26.2 — SIB-12-24m-PATHOGNOMONIC — Allopurinol-Uric-Acid-NOT-Neurology — Orange-Grit-Nappies',
  ADSL:  'AR ADSL-Deficiency — Adenylosuccinate-Lyase-484aa — 22q13.1 — SAICAR-SAdo-Urine-PATHOGNOMONIC — 3-Phenotypes-Neonatal-Severe-Mild — Autistic-Features',
  ADA:   'AR ADA-SCID — Adenosine-Deaminase-363aa — 20q13.12 — Most-Common-Enzyme-SCID — dATP-Destroys-T-B-NK — Strimvelis-EMA2016-Gene-Therapy — NBS-TREC',
  PNP:   'AR PNP-Deficiency — Purine-Nucleoside-Phosphorylase-289aa — 14q11.2 — T-Cell-Selective-DDx-ADA — Autoimmune-Haemolytic-Anaemia-50pct — Spastic-Diplegia',
  XDH:   'AR Xanthinuria-I — Xanthine-Dehydrogenase-1333aa — 2p23.1 — Radiolucent-Xanthine-Stones — Uric-Acid-VERY-LOW-KEY-DDx — Allopurinol-CONTRAINDICATED',
  APRT:  'AR APRT-Deficiency — Adenine-Phosphoribosyltransferase-180aa — 16q24.3 — 2,8-DHA-Crystals-PATHOGNOMONIC — Renal-Failure-If-Untreated — Allopurinol-CURATIVE',
  AMPD1: 'AR AMPD1-Deficiency — Myoadenylate-Deaminase-747aa — 1p13.3 — Forearm-Test-Ammonia-ABSENT-PATHOGNOMONIC — Lactate-Rises-Contrast-McArdle — Common-2pct-European',
  UMPS:  'AR Orotic-Aciduria — UMP-Synthase-480aa — 3q13.33 — Megaloblastic-NOT-B12-Folate — Uridine-Replacement-CURATIVE — DDx-OTC-Ammonia-NORMAL',
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
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats || {};
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 6 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 20 }}>{data.subtitle}</p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
        <KPI label="Total Patients" value={data.total_patients} color="#6366f1" />
        <KPI label="Genes Covered" value={s.genes_covered} color="#22c55e" />
        <KPI label="Seeds" value={data.seed_range} color="#f59e0b" />
        <KPI label="AR Genes" value={s.ar_genes} color="#dc2626" />
        <KPI label="X-Linked Genes" value={s.x_linked_genes} color="#7c3aed" />
        <KPI label="Patients / Gene" value={s.patients_per_gene} color="#0f766e" />
        <KPI label="Lesch-Nyhan SIB %" value={`${s.hprt1_sib_pct ?? 0}%`} color="#dc2626" />
        <KPI label="ADA-SCID T-Cell Absent %" value={`${s.ada_t_cell_absent_pct ?? 0}%`} color="#1d4ed8" />
        <KPI label="APRT Renal Failure %" value={`${s.aprt_renal_failure_pct ?? 0}%`} color="#065f46" />
        <KPI label="UMPS Orotic Aciduria %" value={`${s.umps_orotic_aciduria_pct ?? 0}%`} color="#1e1b4b" />
        <KPI label="AMPD1 Ammonia Absent %" value={`${s.ampd1_ammonia_absent_pct ?? 0}%`} color="#c2410c" />
        <KPI label="XDH Uric Acid Low %" value={`${s.xdh_uric_acid_low_pct ?? 0}%`} color="#0f766e" />
      </div>

      <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>🚨 Critical Clinical Alerts</h3>
      {(data.top_alerts || []).map((a, i) => (
        <Alert key={i} text={a} color={Object.values(GENE_COLORS)[i % 8]} />
      ))}

      <h3 style={{ color: '#e2e8f0', marginTop: 24, marginBottom: 10 }}>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 }}>
        {(data.genes || []).map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{g.locus} · {g.aa} aa · {g.inheritance}</div>
            <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 6 }}>
              {GENE_DISEASE[g.gene]?.split(' — ')[0]}
            </div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>n={g.n_patients} patients</div>
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
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>8-Gene Hereditary Purine & Pyrimidine Metabolism Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Protein', 'Locus', 'AA', 'kDa', 'OMIM Gene', 'Inheritance', 'Gene Class', 'Key Intervention'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 280, wordBreak: 'break-word' }}>{g.protein}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.aa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.kDa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.omim_gene}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 200 }}>{g.gene_class}</td>
                <td style={{ padding: '8px 10px', fontWeight: 600, color:
                  g.gene === 'HPRT1' ? '#dc2626' :
                  g.gene === 'ADA'   ? '#1d4ed8' :
                  g.gene === 'APRT'  ? '#065f46' :
                  g.gene === 'UMPS'  ? '#1e1b4b' :
                  '#94a3b8' }}>
                  {g.gene === 'HPRT1' ? 'Allopurinol (uric acid only, NOT neurology)' :
                   g.gene === 'ADA'   ? 'Strimvelis gene therapy / HSCT / PEG-ADA bridge' :
                   g.gene === 'PNP'   ? 'HSCT (less successful than ADA-SCID)' :
                   g.gene === 'XDH'   ? 'High fluids — allopurinol CONTRAINDICATED' :
                   g.gene === 'APRT'  ? 'Allopurinol CURATIVE + low adenine diet' :
                   g.gene === 'UMPS'  ? 'Uridine replacement CURATIVE' :
                   g.gene === 'ADSL'  ? 'No proven specific therapy; supportive' :
                   'Supportive; significance debated'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(0);
  if (!data) return <Loading />;
  const g = data[selected];
  if (!g) return null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 20 }}>
      <div>
        {data.map((gene, i) => (
          <button key={gene.gene} onClick={() => setSelected(i)} style={{
            display: 'block', width: '100%', textAlign: 'left',
            padding: '10px 14px', marginBottom: 6, borderRadius: 8,
            background: selected === i ? GENE_COLORS[gene.gene] : '#1e293b',
            color: selected === i ? '#fff' : '#94a3b8',
            border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
          }}>
            {gene.gene}
            <div style={{ fontSize: 10, fontWeight: 400, marginTop: 2 }}>{gene.locus}</div>
          </button>
        ))}
      </div>
      <div>
        <h3 style={{ color: GENE_COLORS[g.gene] || '#a5b4fc', marginBottom: 4 }}>{g.gene}</h3>
        <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 12 }}>{g.inheritance} · {g.locus} · {g.aa} aa · {g.kDa} kDa</div>

        <h4 style={{ color: '#e2e8f0', marginBottom: 6 }}>🚨 Key Alerts</h4>
        {(g.key_alerts || []).map((a, i) => (
          <Alert key={i} text={a} color={GENE_COLORS[g.gene]} />
        ))}

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Clinical Description</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '1rem',
          fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap',
          maxHeight: 400, overflowY: 'auto',
        }}>
          {g.alias}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Etiology / Presentation Distribution</h4>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(g.etiologies || {}).map(([k, v]) => (
            <div key={k} style={{
              background: '#1e293b', borderRadius: 8, padding: '8px 12px',
              fontSize: 12, color: '#94a3b8',
            }}>
              <span style={{ fontWeight: 600, color: GENE_COLORS[g.gene] }}>{v}%</span> {k.replace(/_/g, ' ')}
            </div>
          ))}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Sample Patients (first 10)</h4>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                <th style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b' }}>ID</th>
                {Object.keys(g.sample_patients?.[0] || {}).filter(k => k !== 'patient_id').slice(0, 6).map(k => (
                  <th key={k} style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b' }}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(g.sample_patients || []).map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '5px 8px', color: GENE_COLORS[g.gene] }}>{p.patient_id}</td>
                  {Object.entries(p).filter(([k]) => k !== 'patient_id').slice(0, 6).map(([k, v]) => (
                    <td key={k} style={{ padding: '5px 8px', color: '#94a3b8' }}>
                      {typeof v === 'boolean' ? (v ? '✓' : '✗') : String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 16 }}>Definitions & Clinical Concepts</h3>
      {Object.entries(data.concepts || {}).map(([title, body]) => (
        <div key={title} style={{ marginBottom: 20 }}>
          <h4 style={{ color: '#a5b4fc', marginBottom: 6 }}>{title}</h4>
          <div style={{
            background: '#1e293b', borderRadius: 8, padding: '1rem',
            fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap',
          }}>
            {body}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function HeredPurineAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-purine-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-purine-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-purine-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary Purine Atlas — Complete 8-Gene Purine & Pyrimidine Metabolism Reference
        </h1>
        <p style={{ color: '#64748b', fontSize: 13 }}>
          HPRT1 (Lesch-Nyhan-SIB-Allopurinol-NOT-Neurology) · ADSL (SAICAR-SAdo-PATHOGNOMONIC) ·
          ADA (ADA-SCID-Strimvelis-EMA2016) · PNP (T-Cell-Selective-Autoimmune) ·
          XDH (Xanthinuria-Radiolucent-Stones-Allopurinol-CI) · APRT (2,8-DHA-Allopurinol-CURATIVE) ·
          AMPD1 (Forearm-Ammonia-Absent-Common-2pct) · UMPS (Orotic-Aciduria-Uridine-CURATIVE) —
          320 Patients · Seeds 1814–1821
        </p>
      </div>

      {error && <ErrorBox msg={error} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 8,
            background: tab === i ? '#dc2626' : '#1e293b',
            color: tab === i ? '#fff' : '#94a3b8',
            border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div>
        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <GeneTableTab data={breakdown} />}
        {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
        {tab === 3 && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
