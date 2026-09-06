'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FANCA:  '#1a237e',  // deep navy — most common FA group
  FANCD2: '#b71c1c',  // deep red — severe central FA node
  DKC1:   '#4a148c',  // deep purple — X-linked DC telomere
  TERC:   '#1b5e20',  // deep green — AD DC with anticipation
  TERT:   '#e65100',  // deep orange — liver/lung dominant TBD
  ELANE:  '#37474f',  // dark slate — congenital neutropenia
  SBDS:   '#006064',  // deep teal — Shwachman-Diamond
  GATA2:  '#880e4f',  // deep magenta — MonoMAC/Emberger
};

const GENE_DISEASE = {
  FANCA:  'Fanconi Anaemia Group A — Most Common FA 60-70% — DEB Test — RIC HSCT',
  FANCD2: 'Fanconi Anaemia Group D2 — Central FA Node — VACTERL — Severe Phenotype',
  DKC1:   'Dyskeratosis Congenita X-linked — Triad — Telomere <1st%ile — PF ABSOLUTE-CI-HSCT',
  TERC:   'DC Autosomal Dominant — Anticipation Each Generation — Androgen Responsive',
  TERT:   'Telomere Biology — Liver Cirrhosis + IPF Dominant — AD/AR',
  ELANE:  'Severe Congenital Neutropenia SCN1 + Cyclic Neutropenia — G-CSF Standard',
  SBDS:   'Shwachman-Diamond — EPI + Neutropenia + Skeletal — 25% AML/MDS — PERT Mandatory',
  GATA2:  'GATA2 Deficiency MonoMAC — Monocytes <10 PATHOGNOMONIC — HSCT CURATIVE',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|LETHAL/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE/i.test(text);
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
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
      <div style={{ flex: '1 1 340px' }}>
        <h3 style={{ color: '#1a237e', marginBottom: 12 }}>Aggregate — 320 Patients (8×40, seeds 1438–1445)</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <tbody>
            {[
              ['Bone Marrow Failure', s.bone_marrow_failure_pct, '%'],
              ['AML/MDS Lifetime Risk', s.aml_mds_lifetime_pct, '%'],
              ['HSCT Performed', s.hsct_performed_pct, '%'],
              ['Androgen Therapy Response', s.androgen_responsive_pct, '%'],
              ['Telomere <1st Percentile (DC/TBD)', s.telomere_below_1pct_pct, '%'],
              ['Pulmonary Fibrosis', s.pulmonary_fibrosis_pct, '%'],
              ['Neutropenia', s.neutropenia_pct, '%'],
              ['Exocrine Pancreatic Insufficiency (SBDS)', s.exocrine_pancreatic_insuff_pct, '%'],
              ['Monocytopenia <10/μL (GATA2)', s.monocytopenia_pct, '%'],
              ['DEB/MMC Fragility Positive (FA)', s.deb_fragility_pct, '%'],
            ].map(([label, val, unit]) => (
              <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '6px 8px', color: '#555' }}>{label}</td>
                <td style={{ padding: '6px 8px', fontWeight: 700, color: '#1a237e' }}>
                  {val !== undefined && val !== null ? `${val}${unit}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3 style={{ color: '#1a237e', marginTop: 20, marginBottom: 10 }}>8 Genes — Disease Associations</h3>
        {genes && genes.map(g => (
          <div key={g} style={{
            background: GENE_COLORS[g] || '#333',
            color: '#fff', borderRadius: 6, padding: '6px 12px',
            marginBottom: 6, fontSize: 13,
          }}>
            <strong>{g}</strong> — {GENE_DISEASE[g] || g}
          </div>
        ))}
      </div>

      <div style={{ flex: '1 1 340px' }}>
        <h3 style={{ color: '#b71c1c', marginBottom: 10 }}>Critical Clinical Alerts</h3>
        {top_alerts && top_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#1a237e', color: '#fff' }}>
            {['Gene', 'Protein', 'Locus', 'aa', 'Inheritance', 'OMIM Gene', 'OMIM Disease', 'N'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {genes.map((g, i) => (
            <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff', borderBottom: '1px solid #ddd' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
              <td style={{ padding: '7px 10px', maxWidth: 220, fontSize: 12 }}>{g.protein}</td>
              <td style={{ padding: '7px 10px', fontFamily: 'monospace' }}>{g.locus}</td>
              <td style={{ padding: '7px 10px' }}>{g.aa}</td>
              <td style={{ padding: '7px 10px', fontSize: 12, maxWidth: 160 }}>{g.inheritance}</td>
              <td style={{ padding: '7px 10px' }}>{g.omim_gene}</td>
              <td style={{ padding: '7px 10px' }}>{g.omim_disease}</td>
              <td style={{ padding: '7px 10px', fontWeight: 700 }}>{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(Object.keys(data)[0]);
  const g = data[selected];

  return (
    <div style={{ display: 'flex', gap: 20 }}>
      {/* Gene selector */}
      <div style={{ minWidth: 140 }}>
        {Object.keys(data).map(gene => (
          <button key={gene} onClick={() => setSelected(gene)} style={{
            display: 'block', width: '100%', textAlign: 'left',
            padding: '8px 12px', marginBottom: 4, borderRadius: 6, cursor: 'pointer',
            border: 'none',
            background: selected === gene ? (GENE_COLORS[gene] || '#1a237e') : '#eee',
            color: selected === gene ? '#fff' : '#333',
            fontWeight: selected === gene ? 700 : 400, fontSize: 13,
          }}>{gene}</button>
        ))}
      </div>

      {/* Gene details */}
      <div style={{ flex: 1 }}>
        <h3 style={{ color: GENE_COLORS[selected] || '#1a237e', marginBottom: 4 }}>{g.gene} — {g.protein}</h3>
        <p style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
          {g.locus} · {g.aa} · {g.inheritance}
        </p>
        <p style={{ fontSize: 13, color: '#333', marginBottom: 12, lineHeight: 1.6 }}>{g.gene_class}</p>

        <h4 style={{ color: '#b71c1c', marginBottom: 6 }}>Critical Clinical Alerts</h4>
        {g.critical_alerts && g.critical_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}

        <h4 style={{ color: '#1565c0', marginTop: 14, marginBottom: 6 }}>DDx Rules</h4>
        {g.key_ddx_rules && g.key_ddx_rules.map((r, i) => (
          <div key={i} style={{
            background: '#e3f2fd', borderLeft: '4px solid #1565c0',
            padding: '6px 10px', marginBottom: 6, fontSize: 13, borderRadius: '0 4px 4px 0',
          }}>{r}</div>
        ))}

        <h4 style={{ color: '#555', marginTop: 14, marginBottom: 6 }}>Phenotype Rates (%)</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <tbody>
            {g.phenotype_rates && Object.entries(g.phenotype_rates).map(([k, v]) => (
              <tr key={k} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '4px 8px', color: '#555', textTransform: 'capitalize' }}>
                  {k.replace(/_/g, ' ')}
                </td>
                <td style={{ padding: '4px 8px', fontWeight: 700, color: '#1a237e' }}>{v}%</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h4 style={{ color: '#555', marginTop: 14, marginBottom: 6 }}>Etiology Distribution</h4>
        {g.etiologies && g.etiologies.map((e, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
            <div style={{
              width: `${e.pct * 2.5}px`, minWidth: 4, height: 18,
              background: GENE_COLORS[selected] || '#1a237e', borderRadius: 3, marginRight: 8,
            }} />
            <span style={{ fontSize: 12 }}>{e.label} — <strong>{e.pct}%</strong></span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || [];
  return (
    <div>
      {defs.map((d, i) => (
        <div key={i} style={{
          borderLeft: '4px solid #1a237e', padding: '12px 16px',
          marginBottom: 16, background: '#f8f9fa', borderRadius: '0 6px 6px 0',
        }}>
          <h4 style={{ color: '#1a237e', marginBottom: 6, fontSize: 15 }}>{d.term}</h4>
          <p style={{ fontSize: 13, color: '#333', lineHeight: 1.6, margin: 0 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HBMFAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-bmf-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
    fetch(`${API}/api/hereditary-bmf-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/hereditary-bmf-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto', padding: '24px 16px' }}>
      <h1 style={{ color: '#1a237e', marginBottom: 4, fontSize: 22 }}>
        🧬 Hereditary Bone Marrow Failure Syndromes Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 20, fontSize: 14 }}>
        Complete 8-Gene HBMF-Atlas · FANCA-1455aa-16q24.3-AR-FA-Most-Common-60-70pct-DEB-Test-RIC-Mandatory ·
        FANCD2-1451aa-3p25.3-AR-Central-FA-Node-VACTERL-Severe · DKC1-514aa-Xq28-XL-DC-Triad-Telomere-PF-ABSOLUTE-CI-HSCT ·
        TERC-451nt-3q26.2-AD-DC-Anticipation-Androgen-Responsive · TERT-1132aa-5p15.33-AD-AR-Liver-IPF-Dominant ·
        ELANE-267aa-19p13.3-AD-SCN-Cyclic-Neutropenia-GCSF-Standard · SBDS-250aa-7q11.21-AR-Shwachman-Diamond-EPI-PERT-Mandatory ·
        GATA2-480aa-3q21.3-AD-MonoMAC-Monocytes-Pathognomonic-HSCT-Curative ·
        320 patients (8×40, seeds 1438–1445)
      </p>

      {error && (
        <div style={{ background: '#ffebee', border: '1px solid #ef9a9a', borderRadius: 6, padding: 12, marginBottom: 16, color: '#b71c1c' }}>
          Backend error: {error}
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 6, cursor: 'pointer', border: 'none',
            background: tab === i ? '#1a237e' : '#eee',
            color: tab === i ? '#fff' : '#333',
            fontWeight: tab === i ? 700 : 400, fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <GeneTableTab data={breakdown} />}
      {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
