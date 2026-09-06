'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SPAST:   '#1a237e',  // deep navy — most common AD-HSP, spastin
  ATL1:    '#1b5e20',  // deep green — childhood-onset atlastin
  REEP1:   '#37474f',  // dark slate — ER membrane curvature
  SPG11:   '#b71c1c',  // deep red — most common AR-HSP, TCC
  ZFYVE26: '#880e4f',  // deep magenta — Kjellin, macular degeneration
  CYP7B1:  '#e65100',  // deep orange — oxysterol accumulation, CDCA therapy
  KIF1A:   '#4a148c',  // deep purple — kinesin motor, KAND
  DDHD2:   '#006064',  // deep teal — brain lipid droplets, MRS
};

const GENE_DISEASE = {
  SPAST:   'SPG4 AD — Most Common ~40% — Incomplete Penetrance — MLPA for Deletions',
  ATL1:    'SPG3A AD — Childhood Onset — Pure HSP Mild — Atlastin ER Fusion',
  REEP1:   'SPG31 AD — ER Tubule Shaping — CMT2B5 Overlap — MLPA Required',
  SPG11:   'SPG11 AR — Most Common AR-HSP — TCC >90% — Intellectual Disability — Spatacsin',
  ZFYVE26: 'SPG15 AR — Kjellin Syndrome — Macular Degeneration — Annual Fundus',
  CYP7B1:  'SPG5A AR — Oxysterol Accumulation — CDCA Therapy — Plasma Biomarker',
  KIF1A:   'SPG30 AR (Mild) / KAND AD (Severe) — Kinesin Motor — AR≠AD Prognosis',
  DDHD2:   'SPG54 AR — Brain Lipid Droplets — MRS 1.3ppm Pathognomonic',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|PATHOGNOMONIC/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED/i.test(text);
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
        <h3 style={{ color: '#1a237e', marginBottom: 12 }}>Aggregate — 320 Patients (8×40, seeds 1446–1453)</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <tbody>
            {[
              ['Progressive Spastic Paraplegia', s.progressive_spastic_paraplegia_pct, '%'],
              ['Bilateral Leg Spasticity', s.bilateral_leg_spasticity_pct, '%'],
              ['Thin Corpus Callosum on MRI (AR-HSP)', s.thin_corpus_callosum_pct, '%'],
              ['Intellectual Disability (SPG11/SPG15/KAND)', s.intellectual_disability_pct, '%'],
              ['Macular Degeneration (ZFYVE26)', s.macular_degeneration_pct, '%'],
              ['Elevated Plasma Oxysterols (CYP7B1)', s.elevated_oxysterols_pct, '%'],
              ['Brain Lipid Peak MRS (DDHD2)', s.brain_lipid_peak_mrs_pct, '%'],
              ['Bladder Involvement', s.bladder_involvement_pct, '%'],
              ['De Novo Variant (KIF1A-KAND)', s.de_novo_kif1a_pct, '%'],
              ['Physiotherapy Required', s.physiotherapy_required_pct, '%'],
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

export default function HSPAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-spastic-paraplegia-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
    fetch(`${API}/api/hereditary-spastic-paraplegia-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/hereditary-spastic-paraplegia-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto', padding: '24px 16px' }}>
      <h1 style={{ color: '#1a237e', marginBottom: 4, fontSize: 22 }}>
        🧬 Hereditary Spastic Paraplegia Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 20, fontSize: 14 }}>
        Complete 8-Gene HSP-Atlas · SPAST-616aa-2p22.3-AD-SPG4-Most-Common-40pct-Incomplete-Penetrance-Microtubule-Severing ·
        ATL1-558aa-14q22.1-AD-SPG3A-Childhood-Onset-Pure-HSP-Mild-ER-Fusion ·
        REEP1-201aa-2p11.2-AD-SPG31-ER-Tubule-Shaping-CMT2B5-Overlap ·
        SPG11-2443aa-15q21.1-AR-Most-Common-AR-HSP-TCC-Intellectual-Disability-Spatacsin ·
        ZFYVE26-2539aa-14q24.1-AR-SPG15-Kjellin-Macular-Degeneration-Annual-Fundus ·
        CYP7B1-506aa-8q12.3-AR-SPG5A-Oxysterol-Accumulation-CDCA-Therapy-Biomarker ·
        KIF1A-1826aa-2q37.3-AR-SPG30-AD-KAND-Kinesin-Motor-AR-vs-AD-Different-Prognosis ·
        DDHD2-711aa-8p11.23-AR-SPG54-Brain-Lipid-Droplets-MRS-1.3ppm-Pathognomonic ·
        320 patients (8×40, seeds 1446–1453)
      </p>

      {error && (
        <div style={{ background: '#ffebee', border: '1px solid #ef9a9a', borderRadius: 6, padding: 12, marginBottom: 16, color: '#b71c1c' }}>
          Backend error: {error}
        </div>
      )}

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
