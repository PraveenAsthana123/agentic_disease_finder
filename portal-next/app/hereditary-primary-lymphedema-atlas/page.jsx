'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FLT4:    '#01579b',  // deep blue — Milroy/VEGFR3 congenital
  FOXC2:   '#1b5e20',  // deep green — distichiasis-lymphedema
  PROX1:   '#4a148c',  // deep purple — master LEC fate regulator
  GJC2:    '#37474f',  // dark slate — gap junction late-onset
  SOX18:   '#bf360c',  // deep burnt orange — HLTRS ragged dominant negative
  CCBE1:   '#880e4f',  // deep magenta — Hennekam HLS1
  KIF11:   '#006064',  // deep teal — MCLMR microcephaly
  ADAMTS3: '#33691e',  // deep olive — Hennekam HLS3 VEGF-C protease
};

const GENE_DISEASE = {
  FLT4:    'Milroy Disease / PCL-1 AD — Dominant Negative TK Domain — Congenital Bilateral Oedema',
  FOXC2:   'Lymphedema-Distichiasis LDS AD — Distichiasis Pathognomonic — Pubertal Onset — Corneal Risk',
  PROX1:   'HLTS AD — Master LEC Fate — Hypotrichosis + Lymphedema + Telangiectasia Triad',
  GJC2:    'Late-Onset Lymphedema AR / SPG44 Leukodystrophy Biallelic LOF — Cx47 Gap Junction',
  SOX18:   'HLTRS AD — Ragged HMG-Box Dominant Negative — Renal Anomalies 30% — Annual US',
  CCBE1:   'Hennekam HLS1 AR — Intestinal Lymphangiectasia — Protein-Losing Enteropathy — ID 50%',
  KIF11:   'MCLMR AD — Microcephaly + Chorioretinopathy + Lymphedema — De Novo — OCT Mandatory',
  ADAMTS3: 'Hennekam HLS3 AR — VEGF-C Protease — Phenocopy CCBE1 — Test Both Simultaneously',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|PATHOGNOMONIC/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|SIMULTANEOUSLY/i.test(text);
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
        <h3 style={{ color: '#01579b', marginBottom: 12 }}>Aggregate — 320 Patients (8×40, seeds 1454–1461)</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <tbody>
            {[
              ['Lower Limb Lymphedema (any gene)', s.lower_limb_lymphedema_pct, '%'],
              ['Congenital Onset (FLT4 / Hennekam)', s.congenital_onset_pct, '%'],
              ['Intestinal Lymphangiectasia (Hennekam)', s.intestinal_lymphangiectasia_pct, '%'],
              ['Protein-Losing Enteropathy', s.protein_losing_enteropathy_pct, '%'],
              ['Intellectual Disability (Hennekam / MCLMR)', s.intellectual_disability_pct, '%'],
              ['Distichiasis (FOXC2 — pathognomonic)', s.distichiasis_foxc2_pct, '%'],
              ['Hypotrichosis (PROX1 / SOX18)', s.hypotrichosis_pct, '%'],
              ['Primary Microcephaly (KIF11)', s.microcephaly_kif11_pct, '%'],
              ['Chorioretinopathy (KIF11)', s.chorioretinopathy_pct, '%'],
              ['MCT Diet Required (Hennekam)', s.mct_diet_required_pct, '%'],
            ].map(([label, val, unit]) => (
              <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '6px 8px', color: '#555' }}>{label}</td>
                <td style={{ padding: '6px 8px', fontWeight: 700, color: '#01579b' }}>
                  {val !== undefined && val !== null ? `${val}${unit}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3 style={{ color: '#01579b', marginTop: 20, marginBottom: 10 }}>8 Genes — Disease Associations</h3>
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
          <tr style={{ background: '#01579b', color: '#fff' }}>
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
            background: selected === gene ? (GENE_COLORS[gene] || '#01579b') : '#eee',
            color: selected === gene ? '#fff' : '#333',
            fontWeight: selected === gene ? 700 : 400, fontSize: 13,
          }}>{gene}</button>
        ))}
      </div>

      <div style={{ flex: 1 }}>
        <h3 style={{ color: GENE_COLORS[selected] || '#01579b', marginBottom: 4 }}>{g.gene} — {g.protein}</h3>
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
                <td style={{ padding: '4px 8px', fontWeight: 700, color: '#01579b' }}>{v}%</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h4 style={{ color: '#555', marginTop: 14, marginBottom: 6 }}>Etiology Distribution</h4>
        {g.etiologies && g.etiologies.map((e, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
            <div style={{
              width: `${e.pct * 2.5}px`, minWidth: 4, height: 18,
              background: GENE_COLORS[selected] || '#01579b', borderRadius: 3, marginRight: 8,
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
          borderLeft: '4px solid #01579b', padding: '12px 16px',
          marginBottom: 16, background: '#f8f9fa', borderRadius: '0 6px 6px 0',
        }}>
          <h4 style={{ color: '#01579b', marginBottom: 6, fontSize: 15 }}>{d.term}</h4>
          <p style={{ fontSize: 13, color: '#333', lineHeight: 1.6, margin: 0 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryPrimaryLymphedemaAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-primary-lymphedema-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
    fetch(`${API}/api/hereditary-primary-lymphedema-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/hereditary-primary-lymphedema-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto', padding: '24px 16px' }}>
      <h1 style={{ color: '#01579b', marginBottom: 4, fontSize: 22 }}>
        🫧 Hereditary Primary Lymphedema Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 20, fontSize: 14 }}>
        Complete 8-Gene Hereditary-Primary-Lymphedema-Atlas ·
        FLT4-1363aa-5q35.3-AD-Milroy-PCL1-Dominant-Negative-TK-Congenital-Onset ·
        FOXC2-501aa-16q24.1-AD-LDS-Distichiasis-Pathognomonic-Pubertal-Onset-Corneal-Risk ·
        PROX1-737aa-1q32.3-AD-HLTS-Master-LEC-Fate-Hypotrichosis-Lymphedema-Telangiectasia ·
        GJC2-436aa-1q42.13-AR-Late-Onset-Lymphedema-SPG44-Biallelic-LOF-Brain-MRI ·
        SOX18-384aa-20q13.33-AD-HLTRS-Ragged-HMG-Box-DN-Renal-Anomalies-Annual-US ·
        CCBE1-429aa-18q21.32-AR-Hennekam-HLS1-Intestinal-Lymphangiectasia-PLE-ID-50pct-MCT ·
        KIF11-1056aa-10q23.33-AD-MCLMR-Microcephaly-Chorioretinopathy-Lymphedema-De-Novo-OCT ·
        ADAMTS3-1232aa-4q13.3-AR-Hennekam-HLS3-VEGFC-Protease-Phenocopy-CCBE1-Test-Both ·
        320 patients (8×40, seeds 1454–1461)
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
            background: tab === i ? '#01579b' : '#eee',
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
