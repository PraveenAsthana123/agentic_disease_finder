'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  GJB2:    '#1a237e',  // deep navy — most common SNHL gene
  SLC26A4: '#1b5e20',  // deep green — EVA/Pendred
  OTOF:    '#e65100',  // deep orange — ANSD synaptic
  COCH:    '#37474f',  // dark slate — adult progressive AD
  TMC1:    '#4a148c',  // deep purple — MET channel gene therapy
  MYO7A:   '#880e4f',  // deep magenta — Usher 1B most common
  CDH23:   '#bf360c',  // dark ember — Usher 1D tip-link upper
  PCDH15:  '#006064',  // deep teal — Usher 1F Ashkenazi R245X
};

const GENE_DISEASE = {
  GJB2:    'DFNB1 — Connexin 26 — Most Common SNHL — 35delG/235delC — CI Excellent',
  SLC26A4: 'DFNB4/Pendred — Pendrin — EVA — Head Trauma AVOID',
  OTOF:    'DFNB9 — Otoferlin — ANSD Pre-Neural — CI Works',
  COCH:    'DFNA9 — Cochlin — Adult Progressive — No Cure',
  TMC1:    'DFNB7/DFNA36 — MET Channel — M298K AD — Gene Therapy 2024',
  MYO7A:   'Usher 1B — Myosin VIIA — Deaf + RP + Vestibular',
  CDH23:   'Usher 1D / DFNB12 — Cadherin 23 — Tip-Link Upper',
  PCDH15:  'Usher 1F / DFNB23 — Protocadherin 15 — R245X Ashkenazi 1/148',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /CI|CONTRAINDICATED|AVOID|ABSOLUTE|MANDATORY|PROHIBITED/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN/i.test(text);
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
  const { aggregate_stats: s, top_alerts, diseases, genes } = data;
  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
      <div style={{ flex: '1 1 340px' }}>
        <h3 style={{ color: '#1a237e', marginBottom: 12 }}>Aggregate — 320 Patients (8×40, seeds 1430–1437)</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <tbody>
            {[
              ['Congenital Profound SNHL', s.congenital_profound_snhl, '%'],
              ['CI Implanted', s.ci_implanted, '%'],
              ['Retinitis Pigmentosa (Usher)', s.retinitis_pigmentosa, '%'],
              ['Vestibular Areflexia (Usher)', s.vestibular_areflexia, '%'],
              ['Bilateral Symmetric Loss', s.bilateral_symmetric_loss, '%'],
              ['Progressive SNHL (COCH)', s.progressive_snhl, '%'],
              ['Hearing Aid User', s.hearing_aid_user, '%'],
              ['ANSD Phenotype (OTOF)', s.ansd_phenotype, '%'],
              ['Fluctuating HL (EVA)', s.fluctuating_hearing_loss_eva, '%'],
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
            borderLeft: `4px solid ${GENE_COLORS[g] || '#999'}`,
            paddingLeft: 10, marginBottom: 8,
          }}>
            <span style={{ fontWeight: 700, color: GENE_COLORS[g] || '#333' }}>{g}</span>
            <span style={{ color: '#555', fontSize: 13, marginLeft: 8 }}>
              {diseases?.[g] || GENE_DISEASE[g]}
            </span>
          </div>
        ))}
      </div>

      <div style={{ flex: '1 1 340px' }}>
        <h3 style={{ color: '#b71c1c', marginBottom: 12 }}>⚠ Clinical Alerts (12 Rules)</h3>
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
          <tr style={{ background: '#e8eaf6' }}>
            {['Gene', 'Protein', 'aa', 'Locus', 'OMIM Gene', 'OMIM Disease', 'Inheritance', 'N'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #9fa8da' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {genes.map((g, i) => (
            <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f8f9ff', borderBottom: '1px solid #e8eaf6' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
              <td style={{ padding: '7px 10px', maxWidth: 200, fontSize: 12 }}>{g.protein}</td>
              <td style={{ padding: '7px 10px' }}>{g.aa}</td>
              <td style={{ padding: '7px 10px', fontFamily: 'monospace' }}>{g.locus}</td>
              <td style={{ padding: '7px 10px' }}>{g.omim_gene}</td>
              <td style={{ padding: '7px 10px' }}>{g.omim_disease}</td>
              <td style={{ padding: '7px 10px', fontSize: 12 }}>{g.inheritance?.slice(0, 60)}</td>
              <td style={{ padding: '7px 10px', fontWeight: 700 }}>{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  const cur = selected ? data[selected] : null;

  return (
    <div style={{ display: 'flex', gap: 20 }}>
      <div style={{ width: 180, flexShrink: 0 }}>
        {genes.map(g => (
          <button key={g} onClick={() => setSelected(g)} style={{
            display: 'block', width: '100%', textAlign: 'left',
            padding: '8px 12px', marginBottom: 4, border: 'none',
            borderRadius: 6, cursor: 'pointer', fontWeight: 600,
            background: selected === g ? (GENE_COLORS[g] || '#1a237e') : '#e8eaf6',
            color: selected === g ? '#fff' : (GENE_COLORS[g] || '#333'),
          }}>
            {g}
          </button>
        ))}
      </div>
      <div style={{ flex: 1 }}>
        {!cur ? (
          <p style={{ color: '#888' }}>Select a gene to view clinical details.</p>
        ) : (
          <>
            <h3 style={{ color: GENE_COLORS[cur.gene] || '#1a237e', marginBottom: 4 }}>
              {cur.gene} — {cur.protein}
            </h3>
            <p style={{ fontSize: 13, color: '#555', marginBottom: 12 }}>
              {cur.locus} · {cur.aa} · {cur.inheritance?.slice(0, 80)}
            </p>

            <h4 style={{ color: '#1a237e' }}>Gene Class</h4>
            <p style={{ fontSize: 13, lineHeight: 1.6 }}>{cur.gene_class}</p>

            <h4 style={{ color: '#1a237e', marginTop: 16 }}>Key Hallmarks</h4>
            <ul style={{ fontSize: 13, lineHeight: 1.7, paddingLeft: 20 }}>
              {cur.hallmarks?.map((h, i) => <li key={i}>{h}</li>)}
            </ul>

            <h4 style={{ color: '#b71c1c', marginTop: 16 }}>⚠ Treatment Alerts</h4>
            {cur.treatment_alerts?.map((a, i) => <AlertBadge key={i} text={a} />)}

            <h4 style={{ color: '#1b5e20', marginTop: 16 }}>Primary Treatment</h4>
            <p style={{ fontSize: 13, lineHeight: 1.6 }}>{cur.primary_treatment}</p>

            {cur.etiology_distribution && (
              <>
                <h4 style={{ color: '#1a237e', marginTop: 16 }}>Aetiology Distribution</h4>
                {cur.etiology_distribution.map((e, i) => (
                  <div key={i} style={{ marginBottom: 6 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                      <span>{e.etiology}</span>
                      <span style={{ fontWeight: 700, color: GENE_COLORS[cur.gene] }}>{(e.fraction * 100).toFixed(0)}%</span>
                    </div>
                    <div style={{ height: 6, background: '#e8eaf6', borderRadius: 3, marginTop: 2 }}>
                      <div style={{ height: 6, width: `${e.fraction * 100}%`, background: GENE_COLORS[cur.gene] || '#1a237e', borderRadius: 3 }} />
                    </div>
                  </div>
                ))}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { classification, key_diagnostic_rules, treatment_hierarchy } = data;
  return (
    <div>
      <h3 style={{ color: '#1a237e' }}>Disease Classification</h3>
      {classification && Object.entries(classification).map(([cat, entries]) => (
        <div key={cat} style={{ marginBottom: 16 }}>
          <h4 style={{ color: '#3949ab', textTransform: 'capitalize', marginBottom: 6 }}>
            {cat.replace(/_/g, ' ')}
          </h4>
          {Object.entries(entries).map(([disease, desc]) => (
            <div key={disease} style={{ marginBottom: 6, paddingLeft: 16, borderLeft: '3px solid #9fa8da' }}>
              <span style={{ fontWeight: 700, color: '#1a237e' }}>{disease.replace(/_/g, ' ')}</span>
              <span style={{ color: '#555', fontSize: 13, marginLeft: 8 }}>{desc}</span>
            </div>
          ))}
        </div>
      ))}

      <h3 style={{ color: '#b71c1c', marginTop: 24 }}>Key Diagnostic Rules</h3>
      {key_diagnostic_rules && Object.entries(key_diagnostic_rules).map(([rule, text]) => (
        <div key={rule} style={{ marginBottom: 16, background: '#fff8f8', border: '1px solid #ffcdd2', borderRadius: 8, padding: 14 }}>
          <h4 style={{ color: '#b71c1c', marginBottom: 6, fontSize: 14 }}>{rule.replace(/_/g, ' ')}</h4>
          <p style={{ fontSize: 13, lineHeight: 1.6, margin: 0, color: '#333' }}>{text}</p>
        </div>
      ))}

      <h3 style={{ color: '#1b5e20', marginTop: 24 }}>Treatment Hierarchies</h3>
      {treatment_hierarchy && Object.entries(treatment_hierarchy).map(([gene, steps]) => (
        <div key={gene} style={{ marginBottom: 16 }}>
          <h4 style={{ color: '#1b5e20' }}>{gene.replace(/_/g, ' ')}</h4>
          <ol style={{ fontSize: 13, lineHeight: 1.7, paddingLeft: 20 }}>
            {steps.map((s, i) => <li key={i}>{s}</li>)}
          </ol>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryHearingLossAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-hearing-loss-atlas`;
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

  if (error) return <div style={{ padding: '2rem', color: '#b71c1c' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ color: '#1a237e', fontSize: 22, marginBottom: 4 }}>
          Hereditary Hearing Loss Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 14, margin: 0 }}>
          Complete 8-Gene Hereditary Sensorineural Hearing Loss Atlas ·
          GJB2 · SLC26A4 · OTOF · COCH · TMC1 · MYO7A · CDH23 · PCDH15 ·
          320 patients (8×40, seeds 1430–1437)
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e8eaf6', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '7px 18px', border: 'none', borderRadius: '6px 6px 0 0',
            background: tab === t ? '#1a237e' : '#e8eaf6',
            color: tab === t ? '#fff' : '#333',
            fontWeight: tab === t ? 700 : 400,
            cursor: 'pointer', fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Overview'      && <OverviewTab data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
    </div>
  );
}
