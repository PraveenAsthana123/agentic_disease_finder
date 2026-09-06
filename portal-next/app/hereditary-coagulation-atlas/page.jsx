'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  F8:        '#b71c1c',  // deep red — haemophilia A, most common, emicizumab
  F9:        '#c62828',  // crimson — haemophilia B, Leyden/Padua, gene therapy
  VWF:       '#880e4f',  // deep pink — most common inherited bleeding disorder
  F11:       '#4a148c',  // deep purple — Ashkenazi founder, mucosal paradox
  F13A1:     '#1a237e',  // deep navy — FXIII, umbilical cord, normal coag screen
  F7:        '#e65100',  // deep orange — shortest t½, PT alone, rFVIIa
  FGA:       '#1b5e20',  // deep green — fibrinogen, platelet + clot dual defect
  ADAMTS13:  '#212121',  // near-black — TTP/Upshaw-Schulman, plasma exchange life-saving
};

const GENE_DISEASE = {
  F8:        'Haemophilia A XLR — FVIII Deficiency 1:5000-10000 males — Emicizumab SC Prophylaxis FDA 2017 — Inhibitors 25-30% — Valoctocogene Gene Therapy EMA 2022 — Intron 22 Inversion 40-50%',
  F9:        'Haemophilia B XLR (Christmas Disease) — FIX Deficiency 1:25000 males — Leyden Variant Recovery After Puberty — Etranacogene Dezaparvovec FDA 2022 — Inhibitors Rare <5% BUT Anaphylaxis Risk',
  VWF:       'von Willebrand Disease AD/AR — Most Common Inherited Bleeding Disorder 1:100-1000 — DDAVP Type 1/2A Only — DDAVP CONTRAINDICATED Type 2B (Thrombocytopenia) — Type 3 VWF Concentrate Only',
  F11:       'Haemophilia C AR — FXI Deficiency — Ashkenazi 1:450 Lys521Ter Founder — CONCENTRATION-INDEPENDENT Bleeding Mucosal > Trauma — Tranexamic Acid FIRST-LINE — Concentrate Thrombogenic Elderly',
  F13A1:     'FXIII Deficiency AR — 1:5M — Umbilical Cord Stump Bleeding PATHOGNOMONIC — Normal PT/APTT/TT TRAP — ICH 25% Lifetime HIGHEST Rare Coagulopathy — Catridecacog Q4W Prophylaxis MANDATORY',
  F7:        'FVII Deficiency AR — Most Common Rare Coagulopathy 1:500000 — SHORTEST Plasma Half-Life 4-6h — PT Prolonged Alone APTT Normal — rFVIIa 15-30 mcg/kg Q4-6h — Level-Bleeding Poor Correlation',
  FGA:       'Afibrinogenemia AR — Fibrinogen Zero — ALL Coag Tests Prolonged — Platelet Aggregation Defect DUAL — Miscarriage 80% Untreated — Fibrinogen Concentrate 70 mg/kg — ICH Prophylaxis Mandatory',
  ADAMTS13:  'Congenital TTP / Upshaw-Schulman AR — MAHA + Thrombocytopenia — PENTAD NOT REQUIRED — PLASMA EXCHANGE LIFE-SAVING Start Immediately — Prophylactic FFP Q2-3W — Caplacizumab Acute Adjunct',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /URGENT|ABSOLUTE|NEVER|STOP|FATAL|LETHAL|PATHOGNOMONIC|MANDATORY|CONTRAINDICATED|AVOID|FIRST\s+LINE|HIGHEST|LIFE-SAVING|IMMEDIATELY|PROHIBITED|PROPHYLAXIS/i.test(text);
  const isWarning = /MONITOR|SCREEN|ANNUAL|REQUIRED|PROTOCOL|CONTINUOUS|LIFELONG|IMMEDIATELY|CASCADE|DISTINGUISH|START|ASSAY|PARTIAL|BEFORE\s+SURGERY|CONSIDER|TRIAL|RISING|PARADOX/i.test(text);
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
    <div>
      <h2 style={{ color: '#b71c1c', marginBottom: 8 }}>Hereditary Coagulation Atlas — 8-Gene Reference</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>
        320 patients (8 × 40, seeds 1574–1581) · F8 / F9 / VWF / F11 / F13A1 / F7 / FGA / ADAMTS13
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(210px,1fr))', gap: 12, marginBottom: 24 }}>
        {[
          ['Total Patients', s.total_patients],
          ['Mean Dx Age (y)', s.mean_dx_age_years],
          ['Mean Dx Delay (m)', s.mean_dx_delay_months],
          ['Cascade Tested %', s.cascade_tested_pct + '%'],
          ['F8 Inhibitor %', s.f8_inhibitor_pct + '%'],
          ['F8 Emicizumab Prophylaxis %', s.f8_emicizumab_prophylaxis_pct + '%'],
          ['vWD DDAVP Responsive %', s.vwf_ddavp_responsive_pct + '%'],
          ['F11 Tranexamic Acid Response %', s.f11_tranexamic_acid_responsive_pct + '%'],
          ['FXIII ICH Lifetime %', s.f13a1_ich_lifetime_pct + '%'],
          ['FVII ICH Neonatal %', s.f7_ich_neonatal_pct + '%'],
          ['FGA Miscarriage %', s.fga_miscarriage_pct + '%'],
          ['ADAMTS13 PEX Responsive %', s.adamts13_pex_responsive_pct + '%'],
        ].map(([label, value]) => (
          <div key={label} style={{
            background: '#fafafa', borderRadius: 8, padding: '12px 16px',
            borderLeft: '4px solid #b71c1c',
          }}>
            <div style={{ fontSize: 12, color: '#888' }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#b71c1c' }}>{value}</div>
          </div>
        ))}
      </div>

      <h3 style={{ marginBottom: 12 }}>Top Clinical Alerts</h3>
      {top_alerts?.slice(0, 12).map((a, i) => <AlertBadge key={i} text={a} />)}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Gene Summary</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#ffebee' }}>
              {['Gene', 'Protein (short)', 'Locus', 'Inheritance', 'OMIM Disease', 'Mean Dx Age', 'N'].map(h => (
                <th key={h} style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #b71c1c' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes?.map(g => (
              <tr key={g.gene} style={{ borderBottom: '1px solid #f0f0f0' }}>
                <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.protein_short}</td>
                <td style={{ padding: '6px 10px' }}>{g.locus}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.inheritance}</td>
                <td style={{ padding: '6px 10px' }}>{g.omim_disease}</td>
                <td style={{ padding: '6px 10px' }}>{g.mean_dx_age}y</td>
                <td style={{ padding: '6px 10px' }}>{g.n_patients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { genes } = data;
  return (
    <div>
      <h3 style={{ marginBottom: 16 }}>Gene Disease Associations</h3>
      <div style={{ display: 'grid', gap: 12 }}>
        {genes?.map(g => (
          <div key={g.gene} style={{
            padding: 14, borderRadius: 8, background: '#fafafa',
            borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#333'}`,
          }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#333', fontSize: 15 }}>
              {g.gene} — {g.locus}
            </div>
            <div style={{ fontSize: 13, color: '#444', marginTop: 4 }}>
              {GENE_DISEASE[g.gene]}
            </div>
            <div style={{ fontSize: 12, color: '#777', marginTop: 4 }}>
              OMIM: {g.omim_disease} · {g.inheritance} · Mean Dx Age: {g.mean_dx_age}y · N={g.n_patients}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  if (!selected && genes.length) setTimeout(() => setSelected(genes[0]), 0);
  const info = selected ? data[selected] : null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 16 }}>
      <div>
        {genes.map(g => (
          <button key={g} onClick={() => setSelected(g)} style={{
            display: 'block', width: '100%', padding: '8px 12px', marginBottom: 4,
            borderRadius: 6, border: 'none', cursor: 'pointer', textAlign: 'left',
            background: selected === g ? (GENE_COLORS[g] || '#b71c1c') : '#f0f0f0',
            color: selected === g ? '#fff' : '#333',
            fontWeight: selected === g ? 700 : 400, fontSize: 13,
          }}>{g}</button>
        ))}
      </div>
      {info && (
        <div>
          <h3 style={{ color: GENE_COLORS[selected] || '#b71c1c', marginBottom: 4 }}>
            {info.gene} — {info.aa} · {info.kDa} · {info.locus}
          </h3>
          <p style={{ fontSize: 13, color: '#666', marginBottom: 12 }}>
            OMIM Gene: {info.omim_gene} · Disease: {info.omim_disease} · {info.inheritance}
          </p>
          <div style={{ background: '#fafafa', borderRadius: 8, padding: 12, marginBottom: 16, fontSize: 13, lineHeight: 1.6 }}>
            {info.alias}
          </div>
          <div style={{ background: '#ffebee', borderRadius: 8, padding: 12, marginBottom: 16, fontSize: 12, lineHeight: 1.5 }}>
            <strong>Molecular Class: </strong>{info.gene_class}
          </div>

          <h4 style={{ marginBottom: 8 }}>Variant Distribution</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
            {Object.entries(info.etiologies || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
              <span key={k} style={{
                background: GENE_COLORS[selected] || '#b71c1c', color: '#fff',
                borderRadius: 20, padding: '4px 12px', fontSize: 12,
              }}>{k}: {v}</span>
            ))}
          </div>

          <h4 style={{ marginBottom: 8 }}>Clinical Feature Rates</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(180px,1fr))', gap: 8, marginBottom: 16 }}>
            {Object.entries(info.stats || {}).filter(([k]) => !['mean_dx_age','mean_dx_delay_months','mean_dx_age_months'].includes(k)).map(([k, v]) => (
              <div key={k} style={{ background: '#fafafa', padding: '8px 12px', borderRadius: 6, borderLeft: `3px solid ${GENE_COLORS[selected] || '#b71c1c'}` }}>
                <div style={{ fontSize: 11, color: '#888' }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[selected] || '#b71c1c' }}>{typeof v === 'number' ? (v > 10 ? v + '%' : v) : v}</div>
              </div>
            ))}
          </div>

          <h4 style={{ marginBottom: 8 }}>Diagnosis Delay Distribution</h4>
          <div style={{ marginBottom: 16 }}>
            {Object.entries(info.dx_delay_distribution || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <span style={{ width: 55, fontSize: 12, color: '#555' }}>{k}</span>
                <div style={{ flex: 1, background: '#e0e0e0', borderRadius: 4, height: 14 }}>
                  <div style={{ width: `${(v / info.n_patients) * 100}%`, background: GENE_COLORS[selected] || '#b71c1c', height: 14, borderRadius: 4 }} />
                </div>
                <span style={{ fontSize: 12, width: 24 }}>{v}</span>
              </div>
            ))}
          </div>

          <h4 style={{ marginBottom: 8 }}>Key Clinical Alerts</h4>
          {info.key_alerts?.map((a, i) => <AlertBadge key={i} text={a} />)}

          <h4 style={{ marginBottom: 8, marginTop: 12 }}>Sample Patients (first 10)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#e0e0e0' }}>
                  {Object.keys(info.patients?.[0] || {}).map(k => (
                    <th key={k} style={{ padding: '4px 8px', textAlign: 'left' }}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {info.patients?.map(p => (
                  <tr key={p.id} style={{ borderBottom: '1px solid #eee' }}>
                    {Object.values(p).map((v, i) => (
                      <td key={i} style={{ padding: '4px 8px' }}>
                        {typeof v === 'boolean' ? (v ? '✓' : '✗') : v?.toString() ?? '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ marginBottom: 16 }}>Hereditary Coagulation — Key Concepts</h3>
      {Object.entries(data.concepts || {}).map(([k, v]) => (
        <div key={k} style={{ marginBottom: 16, padding: 14, background: '#fafafa', borderRadius: 8, borderLeft: '4px solid #b71c1c' }}>
          <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>{k}</div>
          <div style={{ fontSize: 13, lineHeight: 1.6, color: '#333' }}>{v}</div>
        </div>
      ))}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Pharmacological Distinctions</h3>
      {data.pharmacological_distinctions?.map((d, i) => (
        <div key={i} style={{ marginBottom: 8, padding: '10px 14px', background: '#fff3e0', borderRadius: 8, borderLeft: '4px solid #e65100', fontSize: 13 }}>
          {d}
        </div>
      ))}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Key Standards & References</h3>
      {data.key_standards?.map((s, i) => (
        <div key={i} style={{ marginBottom: 6, fontSize: 13, color: '#444', paddingLeft: 12, borderLeft: '3px solid #b71c1c' }}>
          {s}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryCoagulationAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-coagulation-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-coagulation-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-coagulation-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <div style={{ padding: '2rem', color: 'red' }}>Error: {error}</div>;

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>
          🩸 Hereditary Coagulation Atlas
        </h1>
        <p style={{ color: '#666', fontSize: 14 }}>
          Complete 8-Gene Hereditary Coagulation Factor Deficiency Reference — F8 (Haemophilia A, Emicizumab) ·
          F9 (Haemophilia B, Christmas Disease) · VWF (von Willebrand Disease) · F11 (Haemophilia C, Ashkenazi) ·
          F13A1 (FXIII, Umbilical Cord Stump) · F7 (FVII, Shortest t½) · FGA (Afibrinogenemia) ·
          ADAMTS13 (Congenital TTP) · 320 patients (8×40, seeds 1574–1581)
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: '8px 8px 0 0', border: 'none', cursor: 'pointer',
            background: tab === t ? '#b71c1c' : '#f0f0f0',
            color: tab === t ? '#fff' : '#444', fontWeight: tab === t ? 700 : 400,
            fontSize: 14, borderBottom: tab === t ? '2px solid #b71c1c' : 'none',
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={overview} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
