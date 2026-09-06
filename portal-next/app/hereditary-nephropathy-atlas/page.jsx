'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  COL4A5: '#1565c0',  // deep blue — X-linked Alport, most common hereditary nephritis
  COL4A3: '#0277bd',  // mid blue — AR Alport / TBMN
  NPHS1:  '#b71c1c',  // deep red — congenital nephrotic syndrome, massive proteinuria
  NPHS2:  '#c62828',  // red — FSGS2 steroid-resistant NS
  WT1:    '#4a148c',  // deep purple — DDS/Frasier, gonadoblastoma risk
  UMOD:   '#2e7d32',  // deep green — tubulointerstitial nephritis, gout
  PKD1:   '#e65100',  // deep orange — ADPKD, most common hereditary kidney disease
  TRPC6:  '#880e4f',  // deep pink — FSGS6 adult-onset GOF, donor exclusion
};

const GENE_DISEASE = {
  COL4A5: 'XL Alport XL ESRD 20-30y Males — Anterior Lenticonus PATHOGNOMONIC — ACEi ASAP Delays ESRD 13y — Skin Biopsy Fast Dx — MLPA Mandatory — AVOID Nephrotoxins',
  COL4A3: 'AR Alport / TBMN — Biallelic = Alport Severity — Monoallelic = TBMN 10-15% FSGS Risk — ACEi if Proteinuria — MLPA Mandatory — Partner Testing Before Conception',
  NPHS1:  'CNS Finnish Type AR — Large Placenta >25% BW PATHOGNOMONIC — NO Immunosuppression — Bilateral Nephrectomy + Transplant — Recurrence RARE — Fin-major/Fin-minor 95% Finnish',
  NPHS2:  'FSGS2 AR Steroid-RESISTANT PATHOGNOMONIC — R138Q Most Common European — R229Q Low-Penetrance Modifier — Post-Transplant Recurrence LOW — Cyclosporin Partial Response',
  WT1:    'DDS R394W Hotspot — DMS + DSD + Wilms — Frasier +KTS Splice — FSGS + Gonadoblastoma — NO Wilms — Gonadectomy MANDATORY 46XY — Annual Renal US Until Age 8 DDS',
  UMOD:   'FJHN/MCKD2 AD — Gout PRESENTING FEATURE Teenage — Most Common Hereditary Tubulointerstitial Nephritis — Urine UMOD Low — NSAIDs CONTRAINDICATED — Allopurinol NOT Uricosurics',
  PKD1:   'ADPKD Most Common Hereditary Kidney Disease 1:400 — Tolvaptan FDA 2018 V2R Antagonist — ICA MRA Mandatory Family History — Hepatic Cysts 60-80% — BP <130/80 ACEi Early',
  TRPC6:  'FSGS6 AD GOF Adult-Onset — Steroid-RESISTANT — Living-Donor EXCLUSION Carrier Relatives — Calcineurin Inhibitors Partial — P112Q/N143S/R895C GOF Variants',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /NO\s+IMMUNOSUPPRESSION|ABSOLUTELY\s+CONTRAINDICATED|ABSOLUTE|NEVER|STOP|FATAL|LETHAL|PATHOGNOMONIC|MANDATORY|CURATIVE|EXCLUSION|CONTRAINDICATED|AVOID|ASAP/i.test(text);
  const isWarning = /MONITOR|SCREEN|ANNUAL|REQUIRED|MANDATORY|PROTOCOL|FIRST\s+LINE|FIRST-LINE|CONTINUOUS|LIFELONG|IMMEDIATELY|CASCADE|DISTINGUISH|START|ASSAY|LOW\s+RECURRENCE|PARTIAL/i.test(text);
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
      <h2 style={{ color: '#1565c0', marginBottom: 8 }}>Hereditary Nephropathy Atlas — 8-Gene Reference</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>
        320 patients (8 × 40, seeds 1558–1565) · COL4A5 / COL4A3 / NPHS1 / NPHS2 / WT1 / UMOD / PKD1 / TRPC6
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 12, marginBottom: 24 }}>
        {[
          ['Total Patients', s.total_patients],
          ['Mean Dx Age (y)', s.mean_dx_age_years],
          ['Mean Dx Delay (m)', s.mean_dx_delay_months],
          ['Cascade Tested %', s.cascade_tested_pct + '%'],
          ['COL4A5 ACEi %', s.col4a5_acei_pct + '%'],
          ['COL4A5 SNHL %', s.col4a5_snhl_pct + '%'],
          ['NPHS1 Large Placenta %', s.nphs1_large_placenta_pct + '%'],
          ['NPHS2 Steroid-Resist %', s.nphs2_steroid_resistant_pct + '%'],
          ['WT1 Gonadectomy %', s.wt1_gonadectomy_pct + '%'],
          ['UMOD Gout %', s.umod_gout_pct + '%'],
          ['PKD1 ICA MRA Done %', s.pkd1_ica_mra_done_pct + '%'],
          ['TRPC6 Donor Excl %', s.trpc6_donor_excluded_pct + '%'],
        ].map(([label, value]) => (
          <div key={label} style={{
            background: '#f5f7fa', borderRadius: 8, padding: '12px 16px',
            borderLeft: '4px solid #1565c0',
          }}>
            <div style={{ fontSize: 12, color: '#888' }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#1565c0' }}>{value}</div>
          </div>
        ))}
      </div>

      <h3 style={{ marginBottom: 12 }}>Top Clinical Alerts</h3>
      <div style={{ columnCount: 2, columnGap: 16, marginBottom: 24 }}>
        {top_alerts.map((a, i) => (
          <div key={i} style={{ breakInside: 'avoid', marginBottom: 4 }}>
            <span style={{
              display: 'inline-block', background: GENE_COLORS[a.gene] || '#555',
              color: '#fff', borderRadius: 4, padding: '1px 6px', fontSize: 11, marginRight: 6,
            }}>{a.gene}</span>
            <AlertBadge text={a.alert} />
          </div>
        ))}
      </div>

      <h3 style={{ marginBottom: 12 }}>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 12 }}>
        {genes.map(g => (
          <div key={g.gene} style={{
            border: `2px solid ${GENE_COLORS[g.gene] || '#ccc'}`,
            borderRadius: 10, padding: 14,
          }}>
            <div style={{ fontWeight: 700, fontSize: 16, color: GENE_COLORS[g.gene] }}>{g.gene}</div>
            <div style={{ fontSize: 12, color: '#555', marginBottom: 6 }}>{g.aa} · {g.locus} · {g.inheritance}</div>
            <div style={{ fontSize: 12, marginBottom: 6 }}>{g.protein}</div>
            <div style={{ fontSize: 11, color: '#888' }}>
              {g.n_patients} pts · mean dx {g.mean_dx_age}y · delay {g.mean_dx_delay_months}m
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { genes } = data;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#1565c0', color: '#fff' }}>
            {['Gene', 'Protein', 'aa', 'kDa', 'Locus', 'OMIM Gene', 'OMIM Disease', 'Inheritance', 'N', 'Mean Dx Age', 'Mean Delay'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {genes.map((g, i) => (
            <tr key={g.gene} style={{ background: i % 2 ? '#f5f7fa' : '#fff' }}>
              <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
              <td style={{ padding: '6px 10px', maxWidth: 240, fontSize: 12 }}>{g.protein.split(' — ')[0]}</td>
              <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{g.aa}</td>
              <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{g.kDa}</td>
              <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{g.locus}</td>
              <td style={{ padding: '6px 10px' }}>{g.omim_gene}</td>
              <td style={{ padding: '6px 10px' }}>{g.omim_disease}</td>
              <td style={{ padding: '6px 10px', fontSize: 12, maxWidth: 180 }}>{g.inheritance.split(';')[0]}</td>
              <td style={{ padding: '6px 10px' }}>{g.n_patients}</td>
              <td style={{ padding: '6px 10px' }}>{g.mean_dx_age}y</td>
              <td style={{ padding: '6px 10px' }}>{g.mean_dx_delay_months}m</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Gene Disease Summaries</h3>
      {genes.map(g => (
        <div key={g.gene} style={{ marginBottom: 16, padding: 14, border: `2px solid ${GENE_COLORS[g.gene] || '#ccc'}`, borderRadius: 10 }}>
          <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene], marginBottom: 4 }}>{g.gene} — {GENE_DISEASE[g.gene]}</div>
          <div style={{ fontSize: 12, color: '#555' }}>{g.key_alerts.slice(0, 3).join(' · ')}</div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [gene, setGene] = useState(null);
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  const selected = gene || genes[0];
  const info = data[selected];
  return (
    <div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
        {genes.map(g => (
          <button key={g} onClick={() => setGene(g)} style={{
            padding: '6px 14px', borderRadius: 20, border: 'none', cursor: 'pointer',
            background: selected === g ? (GENE_COLORS[g] || '#1565c0') : '#e0e0e0',
            color: selected === g ? '#fff' : '#333', fontWeight: 600, fontSize: 13,
          }}>{g}</button>
        ))}
      </div>
      {info && (
        <div>
          <h3 style={{ color: GENE_COLORS[selected], marginBottom: 4 }}>{selected} — {info.protein}</h3>
          <div style={{ fontSize: 12, color: '#555', marginBottom: 12, lineHeight: 1.5 }}>{info.alias?.slice(0, 400)}…</div>
          <div style={{ fontSize: 12, color: '#444', background: '#f5f7fa', padding: 12, borderRadius: 8, marginBottom: 12, lineHeight: 1.5 }}>
            <strong>Gene class:</strong> {info.gene_class?.slice(0, 300)}…
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
            <div>
              <strong style={{ fontSize: 13 }}>Age at Diagnosis Distribution</strong>
              {Object.entries(info.age_at_diagnosis_distribution || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                  <span style={{ width: 50, fontSize: 12, color: '#555' }}>{k}</span>
                  <div style={{ flex: 1, background: '#e0e0e0', borderRadius: 4, height: 14 }}>
                    <div style={{ width: `${(v / info.n_patients) * 100}%`, background: GENE_COLORS[selected] || '#1565c0', height: 14, borderRadius: 4 }} />
                  </div>
                  <span style={{ fontSize: 12, width: 24 }}>{v}</span>
                </div>
              ))}
            </div>
            <div>
              <strong style={{ fontSize: 13 }}>Diagnosis Delay Distribution</strong>
              {Object.entries(info.dx_delay_distribution || {}).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                  <span style={{ width: 50, fontSize: 12, color: '#555' }}>{k}</span>
                  <div style={{ flex: 1, background: '#e0e0e0', borderRadius: 4, height: 14 }}>
                    <div style={{ width: `${(v / info.n_patients) * 100}%`, background: '#e65100', height: 14, borderRadius: 4 }} />
                  </div>
                  <span style={{ fontSize: 12, width: 24 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          <h4 style={{ marginBottom: 8 }}>Clinical Feature Rates</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(180px,1fr))', gap: 8, marginBottom: 16 }}>
            {Object.entries(info.stats || {}).filter(([k]) => !['mean_dx_age','mean_dx_delay_months'].includes(k)).map(([k, v]) => (
              <div key={k} style={{ background: '#f5f7fa', padding: '8px 12px', borderRadius: 6, borderLeft: `3px solid ${GENE_COLORS[selected] || '#1565c0'}` }}>
                <div style={{ fontSize: 11, color: '#888' }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: GENE_COLORS[selected] || '#1565c0' }}>{typeof v === 'number' ? (v > 10 ? v + '%' : v) : v}</div>
              </div>
            ))}
          </div>

          <h4 style={{ marginBottom: 8 }}>Variant Distribution</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
            {Object.entries(info.etiologies || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
              <span key={k} style={{
                background: GENE_COLORS[selected] || '#1565c0', color: '#fff',
                borderRadius: 20, padding: '4px 12px', fontSize: 12,
              }}>{k}: {v}</span>
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
      <h3 style={{ marginBottom: 16 }}>Hereditary Nephropathy — Key Concepts</h3>
      {Object.entries(data.concepts || {}).map(([k, v]) => (
        <div key={k} style={{ marginBottom: 16, padding: 14, background: '#f5f7fa', borderRadius: 8, borderLeft: '4px solid #1565c0' }}>
          <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
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
        <div key={i} style={{ marginBottom: 6, fontSize: 13, color: '#444', paddingLeft: 12, borderLeft: '3px solid #1565c0' }}>
          {s}
        </div>
      ))}
    </div>
  );
}

export default function HreditaryNephropathyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-nephropathy-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-nephropathy-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-nephropathy-atlas/definitions`).then(r => r.json()),
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
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#1565c0', marginBottom: 4 }}>
          🧬 Hereditary Nephropathy Atlas
        </h1>
        <p style={{ color: '#666', fontSize: 14 }}>
          Complete 8-Gene Hereditary Nephropathy Reference — COL4A5 (XL Alport) · COL4A3 (AR Alport/TBMN) ·
          NPHS1 (CNS-F) · NPHS2 (FSGS2) · WT1 (DDS/Frasier) · UMOD (FJHN/MCKD2) · PKD1 (ADPKD) · TRPC6 (FSGS6) ·
          320 patients (8×40, seeds 1558–1565)
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: '8px 8px 0 0', border: 'none', cursor: 'pointer',
            background: tab === t ? '#1565c0' : '#f0f0f0',
            color: tab === t ? '#fff' : '#444', fontWeight: tab === t ? 700 : 400,
            fontSize: 14, borderBottom: tab === t ? '2px solid #1565c0' : 'none',
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
