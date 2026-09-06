'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  COL7A1: '#37474f',  // dark slate — RDEB, anchoring fibrils, skin fragility
  LAMB3:  '#4a148c',  // deep purple — JEB, laminin-332, lethality
  KRT5:   '#0d47a1',  // deep blue — EB Simplex, keratin cytoskeleton
  ABCA12: '#b71c1c',  // deep red — Harlequin, neonatal emergency
  SPINK5: '#1b5e20',  // deep green — Netherton, LEKTI, IgE
  ATP2A2: '#e65100',  // deep orange — Darier, SERCA2, warty papules
  STS:    '#880e4f',  // deep pink — XLI, steroid sulfatase, corneal opacities
  IKBKG:  '#006064',  // teal — Incontinentia Pigmenti, NF-κB, XLD
};

const GENE_DISEASE = {
  COL7A1: 'Dystrophic Epidermolysis Bullosa AR/AD — Type VII Collagen — SCC Leading Death Cause — Beremagene FDA 2023',
  LAMB3:  'Junctional EB Herlitz (null/null LETHAL) / Non-Herlitz AR — Laminin-332 — Enamel Hypoplasia PATHOGNOMONIC',
  KRT5:   'Epidermolysis Bullosa Simplex AD — Keratin 5 — Dowling-Meara Severe — Heat Triggers — No DMT',
  ABCA12: 'Harlequin Ichthyosis AR — ABCA12 Lamellar Body Transport — Neonatal Emergency — Acitretin LIFE-SAVING',
  SPINK5: 'Netherton Syndrome AR — LEKTI → KLK5/7 Hyperactive — Trichorrhexis Invaginata PATHOGNOMONIC — IgE >2000 — Dupilumab',
  ATP2A2: 'Darier Disease AD — SERCA2 ER Calcium Pump — Seborrhoeic Warty Papules — Lithium CI — Isotretinoin First-Line',
  STS:    'X-Linked Ichthyosis XLR — Steroid Sulfatase — Corneal Opacities — Contiguous Kallmann (ANOS1) — Cryptorchidism 25%',
  IKBKG:  'Incontinentia Pigmenti XLD — NF-κB Essential Modulator — Males Lethal 80% — 4-Stage Skin — Retinal Vasculopathy Leading Blindness',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /URGENT|ABSOLUTE|NEVER|STOP|FATAL|LETHAL|PATHOGNOMONIC|MANDATORY|CONTRAINDICATED|AVOID|CURATIVE|LIFE-SAVING|IMMEDIATELY|PROHIBITED|PROPHYLAXIS|WORSENS/i.test(text);
  const isWarning = /MONITOR|SCREEN|ANNUAL|REQUIRED|PROTOCOL|CONTINUOUS|LIFELONG|CASCADE|DISTINGUISH|START|ASSAY|PARTIAL|BEFORE\s+SURGERY|CONSIDER|TRIAL|RISING|PARADOX|KEY\s+DDx|SURVEILLANCE/i.test(text);
  const bg = isCI ? '#37474f' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{
      background: '#fafafa', borderRadius: 8, padding: '12px 16px',
      borderLeft: `4px solid ${color || '#37474f'}`,
    }}>
      <div style={{ fontSize: 12, color: '#888' }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#37474f' }}>{value}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, genes } = data;
  return (
    <div>
      <h2 style={{ color: '#37474f', marginBottom: 8 }}>Hereditary Genodermatoses Atlas — 8-Gene Reference</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>
        320 patients (8 × 40, seeds 1590–1597) · COL7A1 / LAMB3 / KRT5 / ABCA12 / SPINK5 / ATP2A2 / STS / IKBKG
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(210px,1fr))', gap: 12, marginBottom: 24 }}>
        <StatCard label="Total Patients" value={s.total_patients} />
        <StatCard label="Mean Dx Age (y)" value={s.mean_dx_age_years} />
        <StatCard label="Mean Dx Delay (m)" value={s.mean_dx_delay_months} />
        <StatCard label="COL7A1 SCC Risk %" value={s.col7a1_scc_risk_pct + '%'} color="#37474f" />
        <StatCard label="LAMB3 Herlitz %" value={s.lamb3_herlitz_pct + '%'} color="#4a148c" />
        <StatCard label="KRT5 Dowling-Meara %" value={s.krt5_dowling_meara_pct + '%'} color="#0d47a1" />
        <StatCard label="ABCA12 Acitretin Response %" value={s.abca12_acitretin_response_pct + '%'} color="#b71c1c" />
        <StatCard label="SPINK5 IgE >2000 %" value={s.spink5_ige_over_2000_pct + '%'} color="#1b5e20" />
        <StatCard label="ATP2A2 Isotretinoin Response %" value={s.atp2a2_isotretinoin_response_pct + '%'} color="#e65100" />
        <StatCard label="STS Corneal Opacity %" value={s.sts_corneal_opacity_pct + '%'} color="#880e4f" />
        <StatCard label="IKBKG Retinal Vasculopathy %" value={s.ikbkg_retinal_vasculopathy_pct + '%'} color="#006064" />
        <StatCard label="Cascade Tested %" value={s.cascade_tested_pct + '%'} />
      </div>

      <h3 style={{ marginBottom: 12 }}>Top Clinical Alerts</h3>
      {top_alerts?.slice(0, 12).map((a, i) => <AlertBadge key={i} text={a} />)}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Gene Summary</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#eceff1' }}>
              {['Gene', 'Protein (short)', 'Locus', 'Inheritance', 'OMIM Disease', 'Mean Dx Age', 'N'].map(h => (
                <th key={h} style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #37474f' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes?.map(g => (
              <tr key={g.gene} style={{ borderBottom: '1px solid #f0f0f0' }}>
                <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.protein_short}</td>
                <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{g.locus}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.inheritance}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.omim_disease}</td>
                <td style={{ padding: '6px 10px' }}>{g.mean_dx_age}</td>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{g.n_patients}</td>
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
  const genes = Object.values(data);
  return (
    <div>
      <h2 style={{ color: '#37474f', marginBottom: 16 }}>Gene-Level Clinical Breakdown</h2>
      {genes.map(g => (
        <div key={g.gene} style={{
          marginBottom: 28, border: '1px solid #cfd8dc', borderRadius: 10,
          overflow: 'hidden',
        }}>
          <div style={{ background: GENE_COLORS[g.gene] || '#37474f', color: '#fff', padding: '10px 16px' }}>
            <strong>{g.gene}</strong> — {g.locus} — {g.aa} — {g.inheritance.split(';')[0]} — OMIM {g.omim_disease}
          </div>
          <div style={{ padding: 16 }}>
            <p style={{ fontSize: 13, color: '#555', marginBottom: 12, lineHeight: 1.5 }}>
              {g.alias?.slice(0, 500)}…
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <strong style={{ fontSize: 12, color: '#888' }}>Variant Classes</strong>
                <table style={{ width: '100%', fontSize: 12, marginTop: 4, borderCollapse: 'collapse' }}>
                  <tbody>
                    {Object.entries(g.etiologies || {}).map(([k, v]) => (
                      <tr key={k} style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <td style={{ padding: '3px 0', color: '#333' }}>{k}</td>
                        <td style={{ padding: '3px 0', fontWeight: 600, textAlign: 'right', color: GENE_COLORS[g.gene] }}>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div>
                <strong style={{ fontSize: 12, color: '#888' }}>Key Stats</strong>
                <table style={{ width: '100%', fontSize: 12, marginTop: 4, borderCollapse: 'collapse' }}>
                  <tbody>
                    {Object.entries(g.stats || {}).map(([k, v]) => (
                      <tr key={k} style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <td style={{ padding: '3px 0', color: '#555' }}>{k.replace(/_/g, ' ')}</td>
                        <td style={{ padding: '3px 0', fontWeight: 600, textAlign: 'right' }}>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div>
      <h2 style={{ color: '#37474f', marginBottom: 16 }}>Clinical Atlas — Key Alerts by Gene</h2>
      {genes.map(g => (
        <div key={g.gene} style={{ marginBottom: 24 }}>
          <h3 style={{ color: GENE_COLORS[g.gene] || '#37474f', marginBottom: 8, fontSize: 15 }}>
            {g.gene} — {GENE_DISEASE[g.gene] || g.alias?.slice(0, 80)}
          </h3>
          {(g.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
          <div style={{
            background: '#fafafa', borderRadius: 8, padding: '10px 14px', marginTop: 8,
            fontSize: 12, color: '#555',
          }}>
            <strong>Dx Delay Distribution:</strong>{' '}
            {Object.entries(g.dx_delay_distribution || {}).map(([k, v]) => `${k}: ${v} pts`).join(' · ')}
          </div>
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
      <h2 style={{ color: '#37474f', marginBottom: 16 }}>Clinical Definitions & Standards</h2>

      <h3 style={{ marginBottom: 12 }}>Core Concepts</h3>
      {Object.entries(concepts || {}).map(([title, text]) => (
        <div key={title} style={{
          marginBottom: 16, background: '#fafafa', borderRadius: 8, padding: '12px 16px',
          borderLeft: '4px solid #37474f',
        }}>
          <strong style={{ color: '#37474f', fontSize: 14 }}>{title}</strong>
          <p style={{ color: '#555', fontSize: 13, lineHeight: 1.6, marginTop: 6 }}>{text}</p>
        </div>
      ))}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Pharmacological Distinctions</h3>
      {(pharmacological_distinctions || []).map((d, i) => (
        <div key={i} style={{
          marginBottom: 10, background: '#fff8e1', borderRadius: 6, padding: '10px 14px',
          borderLeft: '3px solid #f57f17', fontSize: 13, color: '#555', lineHeight: 1.5,
        }}>
          {d}
        </div>
      ))}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Key Clinical Standards</h3>
      {(key_standards || []).map((s, i) => (
        <div key={i} style={{
          marginBottom: 8, background: '#e8f5e9', borderRadius: 6, padding: '10px 14px',
          borderLeft: '3px solid #2e7d32', fontSize: 13, color: '#333', lineHeight: 1.5,
        }}>
          {s}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryGenodermatosesAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-genodermatoses-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) {
      fetch(`${API}/api/hereditary-genodermatoses-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-genodermatoses-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab]);

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 16px', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#37474f', marginBottom: 4 }}>
          🧬 Hereditary Genodermatoses Atlas
        </h1>
        <p style={{ color: '#777', fontSize: 14 }}>
          Complete 8-Gene Hereditary Skin Disease Reference ·
          COL7A1 (Dystrophic EB) · LAMB3 (Junctional EB) · KRT5 (EB Simplex) · ABCA12 (Harlequin) ·
          SPINK5 (Netherton) · ATP2A2 (Darier) · STS (X-Linked Ichthyosis) · IKBKG (Incontinentia Pigmenti) ·
          320 patients, seeds 1590–1597
        </p>
      </div>

      {error && (
        <div style={{ background: '#ffebee', color: '#c62828', padding: '10px 14px', borderRadius: 6, marginBottom: 16 }}>
          Error: {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 6, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '7px 18px', borderRadius: 20, border: 'none', cursor: 'pointer',
            background: tab === t ? '#37474f' : '#eceff1',
            color: tab === t ? '#fff' : '#546e7a',
            fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
