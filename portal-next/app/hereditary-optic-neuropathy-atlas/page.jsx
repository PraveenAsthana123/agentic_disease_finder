'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  OPA1:      '#1565c0',  // deep blue — AD ADOA, most common hereditary optic neuropathy
  OPA3:      '#4a148c',  // deep purple — Costeff / AR 3-MGC / AD OA+cataract
  'MT-ND4':  '#b71c1c',  // deep red — LHON m.11778 worst prognosis
  'MT-ND1':  '#bf360c',  // deep orange-red — LHON m.3460 LHON-Plus
  'MT-ND6':  '#e65100',  // amber-orange — LHON m.14484 best prognosis
  WFS1:      '#1b5e20',  // deep green — Wolfram DIDMOAD
  TMEM126A:  '#006064',  // dark teal — AR OA North African
  ACO2:      '#37474f',  // dark slate — infantile OA + cerebellar ataxia
};

const GENE_DISEASE = {
  OPA1:      'ADOA (AD) — OPA1; haploinsufficiency → classic; GTPase missense → OPA1-plus (ptosis+CPEO+myopathy)',
  OPA3:      'OPA3 (AR/AD) — AR=Costeff: OA+chorea+spastic paraplegia+3-MGC; AD: OA+cataract±chorea',
  'MT-ND4':  'LHON m.11778G>A (Maternal) — most common LHON ~70%; WORST prognosis ~20% recovery; idebenone+Lumevoq',
  'MT-ND1':  'LHON m.3460G>A (Maternal) — second LHON ~13%; LHON-Plus MS-like CNS lesions; brain MRI',
  'MT-ND6':  'LHON m.14484T>C (Maternal) — third LHON ~14%; BEST prognosis ~70% recovery; youngest onset',
  WFS1:      'Wolfram / DIDMOAD (AR) — WFS1; DM→OA→DI→SNHL; neurogenic bladder; annual urology',
  TMEM126A:  'AROA (AR) — TMEM126A; p.Arg55* North African founder; pure slowly progressive OA',
  ACO2:      'Infantile OA + cerebellar ataxia (AR) — ACO2; TCA cycle; avoid valproate; levetiracetam',
};

const LHON_GENES    = ['MT-ND4', 'MT-ND1', 'MT-ND6'];
const FUSION_GENES  = ['OPA1', 'OPA3'];
const SYSTEMIC_GENES = ['WFS1', 'TMEM126A', 'ACO2'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Hereditary Optic Neuropathy Atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-sm-4 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 border-0 shadow-sm">
        <div className="card-body text-center p-2" style={{ borderTop: `4px solid ${color}` }}>
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function AlertBadge({ text, color = '#37474f' }) {
  return (
    <span className="badge me-1 mb-1" style={{ background: color, fontSize: '0.7rem' }}>
      {text}
    </span>
  );
}

/* ── OVERVIEW TAB ── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats;

  const statItems = [
    { key: 'optic_atrophy_any',               label: 'Optic Atrophy (any type)',                    color: '#1565c0' },
    { key: 'male_predominance_lhon',           label: 'Male Carriers — LHON Genes',                  color: '#b71c1c' },
    { key: 'lhon_tobacco_use',                 label: 'Tobacco Use at LHON Onset',                   color: '#b71c1c' },
    { key: 'wolfram_dm_at_presentation',       label: 'T1 Diabetes (Wolfram cohort)',                 color: '#1b5e20' },
    { key: 'didmoad_optic_atrophy',            label: 'Optic Atrophy (Wolfram cohort)',               color: '#1b5e20' },
    { key: 'wolfram_neurogenic_bladder',       label: 'Neurogenic Bladder (Wolfram)',                 color: '#1b5e20' },
    { key: 'opa1_plus_syndromic',              label: 'OPA1-plus Syndromic (OPA1)',                   color: '#1565c0' },
    { key: 'costeff_3mgc_elevated',            label: '3-MGC Elevated (OPA3 AR)',                     color: '#4a148c' },
    { key: 'lhon_best_prognosis_nd6',          label: 'Best Prognosis — MT-ND6 ~70% Recovery',        color: '#e65100' },
    { key: 'aco2_cerebellar_atrophy',          label: 'Cerebellar Atrophy MRI (ACO2)',                color: '#37474f' },
    { key: 'tmem126a_north_african',           label: 'North African Ancestry (TMEM126A)',            color: '#006064' },
    { key: 'lhon_plus_cns_lesions',            label: 'LHON-Plus CNS Lesions (ND1/ND6)',             color: '#bf360c' },
  ];

  return (
    <div>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} color="#37474f" />
        <KPI label="Genes" value={data.genes?.length} color="#37474f" />
        <KPI label="Mitochondrial Fusion" value="2" color="#1565c0" />
        <KPI label="LHON (mtDNA)" value="3" color="#b71c1c" />
        <KPI label="Systemic / Syndromic" value="3" color="#1b5e20" />
        <KPI label="Seeds" value={data.seed_range} color="#37474f" />
      </div>

      <div className="alert alert-danger mb-3">
        <strong>🚨 LHON — TEST mtDNA NOT NUCLEAR DNA:</strong> Standard exome/WGS MISSES mtDNA point variants. Request dedicated mtDNA analysis (m.11778, m.3460, m.14484). Idebenone 900 mg/day — start within 1 year of onset.
      </div>
      <div className="alert alert-warning mb-3">
        <strong>⚠️ LHON PROGNOSIS IS MUTATION-SPECIFIC:</strong> MT-ND4 only ~20% recover. MT-ND6 ~70% recover. NEVER apply one prognosis to all LHON — always quote the specific mutation's recovery rate.
      </div>
      <div className="alert alert-info mb-3">
        <strong>ℹ️ WFS1 DIDMOAD:</strong> T1 DM precedes optic atrophy by years. Child with DM + progressive OA = WFS1 first. Annual urological review — neurogenic bladder → hydronephrosis → CKD.
      </div>
      <div className="alert alert-secondary mb-4">
        <strong>🔬 OPA3 Costeff:</strong> Urine 3-methylglutaconic acid (3-MGC) is cheap, fast, diagnostic — send BEFORE gene panel. Iraqi-Jewish ancestry + OA + chorea = c.143-1G>A targeted first.
        <strong> | TMEM126A:</strong> North African / Moroccan — p.Arg55* targeted Sanger first.
        <strong> | ACO2:</strong> AVOID valproate — use levetiracetam for seizures.
      </div>

      <h6 className="fw-bold mb-3">Aggregate Clinical Features (320 patients, 8 genes)</h6>
      <div className="row g-2 mb-4">
        {statItems.map(({ key, label, color }) => s?.[key] != null && (
          <div key={key} className="col-6 col-md-4 col-lg-3">
            <div className="card border-0 shadow-sm">
              <div className="card-body p-2" style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fw-bold" style={{ color }}>{s[key]}%</div>
                <div className="text-muted small">{label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-2">Gene Classification</h6>
      <div className="row g-3 mb-4">
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#1565c0', color: 'white' }}>
              <strong>Mitochondrial Fusion GTPases (2 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {FUSION_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]?.trim()}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#b71c1c', color: 'white' }}>
              <strong>LHON — Primary mtDNA Mutations (3 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {LHON_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]?.trim()}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#1b5e20', color: 'white' }}>
              <strong>Systemic / Syndromic OA (3 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {SYSTEMIC_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]?.trim()}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <h6 className="fw-bold mb-2">Top Clinical Alerts</h6>
      <div className="mb-3">
        {(data.top_alerts || []).map((a, i) => (
          <AlertBadge key={i} text={a}
            color={a.includes('OPA1') ? '#1565c0' : a.includes('OPA3') || a.includes('3-MGC') ? '#4a148c' :
                   a.includes('MT-ND4') || a.includes('ND4') ? '#b71c1c' :
                   a.includes('MT-ND1') || a.includes('LHON-Plus') ? '#bf360c' :
                   a.includes('MT-ND6') || a.includes('ND6') ? '#e65100' :
                   a.includes('WFS1') || a.includes('DIDMOAD') || a.includes('DM') || a.includes('bladder') ? '#1b5e20' :
                   a.includes('TMEM126A') || a.includes('North African') ? '#006064' :
                   a.includes('ACO2') || a.includes('valproate') ? '#37474f' :
                   a.includes('LHON') || a.includes('idebenone') || a.includes('mtDNA') || a.includes('tobacco') ? '#b71c1c' :
                   '#546e7a'} />
        ))}
      </div>

      <div className="row g-3">
        {Object.entries(data.diseases || {}).map(([gene, desc]) => (
          <div key={gene} className="col-12 col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-3" style={{ borderLeft: `5px solid ${GENE_COLORS[gene] || '#546e7a'}` }}>
                <div className="fw-bold small mb-1" style={{ color: GENE_COLORS[gene] }}>{gene}</div>
                <div className="text-muted" style={{ fontSize: '0.78rem' }}>{desc}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);

  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover align-middle">
        <thead className="table-dark">
          <tr>
            <th>Gene</th><th>Protein</th><th>aa</th><th>Locus</th>
            <th>Inheritance</th><th>OMIM Gene</th><th>OMIM Disease</th>
            <th>Organ System</th><th>N Patients</th>
          </tr>
        </thead>
        <tbody>
          {genes.map(g => (
            <tr key={g.gene}>
              <td><span className="fw-bold" style={{ color: GENE_COLORS[g.gene] }}>{g.gene}</span></td>
              <td style={{ fontSize: '0.8rem' }}>{g.protein}</td>
              <td>{g.aa}</td>
              <td><code style={{ fontSize: '0.75rem' }}>{g.locus}</code></td>
              <td>
                <span className={`badge ${
                  g.inheritance?.startsWith('AR') ? 'bg-success' :
                  g.inheritance?.startsWith('Maternal') ? 'bg-danger' :
                  'bg-primary'}`}
                  style={{ fontSize: '0.65rem' }}>
                  {g.inheritance?.split(' ')[0]}
                </span>
              </td>
              <td><a href={`https://omim.org/entry/${g.omim_gene}`} target="_blank" rel="noreferrer"
                style={{ fontSize: '0.8rem' }}>{g.omim_gene}</a></td>
              <td><a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer"
                style={{ fontSize: '0.8rem' }}>{g.omim_disease}</a></td>
              <td style={{ fontSize: '0.75rem', maxWidth: 200 }}>{g.organ_system}</td>
              <td className="text-center">{g.n_patients}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(Object.keys(data)[0]);
  const g = data[selected];
  if (!g) return null;

  return (
    <div className="row g-3">
      <div className="col-md-2">
        <div className="list-group list-group-flush">
          {Object.keys(data).map(gene => (
            <button key={gene}
              className={`list-group-item list-group-item-action py-1 px-2 ${selected === gene ? 'active' : ''}`}
              style={selected === gene ? { background: GENE_COLORS[gene], borderColor: GENE_COLORS[gene] } : {}}
              onClick={() => setSelected(gene)}>
              <span className="fw-bold small">{gene}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="col-md-10">
        <div className="card border-0 shadow-sm">
          <div className="card-header" style={{ background: GENE_COLORS[selected], color: 'white' }}>
            <strong>{g.gene}</strong> — {g.protein} | {g.aa} | {g.locus} | {g.inheritance?.split(' ')[0]}
          </div>
          <div className="card-body">
            <div className="row g-3 mb-3">
              <div className="col-md-6">
                <h6 className="fw-bold">Clinical Hallmarks</h6>
                <ul className="small mb-0">
                  {(g.hallmarks || []).map((h, i) => <li key={i} className="mb-1">{h}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold">Treatment Alerts</h6>
                <ul className="small mb-0">
                  {(g.treatment_alerts || []).map((t, i) => <li key={i} className="mb-1">{t}</li>)}
                </ul>
              </div>
            </div>

            <div className="mb-3">
              <h6 className="fw-bold">Feature Frequencies ({g.n_patients} patients)</h6>
              <div className="row g-1">
                {Object.entries(g.stats || {}).map(([k, v]) => (
                  <div key={k} className="col-6 col-md-4">
                    <div className="d-flex align-items-center gap-2 small">
                      <div style={{ width: 40, height: 8, borderRadius: 4, background: '#e0e0e0', position: 'relative', flexShrink: 0 }}>
                        <div style={{ width: `${v}%`, height: '100%', borderRadius: 4, background: GENE_COLORS[selected] }} />
                      </div>
                      <span className="text-muted" style={{ fontSize: '0.7rem' }}>{k.replace(/_/g, ' ')} <strong>{v}%</strong></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mb-3">
              <h6 className="fw-bold">Etiology Distribution</h6>
              <ul className="small mb-0">
                {(g.etiology_distribution || []).map((e, i) => (
                  <li key={i}><strong>{Math.round(e.fraction * 100)}%</strong> — {e.etiology}</li>
                ))}
              </ul>
            </div>

            <div>
              <h6 className="fw-bold">Primary Treatment</h6>
              <p className="small mb-0 text-muted">{g.primary_treatment}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;

  return (
    <div>
      <h6 className="fw-bold mb-3">Disease Classification</h6>
      {Object.entries(data.classification || {}).map(([cat, genes]) => (
        <div key={cat} className="mb-3">
          <h6 className="text-muted small fw-bold border-bottom pb-1">{cat.replace(/_/g, ' ')}</h6>
          <ul className="small">
            {Object.entries(genes).map(([k, v]) => (
              <li key={k}><strong>{k.replace(/_/g, ' ')}</strong>: {v}</li>
            ))}
          </ul>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Key Diagnostic Rules</h6>
      {Object.entries(data.key_diagnostic_rules || {}).map(([rule, text]) => (
        <div key={rule} className="mb-3 p-3 rounded" style={{ background: '#f8f9fa', borderLeft: '4px solid #37474f' }}>
          <div className="fw-bold small mb-1" style={{ color: '#37474f' }}>{rule.replace(/_/g, ' ')}</div>
          <div className="small text-muted">{text}</div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Treatment Hierarchies</h6>
      {Object.entries(data.treatment_hierarchy || {}).map(([gene, steps]) => (
        <div key={gene} className="mb-3">
          <h6 className="small fw-bold" style={{ color: '#37474f' }}>{gene.replace(/_/g, ' ')}</h6>
          <ol className="small mb-0">
            {steps.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
          </ol>
        </div>
      ))}
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function HereditaryOpticNeuropathyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-optic-neuropathy-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-optic-neuropathy-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-optic-neuropathy-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, def]) => { setOverview(ov); setBreakdown(bd); setDefinitions(def); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h4 className="fw-bold mb-1">🧬 Hereditary Optic Neuropathy Atlas</h4>
        <p className="text-muted small mb-0">
          Complete 8-Gene Hereditary Optic Neuropathy Reference —
          OPA1 (ADOA/OPA1-plus) · OPA3 (Costeff/3-MGC) ·
          MT-ND4 (LHON m.11778) · MT-ND1 (LHON m.3460) · MT-ND6 (LHON m.14484) ·
          WFS1 (Wolfram/DIDMOAD) · TMEM126A (AROA) · ACO2 (infantile OA+ataxia) |
          320 patients · 8×40 · seeds 1414–1421
        </p>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'       && <OverviewTab data={overview} />}
      {tab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'    && <DefinitionsTab data={definitions} />}
    </div>
  );
}
