'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MYO7A:  '#1565c0',  // deep blue — USH1B most common USH1
  USH2A:  '#0288d1',  // mid blue — USH2A most common overall
  CDH23:  '#006064',  // dark teal — USH1D tip-link upper end
  PCDH15: '#2e7d32',  // deep green — USH1F tip-link lower end
  ADGRV1: '#e65100',  // amber — USH2C largest human protein
  CLRN1:  '#6a1b9a',  // deep purple — USH3A progressive SNHL
  WHRN:   '#c62828',  // deep red — USH2D rarest USH2
  SANS:   '#37474f',  // slate — USH1G rarest USH1
};

const GENE_DISEASE = {
  MYO7A:  'USH1B (AR) — most common USH1 ~40-55%; congenital profound SNHL; vestibular areflexia; CI excellent; Acadian founder',
  USH2A:  'USH2A (AR) — most common Usher overall ~40%; moderate-severe HF SNHL; NORMAL vestibular; c.2299delG European founder 30%',
  CDH23:  'USH1D (AR) — ~20% USH1; TIP LINK UPPER END; DFNB12 milder alleles; Ca2+ dependent',
  PCDH15: 'USH1F (AR) — ~15-20% USH1; TIP LINK LOWER END; gates TMC channels; Roma founder p.Arg929Stop',
  ADGRV1: 'USH2C (AR) — 2nd most common USH2; LARGEST HUMAN PROTEIN 6307aa/692kDa; ankle-link complex',
  CLRN1:  'USH3A (AR) — PROGRESSIVE SNHL (pathognomonic DDx); Finnish p.Asn48Lys; Ashkenazi p.Tyr176Ser',
  WHRN:   'USH2D (AR) — rarest USH2 (<5%); PDZ scaffold; ankle-link + stereocilia tip; DFNB31',
  SANS:   'USH1G (AR) — rarest USH1 (<5%); tip-link ASSEMBLY SCAFFOLD; MYO7A→SANS→HARMONIN→CDH23→PCDH15→TMC',
};

const USH1_GENES   = ['MYO7A', 'CDH23', 'PCDH15', 'SANS'];
const USH2_GENES   = ['USH2A', 'ADGRV1', 'WHRN'];
const USH3_GENES   = ['CLRN1'];

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Hereditary Usher Syndrome Atlas…</p>
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
  const m = data.aggregate_metrics || {};

  const statItems = [
    { key: 'cochlear_implant_pct',    label: 'Cochlear Implant Fitted',         color: '#1565c0' },
    { key: 'ci_pct',                  label: 'Cochlear Implant Fitted',         color: '#1565c0' },
    { key: 'vestibular_absent_pct',   label: 'Vestibular Areflexia (USH1)',     color: '#e65100' },
    { key: 'vestibular_normal_pct',   label: 'Normal Vestibular (USH2)',        color: '#2e7d32' },
    { key: 'profound_snhl_pct',       label: 'Profound SNHL (≥90 dBHL)',        color: '#1565c0' },
    { key: 'walking_delay_pct',       label: 'Delayed Walking (USH1)',          color: '#c62828' },
    { key: 'low_va_pct',              label: 'VA < 0.3 (advanced RP)',          color: '#6a1b9a' },
    { key: 'progressive_snhl_pct',    label: 'Progressive SNHL (USH3)',         color: '#6a1b9a' },
  ];

  return (
    <div>
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} color="#37474f" />
        <KPI label="Genes" value="8" color="#37474f" />
        <KPI label="USH Type 1 (profound)" value="4 genes" color="#1565c0" />
        <KPI label="USH Type 2 (moderate HF)" value="3 genes" color="#0288d1" />
        <KPI label="USH Type 3 (progressive)" value="1 gene" color="#6a1b9a" />
        <KPI label="Seeds" value={data.seeds} color="#37474f" />
      </div>

      <div className="alert alert-success mb-3">
        <strong>🦻 ALL USHER — COCHLEAR IMPLANT CANDIDATES:</strong> All Usher syndrome types (USH1, USH2, USH3) are excellent cochlear implant candidates. USH1: implant BEFORE AGE 2 for optimal speech-language development. USH2: when hearing aids insufficient. USH3: consider early CI — concurrent progressive RP will eventually limit lip-reading.
      </div>
      <div className="alert alert-primary mb-3">
        <strong>🧬 USH2A c.2299delG — TEST FIRST:</strong> European-descent patients with USH2 phenotype: test USH2A c.2299delG (p.Glu767SerfsX21) FIRST. ~30% of all USH2A alleles in European population. Simple targeted Sanger test. Carrier frequency ~1/72 in Europeans.
      </div>
      <div className="alert alert-warning mb-3">
        <strong>⚠️ CLRN1 — PROGRESSIVE SNHL (pathognomonic):</strong> USH3A/CLRN1 is the ONLY Usher type with PROGRESSIVE SNHL. USH1 and USH2 both have congenital/stable SNHL. Progressive SNHL + RP in a young person → always test CLRN1. Finnish founder p.Asn48Lys; Ashkenazi founder p.Tyr176Ser.
      </div>
      <div className="alert alert-danger mb-4">
        <strong>🔬 ADGRV1 — LARGEST HUMAN PROTEIN (6307 aa):</strong> ADGRV1/VLGR1/GPR98 encodes the LARGEST known human protein at 6307 aa / ~692 kDa. Clinical significance: gene therapy delivery challenging (exceeds standard AAV capacity ~4.7 kb). Dual-vector split-intein AAV strategies under development.
      </div>

      <h6 className="fw-bold mb-3">Aggregate Clinical Features (320 patients, 8 genes)</h6>
      <div className="row g-2 mb-4">
        {statItems.map(({ key, label, color }) => m?.[key] != null && (
          <div key={key} className="col-6 col-md-4 col-lg-3">
            <div className="card border-0 shadow-sm">
              <div className="card-body p-2" style={{ borderLeft: `4px solid ${color}` }}>
                <div className="fw-bold" style={{ color }}>{m[key]}%</div>
                <div className="text-muted small">{label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-3">Tip-Link Anatomy (USH1 Structural Pearl)</h6>
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body p-3">
          <div className="d-flex align-items-center gap-3 flex-wrap">
            <div className="text-center">
              <div className="badge bg-primary p-2 mb-1" style={{ fontSize: '0.8rem' }}>MYO7A (motor)</div>
              <div className="text-muted small">transports SANS to tip</div>
            </div>
            <div className="text-muted fw-bold">→</div>
            <div className="text-center">
              <div className="badge bg-secondary p-2 mb-1" style={{ fontSize: '0.8rem' }}>SANS (scaffold)</div>
              <div className="text-muted small">assembles complex at tip</div>
            </div>
            <div className="text-muted fw-bold">→</div>
            <div className="text-center">
              <div className="badge p-2 mb-1" style={{ background: '#006064', fontSize: '0.8rem' }}>CDH23 (upper end)</div>
              <div className="text-muted small">tip-link upper anchor</div>
            </div>
            <div className="text-muted fw-bold">↕</div>
            <div className="text-center">
              <div className="badge p-2 mb-1" style={{ background: '#2e7d32', fontSize: '0.8rem' }}>PCDH15 (lower end)</div>
              <div className="text-muted small">tip-link lower anchor</div>
            </div>
            <div className="text-muted fw-bold">→</div>
            <div className="text-center">
              <div className="badge bg-dark p-2 mb-1" style={{ fontSize: '0.8rem' }}>TMC1/TMC2</div>
              <div className="text-muted small">mechanotransduction</div>
            </div>
          </div>
          <div className="text-muted small mt-2">
            <strong>Key pearl:</strong> CDH23 = upper end (inserts into cuticular plate) · PCDH15 = lower end (gates TMC channels) · SANS = tip-link assembly scaffold (transported by MYO7A)
          </div>
        </div>
      </div>

      <h6 className="fw-bold mb-2">Usher Type Classification</h6>
      <div className="row g-3 mb-4">
        <div className="col-md-4">
          <div className="card border-0 shadow-sm">
            <div className="card-header" style={{ background: '#1565c0', color: 'white' }}>
              <strong>USH Type 1 — Profound (4 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {USH1_GENES.map(g => (
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
            <div className="card-header" style={{ background: '#0288d1', color: 'white' }}>
              <strong>USH Type 2 — Moderate HF (3 genes)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {USH2_GENES.map(g => (
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
            <div className="card-header" style={{ background: '#6a1b9a', color: 'white' }}>
              <strong>USH Type 3 — Progressive (1 gene)</strong>
            </div>
            <ul className="list-group list-group-flush small">
              {USH3_GENES.map(g => (
                <li key={g} className="list-group-item py-1">
                  <span className="fw-bold" style={{ color: GENE_COLORS[g] }}>{g}</span>{' — '}
                  <span className="text-muted">{GENE_DISEASE[g].split('—')[1]?.split(';')[0]?.trim()}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <h6 className="fw-bold mb-2">Clinical Alerts</h6>
      <div className="mb-3">
        {(data.key_clinical_alerts || []).map((a, i) => (
          <AlertBadge key={i} text={a}
            color={
              a.includes('MYO7A') ? '#1565c0' :
              a.includes('USH2A') || a.includes('c.2299') ? '#0288d1' :
              a.includes('CLRN1') || a.includes('PROGRESSIVE') ? '#6a1b9a' :
              a.includes('ADGRV1') || a.includes('LARGEST') ? '#e65100' :
              a.includes('CDH23') || a.includes('UPPER') || a.includes('LOWER') ? '#006064' :
              a.includes('SANS') ? '#37474f' :
              a.includes('CI') || a.includes('COCHLEAR') ? '#1565c0' :
              '#546e7a'
            } />
        ))}
      </div>

      <h6 className="fw-bold mb-2">Atlas Summary</h6>
      <p className="text-muted small">{data.atlas_summary}</p>

      <div className="row g-3">
        {Object.entries(data.gene_summary || {}).map(([gene, info]) => (
          <div key={gene} className="col-12 col-md-6">
            <div className="card border-0 shadow-sm h-100">
              <div className="card-body p-3" style={{ borderLeft: `5px solid ${GENE_COLORS[gene] || '#546e7a'}` }}>
                <div className="fw-bold small mb-1" style={{ color: GENE_COLORS[gene] }}>{gene}</div>
                <div className="text-muted" style={{ fontSize: '0.78rem' }}>
                  {info.disease_category || GENE_DISEASE[gene]}
                </div>
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
  const genes = data.gene_breakdowns || [];

  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover align-middle">
        <thead className="table-dark">
          <tr>
            <th>Gene</th><th>Usher Type</th><th>Disease</th><th>Locus</th>
            <th>SNHL Type</th><th>Vestibular</th><th>RP Onset</th><th>N Patients</th>
          </tr>
        </thead>
        <tbody>
          {genes.map(g => {
            const ushType = USH1_GENES.includes(g.gene) ? 'USH1' : USH3_GENES.includes(g.gene) ? 'USH3' : 'USH2';
            return (
              <tr key={g.gene}>
                <td><span className="fw-bold" style={{ color: GENE_COLORS[g.gene] }}>{g.gene}</span></td>
                <td>
                  <span className={`badge ${ushType === 'USH1' ? 'bg-primary' : ushType === 'USH3' ? 'bg-warning text-dark' : 'bg-info text-dark'}`}
                    style={{ fontSize: '0.65rem' }}>{ushType}</span>
                </td>
                <td style={{ fontSize: '0.8rem', maxWidth: 180 }}>{g.disease_category?.split('—')[0]?.trim()}</td>
                <td><code style={{ fontSize: '0.75rem' }}>{g.locus}</code></td>
                <td style={{ fontSize: '0.75rem' }}>
                  {USH1_GENES.includes(g.gene) ? 'Profound flat' :
                   USH3_GENES.includes(g.gene) ? 'Progressive bilateral' :
                   'Mod-severe HF sloping'}
                </td>
                <td style={{ fontSize: '0.75rem' }}>
                  <span className={`badge ${USH1_GENES.includes(g.gene) ? 'bg-danger' : USH2_GENES.includes(g.gene) ? 'bg-success' : 'bg-secondary'}`}
                    style={{ fontSize: '0.6rem' }}>
                    {USH1_GENES.includes(g.gene) ? 'Absent' : USH2_GENES.includes(g.gene) ? 'Normal' : 'Variable'}
                  </span>
                </td>
                <td style={{ fontSize: '0.75rem' }}>
                  {g.avg_rp_onset_years ? `~${g.avg_rp_onset_years}y` : '—'}
                </td>
                <td className="text-center">{g.n_patients}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.gene_breakdowns || [];
  const [selectedGene, setSelectedGene] = useState(genes[0]?.gene || '');
  const g = genes.find(x => x.gene === selectedGene);
  if (!g) return null;

  const ushType = USH1_GENES.includes(g.gene) ? 'USH1' : USH3_GENES.includes(g.gene) ? 'USH3' : 'USH2';

  const statItems = [
    { key: 'ci_pct',                  label: 'Cochlear Implant' },
    { key: 'ha_pct',                  label: 'Hearing Aid (no CI)' },
    { key: 'profound_snhl_pct',       label: 'Profound SNHL (≥90 dBHL)' },
    { key: 'vestibular_absent_pct',   label: 'Vestibular Absent' },
    { key: 'vestibular_normal_pct',   label: 'Vestibular Normal' },
    { key: 'walking_delay_pct',       label: 'Delayed Walking' },
    { key: 'low_va_pct',              label: 'VA < 0.3 (advanced RP)' },
    { key: 'progressive_snhl_pct',    label: 'Progressive SNHL' },
  ];

  return (
    <div className="row g-3">
      <div className="col-md-2">
        <div className="list-group list-group-flush">
          {genes.map(gene => (
            <button key={gene.gene}
              className={`list-group-item list-group-item-action py-1 px-2 ${selectedGene === gene.gene ? 'active' : ''}`}
              style={selectedGene === gene.gene ? { background: GENE_COLORS[gene.gene], borderColor: GENE_COLORS[gene.gene] } : {}}
              onClick={() => setSelectedGene(gene.gene)}>
              <span className="fw-bold small">{gene.gene}</span>
              <div style={{ fontSize: '0.6rem', opacity: 0.8 }}>
                {USH1_GENES.includes(gene.gene) ? 'USH1' : USH3_GENES.includes(gene.gene) ? 'USH3' : 'USH2'}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="col-md-10">
        <div className="card border-0 shadow-sm">
          <div className="card-header" style={{ background: GENE_COLORS[g.gene], color: 'white' }}>
            <strong>{g.gene}</strong> — {ushType} | {g.locus} | {g.protein_size}
          </div>
          <div className="card-body">
            <div className="row g-3 mb-3">
              <div className="col-md-6">
                <h6 className="fw-bold">Pathognomonic / Key Features</h6>
                <ul className="small mb-0">
                  <li>{g.pathognomonic?.slice(0, 300)}…</li>
                  {(g.key_features || []).slice(0, 5).map((h, i) => <li key={i} className="mb-1">{h}</li>)}
                </ul>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold">Treatment / Management</h6>
                <p className="small mb-2">{g.treatment?.slice(0, 400)}…</p>
                <h6 className="fw-bold">Key DDx</h6>
                <p className="small mb-0 text-muted">{g.key_ddx?.slice(0, 300)}…</p>
              </div>
            </div>

            <div className="mb-3">
              <h6 className="fw-bold">Feature Frequencies ({g.n_patients} patients)</h6>
              <div className="row g-1">
                {statItems.map(({ key, label }) => g[key] != null && (
                  <div key={key} className="col-6 col-md-4">
                    <div className="d-flex align-items-center gap-2 small">
                      <div style={{ width: 40, height: 8, borderRadius: 4, background: '#e0e0e0', position: 'relative', flexShrink: 0 }}>
                        <div style={{ width: `${Math.min(g[key], 100)}%`, height: '100%', borderRadius: 4, background: GENE_COLORS[g.gene] }} />
                      </div>
                      <span className="text-muted" style={{ fontSize: '0.7rem' }}>{label} <strong>{g[key]}%</strong></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {g.systemic_involvement && typeof g.systemic_involvement === 'string' && (
              <div className="mb-3">
                <h6 className="fw-bold">Systemic / Clinical Notes</h6>
                <p className="small mb-0 text-muted">{g.systemic_involvement}</p>
              </div>
            )}

            <div className="mb-3">
              <h6 className="fw-bold">Audiogram / ERG Pattern</h6>
              <p className="small mb-0 text-muted">{g.morphology}</p>
            </div>

            {g.sample_patients?.length > 0 && (
              <div>
                <h6 className="fw-bold">Sample Patients</h6>
                <div className="table-responsive">
                  <table className="table table-sm table-striped" style={{ fontSize: '0.75rem' }}>
                    <thead>
                      <tr>
                        <th>Age</th><th>SNHL (dBHL)</th><th>SNHL Type</th>
                        <th>Vestibular</th><th>Walking Delay</th><th>CI</th><th>VA Residual</th>
                      </tr>
                    </thead>
                    <tbody>
                      {g.sample_patients.slice(0, 5).map((p, i) => (
                        <tr key={i}>
                          <td>{p.age}y</td>
                          <td>{p.snhl_db}</td>
                          <td>{p.snhl_type}</td>
                          <td>{p.vestibular}</td>
                          <td>{p.walking_delay ? '⚠️ Yes' : '—'}</td>
                          <td>{p.ci_fitted ? '✓' : '—'}</td>
                          <td>{p.va_residual?.toFixed(2) ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
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
      <h6 className="fw-bold mb-3">Gene Entries</h6>
      {Object.entries(data.gene_entries || {}).map(([gene, entry]) => (
        <div key={gene} className="mb-3 p-3 rounded" style={{ background: '#f8f9fa', borderLeft: `4px solid ${GENE_COLORS[gene] || '#37474f'}` }}>
          <div className="fw-bold small mb-1" style={{ color: GENE_COLORS[gene] || '#37474f' }}>{gene}</div>
          <div className="small text-muted">{typeof entry === 'string' ? entry : JSON.stringify(entry)}</div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Clinical Glossary</h6>
      {Object.entries(data.ush_glossary || {}).map(([term, def]) => (
        <div key={term} className="mb-3 p-3 rounded" style={{ background: '#f8f9fa', borderLeft: '4px solid #546e7a' }}>
          <div className="fw-bold small mb-1" style={{ color: '#546e7a' }}>{term.replace(/_/g, ' ')}</div>
          <div className="small text-muted">{typeof def === 'string' ? def : JSON.stringify(def)}</div>
        </div>
      ))}
    </div>
  );
}

/* ── MAIN PAGE ── */
export default function HereditaryUsherSyndromeAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-usher-syndrome-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-usher-syndrome-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-usher-syndrome-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, def]) => { setOverview(ov); setBreakdown(bd); setDefinitions(def); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h4 className="fw-bold mb-1">🧬 Hereditary Usher Syndrome Atlas</h4>
        <p className="text-muted small mb-0">
          Complete 8-Gene Hereditary Usher Syndrome Reference —
          MYO7A (USH1B, most common USH1) · USH2A (most common overall, c.2299delG) ·
          CDH23 (USH1D, tip-link upper end) · PCDH15 (USH1F, tip-link lower end) ·
          ADGRV1 (USH2C, LARGEST human protein 6307aa) · CLRN1 (USH3A, progressive SNHL) ·
          WHRN (USH2D, rarest USH2) · SANS (USH1G, rarest USH1) |
          320 patients · 8×40 · seeds 2430–2437
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
