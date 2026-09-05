'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  RPGR:   '#1a237e',  // deep navy — X-linked, most common XLRP
  USH2A:  '#880e4f',  // deep rose — syndromic HL+RP
  ABCA4:  '#e65100',  // deep orange — macular, Stargardt
  RDH12:  '#b71c1c',  // deep red — severe LCA infantile
  PRPF31: '#1b5e20',  // deep green — AD spliceopathy
  EYS:    '#4a148c',  // deep violet — AR, largest retinal gene
  CNGB3:  '#006064',  // deep teal — achromatopsia, gene therapy
  RS1:    '#37474f',  // dark slate — X-linked retinoschisis
};

const GENE_DISEASE = {
  RPGR:   'RP3 (XLRP)',
  USH2A:  'Usher IIA',
  ABCA4:  'Stargardt/STGD1',
  RDH12:  'LCA13/EOSRD',
  PRPF31: 'RP11 (AD)',
  EYS:    'RP25 (AR)',
  CNGB3:  'ACHM3',
  RS1:    'XLRS',
};

function Loading() {
  return (
    <div className="text-center py-5">
      <div className="spinner-border text-primary" role="status" />
      <p className="mt-3 text-muted">Loading Retinal Dystrophy atlas…</p>
    </div>
  );
}

function ErrorMsg({ msg }) {
  return <div className="alert alert-danger m-4"><strong>Error:</strong> {msg}</div>;
}

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm border-0">
        <div className="card-body text-center p-3">
          <div className="fw-bold fs-3" style={{ color: color || '#1a237e' }}>{value}</div>
          <div className="small text-muted">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color }) {
  const p = Math.min(100, Math.max(0, Math.round(pct || 0)));
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{p}%</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${p}%`, backgroundColor: color || '#1a237e' }} />
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  const agg = ov.aggregate_clinical || {};
  const alerts = ov.drug_alerts || [];
  const pearls = ov.clinical_pearls || [];

  return (
    <div>
      {/* Hero */}
      <div className="rounded-3 p-4 mb-4 text-white" style={{ background: 'linear-gradient(135deg,#1a237e,#006064)' }}>
        <h2 className="fw-bold">{ov.atlas_name}</h2>
        <p className="mb-1 opacity-90">{ov.atlas_subtitle}</p>
        <div className="d-flex gap-3 flex-wrap mt-2">
          <span className="badge bg-light text-dark">{ov.n_genes} Genes</span>
          <span className="badge bg-light text-dark">{ov.n_patients} Patients</span>
          <span className="badge bg-light text-dark">Seeds {ov.seeds}</span>
          <span className="badge bg-light text-dark">8 Genes: {(ov.genes || []).join(' · ')}</span>
        </div>
      </div>

      {/* KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Night Blindness %" value={`${agg.night_blindness_pct || 0}%`} color="#1a237e" />
        <KPI label="Photophobia %" value={`${agg.photophobia_pct || 0}%`} color="#880e4f" />
        <KPI label="Nystagmus %" value={`${agg.nystagmus_pct || 0}%`} color="#006064" />
        <KPI label="Hearing Loss %" value={`${agg.hearing_loss_pct || 0}%`} color="#e65100" />
        <KPI label="GT Eligible %" value={`${agg.gene_therapy_eligible_pct || 0}%`} color="#1b5e20" />
        <KPI label="Severe %" value={`${agg.severity_severe_pct || 0}%`} color="#b71c1c" />
      </div>

      {/* Description */}
      <div className="card border-0 shadow-sm mb-4">
        <div className="card-body">
          <h5 className="fw-bold mb-2">Atlas Description</h5>
          <p className="text-muted mb-0" style={{ lineHeight: 1.7 }}>{ov.description}</p>
        </div>
      </div>

      {/* Aggregate bars */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Aggregate Visual &amp; Retinal Features</h6>
              <BarRow label="Night blindness (rod dysfunction)" pct={agg.night_blindness_pct} color="#1a237e" />
              <BarRow label="Photophobia (cone / broad dysfunction)" pct={agg.photophobia_pct} color="#006064" />
              <BarRow label="Nystagmus (infantile onset)" pct={agg.nystagmus_pct} color="#4a148c" />
              <BarRow label="Retinoschisis (RS1-XLRS)" pct={agg.schisis_pct} color="#37474f" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card border-0 shadow-sm h-100">
            <div className="card-body">
              <h6 className="fw-bold mb-3">Syndromic &amp; Therapeutic Features</h6>
              <BarRow label="Hearing loss (USH2A syndromic)" pct={agg.hearing_loss_pct} color="#880e4f" />
              <BarRow label="Colour blindness / achromatopsia" pct={agg.color_blind_pct} color="#e65100" />
              <BarRow label="Macular primary lesion" pct={agg.macular_primary_pct} color="#b71c1c" />
              <BarRow label="Gene therapy eligible" pct={agg.gene_therapy_eligible_pct} color="#1b5e20" />
            </div>
          </div>
        </div>
      </div>

      {/* Drug Alerts */}
      {alerts.length > 0 && (
        <div className="mb-4">
          <h5 className="fw-bold mb-3">Critical Drug &amp; Management Alerts</h5>
          {alerts.map((a, i) => (
            <div key={i} className="alert alert-warning border-warning shadow-sm mb-3">
              <div className="fw-bold mb-1">&#9888; {a.title}</div>
              <div className="small">{a.body}</div>
            </div>
          ))}
        </div>
      )}

      {/* Clinical Pearls */}
      {pearls.length > 0 && (
        <div className="card border-0 shadow-sm mb-4">
          <div className="card-body">
            <h5 className="fw-bold mb-3">Clinical Pearls — Retinal Dystrophy Hierarchy</h5>
            <ul className="mb-0 small" style={{ lineHeight: 2 }}>
              {pearls.map((p, i) => <li key={i}>{p}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div>
      <h5 className="fw-bold mb-3">Per-Gene Summary Table</h5>
      <div className="table-responsive">
        <table className="table table-bordered table-hover align-middle small">
          <thead className="table-dark">
            <tr>
              <th>Gene</th>
              <th>Disease</th>
              <th>Locus</th>
              <th>aa / kDa</th>
              <th>Inheritance</th>
              <th>Night Blind %</th>
              <th>Photophobia %</th>
              <th>Nystagmus %</th>
              <th>HL %</th>
              <th>GT Eligible %</th>
            </tr>
          </thead>
          <tbody>
            {genes.map(g => {
              const s = g.stats || {};
              return (
                <tr key={g.gene}>
                  <td>
                    <span className="badge" style={{ backgroundColor: GENE_COLORS[g.gene] || '#555' }}>
                      {g.gene}
                    </span>
                  </td>
                  <td>{GENE_DISEASE[g.gene] || g.gene}</td>
                  <td className="text-nowrap">{g.locus}</td>
                  <td className="text-nowrap">{g.aa} / {g.kDa}</td>
                  <td><span className="badge bg-secondary text-wrap">{g.inheritance}</span></td>
                  <td className="fw-bold" style={{ color: '#1a237e' }}>{s.night_blindness_pct}%</td>
                  <td>{s.photophobia_pct}%</td>
                  <td>{s.nystagmus_pct}%</td>
                  <td className="fw-bold" style={{ color: '#880e4f' }}>{s.hearing_loss_pct}%</td>
                  <td className="fw-bold" style={{ color: '#1b5e20' }}>{s.gene_therapy_eligible_pct}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [selected, setSelected] = useState(null);
  const genes = Object.values(data);

  return (
    <div>
      <h5 className="fw-bold mb-3">Clinical Atlas — Select a Gene</h5>
      <div className="row g-2 mb-4">
        {genes.map(g => (
          <div key={g.gene} className="col-6 col-md-3">
            <button
              className={`btn w-100 fw-bold ${selected?.gene === g.gene ? 'text-white' : 'btn-outline-secondary'}`}
              style={selected?.gene === g.gene ? { backgroundColor: GENE_COLORS[g.gene] } : {}}
              onClick={() => setSelected(g)}
            >
              {g.gene}
              <div className="small fw-normal">{GENE_DISEASE[g.gene]}</div>
            </button>
          </div>
        ))}
      </div>

      {selected && (
        <div className="card border-0 shadow">
          <div className="card-header text-white fw-bold" style={{ backgroundColor: GENE_COLORS[selected.gene] || '#1a237e' }}>
            {selected.gene} — {selected.protein}
          </div>
          <div className="card-body">
            <div className="row g-4">
              <div className="col-md-6">
                <h6 className="fw-bold text-muted">Gene / Protein</h6>
                <p className="small">{selected.alias}</p>

                <h6 className="fw-bold text-muted mt-3">Molecular Mechanism</h6>
                <p className="small">{selected.gene_class}</p>

                <h6 className="fw-bold text-muted mt-3">Phenotype</h6>
                <p className="small">{selected.phenotype}</p>
              </div>
              <div className="col-md-6">
                <h6 className="fw-bold text-danger">Hallmark / Red Flag</h6>
                <p className="small">{selected.hallmark}</p>

                <h6 className="fw-bold text-primary mt-3">Treatment Alert</h6>
                <p className="small">{selected.treatment_alert}</p>

                <h6 className="fw-bold text-muted mt-3">Differential Diagnosis</h6>
                <p className="small">{selected.key_ddx}</p>

                <div className="row g-2 mt-2">
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>Vision pattern:</strong> {selected.vision_pattern}
                    </div>
                  </div>
                  <div className="col-12">
                    <div className="bg-light rounded p-2 small">
                      <strong>ERG pattern:</strong> {selected.erg_pattern}
                    </div>
                  </div>
                </div>

                {/* Mini stats */}
                {selected.stats && (
                  <div className="mt-3">
                    <h6 className="fw-bold text-muted">Cohort Stats ({selected.cohort_n} patients)</h6>
                    <div className="row g-2 text-center">
                      {[
                        ['Night Blind', `${selected.stats.night_blindness_pct}%`, '#1a237e'],
                        ['Photophobia', `${selected.stats.photophobia_pct}%`, '#006064'],
                        ['Nystagmus', `${selected.stats.nystagmus_pct}%`, '#4a148c'],
                        ['HL', `${selected.stats.hearing_loss_pct}%`, '#880e4f'],
                        ['Schisis', `${selected.stats.schisis_pct}%`, '#37474f'],
                        ['GT Eligible', `${selected.stats.gene_therapy_eligible_pct}%`, '#1b5e20'],
                      ].map(([l, v, c]) => (
                        <div key={l} className="col-4">
                          <div className="border rounded p-1">
                            <div className="fw-bold small" style={{ color: c }}>{v}</div>
                            <div className="text-muted" style={{ fontSize: 10 }}>{l}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {!selected && (
        <div className="text-center text-muted py-5">
          <div style={{ fontSize: 48 }}>&#x1f441;&#xfe0f;</div>
          <p>Select a gene above to view its full clinical profile</p>
        </div>
      )}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const terms = data.terms || [];
  return (
    <div>
      <h5 className="fw-bold mb-3">Clinical Definitions — Retinal Dystrophies</h5>
      <div className="accordion" id="defAccordion">
        {terms.map((t, i) => (
          <div key={i} className="accordion-item border-0 shadow-sm mb-2">
            <h2 className="accordion-header">
              <button
                className="accordion-button collapsed fw-bold"
                type="button"
                data-bs-toggle="collapse"
                data-bs-target={`#def${i}`}
              >
                {t.term}
              </button>
            </h2>
            <div id={`def${i}`} className="accordion-collapse collapse" data-bs-parent="#defAccordion">
              <div className="accordion-body small text-muted" style={{ lineHeight: 1.7 }}>
                {t.definition}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function RetinalDystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/retinal-dystrophy-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/retinal-dystrophy-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/retinal-dystrophy-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <ErrorMsg msg={error} />;

  return (
    <div className="container-fluid py-4 px-3 px-md-4">
      <h1 className="fw-bold mb-1" style={{ color: '#1a237e' }}>
        &#x1f441;&#xfe0f; Retinal Dystrophy Atlas
      </h1>
      <p className="text-muted mb-3">
        Complete 8-Gene Hereditary Retinal Dystrophy Reference —
        RPGR · USH2A · ABCA4 · RDH12 · PRPF31 · EYS · CNGB3 · RS1
        (320 patients, seeds 1246–1253)
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === t ? ' active fw-bold' : ''}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
