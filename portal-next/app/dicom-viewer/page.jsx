'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = { Preliminary: 'warning', Final: 'success', Amended: 'info', Addendum: 'secondary' };
const CLS_COLOR = { Lesional: 'danger', 'Non-Lesional': 'info', Equivocal: 'warning', Normal: 'success', Unknown: 'secondary' };
const FIELD_BADGE = { '3.0T': 'primary', '1.5T': 'secondary', '7.0T': 'dark' };

export default function DICOMViewerPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedStudy, setExpandedStudy] = useState(null);
  const [tagStudy, setTagStudy] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/dicom-viewer/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/dicom-viewer/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/dicom-viewer/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'studies', label: '🗂 Study Browser' },
    { id: 'tags', label: '🏷 DICOM Tags' },
    { id: 'protocols', label: '⚙ Protocols' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const protDist = overview.protocol_distribution || {};
  const fieldDist = overview.field_strength_distribution || {};
  const mfrDist = overview.manufacturer_distribution || {};
  const readDist = overview.reading_status_distribution || {};
  const clsDist = overview.classification_distribution || {};
  const lesionDist = overview.lesion_type_distribution || {};

  const studies = breakdown?.studies || [];

  return (
    <div>
      <h3>DICOM Study Browser</h3>
      <p className="text-muted small">
        NEMA PS3 / ISO 12052 — MR Image Storage (1.2.840.10008.5.1.4.1.1.4) &nbsp;·&nbsp;
        Real data from <code>clinical.db mri_findings</code>
      </p>

      {/* KPI strip */}
      <div className="row mb-3">
        {[
          { label: 'Studies', value: overview.total_studies, color: 'primary', icon: '🗂' },
          { label: 'Series', value: overview.total_series, color: 'info', icon: '📷' },
          { label: 'Instances', value: overview.total_instances?.toLocaleString(), color: 'secondary', icon: '🖼' },
          { label: 'Avg Series/Study', value: overview.avg_series_per_study, color: 'warning', icon: '📊' },
          { label: 'Final Reads', value: `${overview.final_read_pct}%`, color: 'success', icon: '✅' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className="h4 mb-0">{c.icon}</div>
                <div className={`h4 mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Protocol Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Protocol Distribution</div>
              <div className="card-body">
                {Object.entries(protDist).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{k}</span><span className="fw-bold">{v}</span>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div className="progress-bar bg-primary" style={{ width: `${Math.round(v / overview.total_studies * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Field Strength + Manufacturer */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm mb-2">
              <div className="card-header fw-bold">Field Strength</div>
              <div className="card-body py-2">
                {Object.entries(fieldDist).map(([k, v]) => (
                  <span key={k} className={`badge bg-${FIELD_BADGE[k] || 'secondary'} me-2 mb-1`}>
                    {k}: {v}
                  </span>
                ))}
              </div>
            </div>
            <div className="card shadow-sm mb-2">
              <div className="card-header fw-bold">Manufacturer</div>
              <div className="card-body py-2">
                {Object.entries(mfrDist).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between small border-bottom py-1">
                    <span>{k}</span><span className="fw-bold">{v}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Reading Status</div>
              <div className="card-body py-2">
                {Object.entries(readDist).map(([k, v]) => (
                  <span key={k} className={`badge bg-${STATUS_COLOR[k] || 'secondary'} me-2 mb-1`}>
                    {k}: {v}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Classification & Lesion */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">MRI Classification (Clinical Linkage)</div>
              <div className="card-body">
                {Object.entries(clsDist).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between align-items-center mb-2">
                    <span><span className={`badge bg-${CLS_COLOR[k] || 'secondary'} me-2`}>{k}</span></span>
                    <div className="d-flex align-items-center gap-2" style={{ minWidth: 140 }}>
                      <div className="progress flex-grow-1" style={{ height: 8 }}>
                        <div className={`progress-bar bg-${CLS_COLOR[k] || 'secondary'}`}
                          style={{ width: `${Math.round(v / overview.total_studies * 100)}%` }} />
                      </div>
                      <small>{v}</small>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Lesion type */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Lesion Type Distribution</div>
              <div className="card-body">
                {Object.entries(lesionDist).map(([k, v]) => (
                  <div key={k} className="d-flex justify-content-between small border-bottom py-1">
                    <span>{k}</span><span className="badge bg-secondary">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Study Browser ── */}
      {tab === 'studies' && breakdown && (
        <div>
          <p className="text-muted small">Showing {studies.length} most recent of {breakdown.all_study_count} studies</p>
          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle">
              <thead className="table-dark">
                <tr>
                  <th>Patient</th>
                  <th>Date</th>
                  <th>Protocol</th>
                  <th>Field</th>
                  <th>Scanner</th>
                  <th>Series</th>
                  <th>Instances</th>
                  <th>Classification</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {studies.map(s => (
                  <>
                    <tr key={s.study_uid}>
                      <td><code className="small">{s.patient_id}</code></td>
                      <td className="small">{s.study_date.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3')}</td>
                      <td className="small">{s.protocol_label}</td>
                      <td><span className={`badge bg-${FIELD_BADGE[s.field_strength] || 'secondary'}`}>{s.field_strength}</span></td>
                      <td className="small text-muted">{s.scanner_model}</td>
                      <td className="text-center">{s.series_count}</td>
                      <td className="text-center">{s.instance_count}</td>
                      <td><span className={`badge bg-${CLS_COLOR[s.mri_classification] || 'secondary'}`}>{s.mri_classification}</span></td>
                      <td><span className={`badge bg-${STATUS_COLOR[s.reading_status] || 'secondary'}`}>{s.reading_status}</span></td>
                      <td>
                        <button className="btn btn-outline-primary btn-sm"
                          onClick={() => setExpandedStudy(expandedStudy === s.study_uid ? null : s.study_uid)}>
                          {expandedStudy === s.study_uid ? '▲' : '▼'}
                        </button>
                      </td>
                    </tr>
                    {expandedStudy === s.study_uid && (
                      <tr key={`${s.study_uid}-detail`}>
                        <td colSpan={10} className="bg-light">
                          <div className="p-2">
                            <div className="row">
                              <div className="col-md-4">
                                <strong>Study UID:</strong><br />
                                <code className="small">{s.study_uid}</code>
                              </div>
                              <div className="col-md-4">
                                <strong>Accession:</strong> {s.accession_number}<br />
                                <strong>Radiologist:</strong> {s.radiologist}<br />
                                <strong>Lesion:</strong> {s.lesion_type} — {s.lesion_location} ({s.laterality})
                              </div>
                              <div className="col-md-4">
                                <strong>WADO-RS:</strong><br />
                                <code className="small text-muted" style={{ wordBreak: 'break-all' }}>{s.wado_retrieve_url}</code>
                              </div>
                            </div>
                            <hr className="my-2" />
                            <strong>Series:</strong>
                            <table className="table table-xs table-bordered mt-1 mb-0 small">
                              <thead><tr><th>#</th><th>Sequence</th><th>Slices</th><th>Thickness</th><th>TR (ms)</th><th>TE (ms)</th><th>FA°</th></tr></thead>
                              <tbody>
                                {(s.series || []).map(sr => (
                                  <tr key={sr.series_uid}>
                                    <td>{sr.series_number}</td>
                                    <td>{sr.sequence_name}</td>
                                    <td>{sr.instance_count}</td>
                                    <td>{sr.slice_thickness_mm} mm</td>
                                    <td>{sr.tr_ms}</td>
                                    <td>{sr.te_ms}</td>
                                    <td>{sr.flip_angle_deg}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>

          {/* Radiologist workload */}
          <div className="card shadow-sm mt-3">
            <div className="card-header fw-bold">Radiologist Workload</div>
            <div className="card-body">
              {Object.entries(breakdown.radiologist_workload || {}).map(([k, v]) => (
                <div key={k} className="d-flex justify-content-between small border-bottom py-1">
                  <span>{k}</span><span className="badge bg-primary">{v} studies</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── DICOM Tags ── */}
      {tab === 'tags' && breakdown && (
        <div>
          <p className="text-muted small mb-2">Select a study to inspect its DICOM tag values</p>
          <div className="mb-3">
            <select className="form-select form-select-sm" style={{ maxWidth: 320 }}
              value={tagStudy || ''} onChange={e => setTagStudy(e.target.value)}>
              <option value="">-- select patient --</option>
              {studies.map(s => (
                <option key={s.study_uid} value={s.study_uid}>
                  {s.patient_id} — {s.study_date.replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3')} — {s.protocol_label}
                </option>
              ))}
            </select>
          </div>
          {tagStudy && (() => {
            const st = studies.find(s => s.study_uid === tagStudy);
            if (!st) return null;
            return (
              <div className="card shadow-sm">
                <div className="card-header fw-bold">
                  DICOM Tags — {st.patient_id} &nbsp;
                  <span className={`badge bg-${STATUS_COLOR[st.reading_status]}`}>{st.reading_status}</span>
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-dark">
                      <tr><th>Tag</th><th>VR</th><th>Name</th><th>Value</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(st.dicom_tags || {}).map(([tag, info]) => (
                        <tr key={tag}>
                          <td><code className="small">{tag}</code></td>
                          <td><span className="badge bg-secondary">{info.vr}</span></td>
                          <td className="small">{info.tag}</td>
                          <td className="small text-break">{String(info.value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* ── Protocols ── */}
      {tab === 'protocols' && defs && (
        <div className="row">
          {(defs.protocols || []).map(p => (
            <div key={p.code} className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold d-flex justify-content-between align-items-center">
                  <span>{p.label}</span>
                  <div>
                    <span className={`badge bg-${FIELD_BADGE[p.field_strength] || 'secondary'} me-1`}>{p.field_strength}</span>
                    <span className="badge bg-secondary">{p.priority}</span>
                  </div>
                </div>
                <div className="card-body">
                  <p className="small fw-bold mb-1">Sequences ({p.sequences.length}):</p>
                  <ul className="list-unstyled small">
                    {p.sequences.map((seq, i) => (
                      <li key={i} className="border-bottom py-1">
                        <span className="badge bg-info text-dark me-1">{i + 1}</span> {seq}
                      </li>
                    ))}
                  </ul>
                  <p className="text-muted small mb-0">
                    Protocol count: <strong>{(overview.protocol_distribution || {})[p.label] || 0}</strong> studies
                  </p>
                </div>
              </div>
            </div>
          ))}

          {/* Protocol-to-series mapping */}
          {breakdown?.protocol_series_map && (
            <div className="col-12 mt-2">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Protocol → Sequence Composition</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-dark">
                      <tr><th>Protocol</th><th>Studies</th><th>Unique Sequences</th></tr>
                    </thead>
                    <tbody>
                      {Object.entries(breakdown.protocol_series_map).map(([k, v]) => (
                        <tr key={k}>
                          <td className="small">{k}</td>
                          <td>{v.studies}</td>
                          <td className="small">{v.sequences.join(' · ')}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-12 mb-3">
            <div className="alert alert-info small">
              <strong>{defs.title}</strong> — {defs.standard}<br />
              {defs.clinical_relevance}
            </div>
          </div>

          {/* Hierarchy */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">DICOM Hierarchy</div>
              <div className="card-body">
                {Object.entries(defs.hierarchy || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <span className="badge bg-primary me-1">{k}</span>
                    <p className="small text-muted mb-0 mt-1">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* DICOMweb services */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">DICOMweb Services</div>
              <div className="card-body">
                {Object.entries(defs.wado_services || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <span className="badge bg-dark me-1">{k}</span>
                    <p className="small text-muted mb-0 mt-1">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Field strength notes */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Field Strength Notes</div>
              <div className="card-body">
                {Object.entries(defs.field_strength_notes || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <span className={`badge bg-${FIELD_BADGE[k] || 'secondary'} me-1`}>{k}</span>
                    <p className="small text-muted mb-0 mt-1">{v}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* SOP Classes */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">SOP Classes</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-dark"><tr><th>Name</th><th>UID</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.sop_classes || []).map(s => (
                      <tr key={s.uid}>
                        <td className="small fw-bold">{s.name}</td>
                        <td><code className="small">{s.uid}</code></td>
                        <td className="small text-muted">{s.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Key Tags */}
          <div className="col-md-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Key DICOM Tags</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark">
                    <tr><th>Tag</th><th>VR</th><th>Name</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {(defs.key_tags || []).map(t => (
                      <tr key={t.tag}>
                        <td><code className="small">{t.tag}</code></td>
                        <td><span className="badge bg-secondary">{t.vr}</span></td>
                        <td className="small fw-bold">{t.name}</td>
                        <td className="small text-muted">{t.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Reading statuses */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Reading Statuses</div>
              <div className="card-body">
                {Object.entries(defs.reading_statuses || {}).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <span className={`badge bg-${STATUS_COLOR[k] || 'secondary'} me-1`}>{k}</span>
                    <span className="small text-muted">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
