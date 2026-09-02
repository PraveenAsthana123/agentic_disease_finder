'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'N-Module Assembly', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — FAD-oxidoreductase / N-module / FOXRED1
const LIGHT = '#e8eaf6';

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ff  = data.feature_frequencies_pct || {};
  const bf  = data.biochemical_fingerprint || {};
  const p   = data.protein || {};
  const mod = data.foxred1_module_summary || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"             value={data.gene}                  color={COLOR} />
        <KPI label="HCM Rate"         value="~10% (LOW)"                 color="#2e7d32" />
        <KPI label="OMIM Gene"        value={`*${data.omim_gene}`}       color={COLOR} />
        <KPI label="Chromosome"       value={data.chromosome}            color={COLOR} />
        <KPI label="Inheritance"      value={data.inheritance}           color={COLOR} />
        <KPI label="Protein"          value={`${p.size_kda} kDa`}       color={COLOR} />
      </div>

      <div className="alert mb-4" style={{ background: '#fff3e0', borderLeft: '4px solid #e65100' }}>
        <strong>🟠 FAD-BINDING DOMAIN — BUT NO RIBOFLAVIN RESPONSE — Critical DDx vs ACAD9</strong> —
        FOXRED1 contains an FAD-binding oxidoreductase domain, making it superficially similar to ACAD9
        (also FAD-binding). CRITICAL DIFFERENCE: ACAD9 deficiency is riboflavin-responsive (50–60%, Level B).
        FOXRED1 deficiency has ZERO riboflavin response. Do NOT treat as ACAD9 without WES confirmation.
        Riboflavin trial + WES locus (11q24.2 vs 3q21.3) is the mandatory discriminator.
      </div>

      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>🔵 FOXRED1 (11q24.2) — N-Module CI Assembly Chaperone — Different Module from ACAD9/NDUFAF3/4/5/TIMMDC1</strong> —
        FOXRED1 acts on the N-module (NADH dehydrogenase module, matrix arm tip) — entirely distinct from:
        MCIA/ND2-ND5 module (ACAD9, NDUFAF1, ECSIT, TMEM126B), ND1-module (NDUFAF3/4/5, TIMMDC1).
        Same chromosome 11 as NDUFV1 (11q13.2): leukodystrophy on MRI points to NDUFV1, not FOXRED1.
      </div>

      <div className="alert mb-4" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
        <strong>🟢 LOW HCM (~10%) — Key DDx vs TIMMDC1 (&gt;80%), SCO2 (~65%), NDUFV2 (~60%)</strong> —
        FOXRED1 deficiency: low HCM rate (~10%). If HCM is prominent in a CI-deficiency patient,
        consider TIMMDC1 (ND1-module, integral IMM, &gt;80%) or SCO2/NDUFV2 (high HCM).
        Low HCM + N-module BN-PAGE + isolated CI + 11q24.2 = FOXRED1 fingerprint.
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="🔄 Key Pathway Note — FAD-Oxidoreductase / N-Module / No Riboflavin / Chr11 vs NDUFV1">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      {mod.gene && (
        <SectionCard title="⚙️ FOXRED1 Module Summary — N-Module Chaperone">
          <div className="alert mb-2" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <strong>Gene:</strong> {mod.gene}
          </div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="small text-muted mb-1">Module class</div>
              <div className="fw-bold small">{mod.module_class}</div>
            </div>
            <div className="col-md-6">
              <div className="small text-muted mb-1">Assembly position</div>
              <div className="fw-bold small">{mod.assembly_position}</div>
            </div>
          </div>
          <div className="mt-2">
            <div className="small text-muted mb-1">FAD domain — no riboflavin response</div>
            <p className="small mb-1">{mod.fad_domain_unique}</p>
            <div className="small text-muted mb-1">N-module chaperone role</div>
            <p className="small mb-1">{mod.n_module_chaperone_role}</p>
            <div className="small text-muted mb-1">Effect of FOXRED1 loss</div>
            <p className="small mb-0">{mod.foxred1_loss_effect}</p>
          </div>
        </SectionCard>
      )}

      <SectionCard title="🔬 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted">{k.replace(/_/g, ' ')}</span>
            <span
              className={
                k === 'Complex_I' ? 'text-danger fw-bold' :
                k === 'Riboflavin_response' ? 'text-danger' :
                k === 'HCM_rate' ? 'text-success fw-bold' :
                k.startsWith('Complex_') ? 'text-success' : 'fw-bold'
              }
            >{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (40-patient cohort, seed-691)">
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k.replace(/_/g, ' ')} value={v}
            color={
              k === 'HCM' ? '#2e7d32' :           // green — low HCM, protective
              k === 'Riboflavin_responder' || k === 'Peripheral_neuropathy' ||
              k === 'Leukodystrophy' || k === 'Hepatopathy' || k === 'Olfactory_bulb_lesions'
                ? '#4caf50'   // green for hard-0 protective features
                : COLOR
            } />
        ))}
      </SectionCard>

      <SectionCard title="⚖️ Key DDx" borderColor="#1565c0">
        {(data.key_ddx || []).map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#e3f2fd', borderLeft: '4px solid #1565c0' }}>
            <div className="fw-bold small mb-1" style={{ color: '#1565c0' }}>{d.feature}</div>
            <p className="small mb-1">{d.significance}</p>
            <div className="text-muted small">Target gene: <strong>{d.target_gene}</strong></div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚨 Clinical Alerts" borderColor="#b71c1c">
        {(data.clinical_alerts || []).map((a, i) => (
          <div key={i} className="mb-1 p-2 rounded small"
            style={{
              background: a.startsWith('🔴') ? '#ffebee' : a.startsWith('🟠') ? '#fff3e0' : a.startsWith('🟡') ? '#fff8e1' : a.startsWith('🟢') ? '#e8f5e9' : '#e3f2fd',
              borderLeft: `4px solid ${a.startsWith('🔴') ? '#c62828' : a.startsWith('🟠') ? '#e65100' : a.startsWith('🟡') ? '#f57f17' : a.startsWith('🟢') ? '#2e7d32' : '#1565c0'}`
            }}>
            {a}
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ────────────────────────────────────────────────
function PatientsTab({ data }) {
  const [search, setSearch] = useState('');
  if (!data) return <p className="text-muted">Loading…</p>;

  const pts = data.patients || [];
  const filtered = pts.filter(p =>
    (p.allele1 || '').toLowerCase().includes(search.toLowerCase()) ||
    (p.allele2 || '').toLowerCase().includes(search.toLowerCase()) ||
    (p.outcome || '').toLowerCase().includes(search.toLowerCase())
  );

  const od = data.outcome_distribution || {};

  return (
    <>
      <div className="row g-3 mb-4">
        {Object.entries(od).map(([outcome, count]) => (
          <div key={outcome} className="col-6 col-md-3">
            <div className="card shadow-sm text-center">
              <div className="card-body py-2">
                <div className="fw-bold fs-5" style={{ color: outcome.includes('deceased') ? '#c62828' : COLOR }}>{count}</div>
                <div className="text-muted small">{outcome.replace(/_/g, ' ')}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="📊 CI Activity Distribution">
            {Object.entries(data.ci_activity_stats?.bands || {}).map(([band, count]) => (
              <Bar key={band} label={band} value={Math.round(count / (data.cohort_n || 40) * 100)} />
            ))}
            <div className="small text-muted mt-2">
              Mean: {data.ci_activity_stats?.mean}% | Min: {data.ci_activity_stats?.min}% | Max: {data.ci_activity_stats?.max}%
            </div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⏱️ Onset Distribution">
            {Object.entries(data.onset_stats?.bands || {}).map(([band, count]) => (
              <Bar key={band} label={band} value={Math.round(count / (data.cohort_n || 40) * 100)} />
            ))}
            <div className="small text-muted mt-2">
              Mean onset: {data.onset_stats?.mean_months} months
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🧬 Variant Frequency">
        {Object.entries(data.variant_frequency || {}).map(([v, count]) => (
          <div key={v} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="font-monospace text-muted">{v}</span>
            <span className="fw-bold">{count}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔑 N-Module Key Features">
        {Object.entries(data.key_n_module_features || {}).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted">{k.replace(/_/g, ' ')}</span>
            <span className={
              v === true ? 'text-success fw-bold' :
              v === false ? 'text-danger' :
              typeof v === 'number' ? (v < 15 ? 'text-success fw-bold' : 'fw-bold') :
              'fw-bold'
            }>{String(v)}{typeof v === 'number' && k.includes('HCM') ? '%' : ''}</span>
          </div>
        ))}
      </SectionCard>

      <div className="row mb-3">
        <div className="col-md-6">
          <input className="form-control form-control-sm" placeholder="Filter by allele / outcome…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
      </div>

      {filtered.length > 0 && (
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>#</th><th>Onset (m)</th><th>Sex</th><th>Allele 1</th><th>Allele 2</th>
                <th>CI %</th><th>Leigh MRI</th><th>Lactic Ac.</th><th>HCM</th><th>Seizures</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.onset_age_months}</td>
                  <td>{p.sex}</td>
                  <td className="font-monospace" style={{ fontSize: '0.7rem' }}>{p.allele1}</td>
                  <td className="font-monospace" style={{ fontSize: '0.7rem' }}>{p.allele2}</td>
                  <td><span className="badge bg-danger">{p.ci_activity_pct}%</span></td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td>{p.lactic_acidosis ? '✅' : '—'}</td>
                  <td>{p.hcm ? <span className="badge bg-danger" style={{ fontSize: '0.6rem' }}>HCM</span> : '—'}</td>
                  <td>{p.seizures ? '⚠️' : '—'}</td>
                  <td><span className="badge" style={{ background: p.outcome?.includes('deceased') ? '#c62828' : COLOR, fontSize: '0.65rem' }}>{p.outcome}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <SectionCard title="💊 Treatment Summary" borderColor="#2e7d32">
        {Object.entries(data.treatment_summary || {}).map(([k, v]) => (
          <div key={k} className="mb-2 border-bottom pb-1">
            <div className="small fw-bold" style={{ color: k === 'absolute_ci' || k === 'do_not_use' ? '#c62828' : k === 'avoid' ? '#e65100' : '#2e7d32' }}>
              {k.replace(/_/g, ' ').toUpperCase()}
            </div>
            <div className="small text-muted">{Array.isArray(v) ? v.join(' · ') : v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🧪 Variant Table" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>cDNA</th><th>Protein</th><th>Domain</th><th>Severity</th><th>CI Range</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {(data.variant_table || []).map(v => (
                <tr key={v.hgvs_p}>
                  <td className="font-monospace small">{v.hgvs_c}</td>
                  <td className="font-monospace small fw-bold">{v.hgvs_p}</td>
                  <td className="text-muted small">{v.domain}</td>
                  <td><span className="badge" style={{ background: v.severity === 'severe' ? '#c62828' : v.severity === 'moderate' ? '#f57f17' : '#1565c0', fontSize: '0.65rem' }}>{v.severity}</span></td>
                  <td className="text-muted small">{v.ci_range}</td>
                  <td className="text-muted small">{v.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: N-Module Assembly ─────────────────────────────────────────────────────
function NModuleTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>FOXRED1 — N-Module CI Assembly Chaperone (FAD-Oxidoreductase Domain)</strong><br />
        <span className="small text-muted">
          FOXRED1 acts as a dedicated chaperone for the N-module (NADH dehydrogenase module — matrix arm tip).
          It contains an FAD-binding oxidoreductase domain but is NOT riboflavin-responsive — the critical
          distinguisher from ACAD9 (50-60% riboflavin response, MCIA/ND2-ND5 module). FOXRED1 enables
          N-module sub-assembly; its loss stalls the N-module, preventing holoenzyme CI formation.
          Completely distinct from MCIA class (ACAD9/NDUFAF1/ECSIT/TMEM126B) and ND1-module class
          (NDUFAF3/4/5/TIMMDC1). WES (11q24.2) is mandatory. Same chromosome 11 as NDUFV1 (11q13.2).
        </span>
      </div>

      <SectionCard title="🔬 DDx Matrix — FOXRED1 vs Key CI Genes" borderColor="#1565c0">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: '#1565c0', color: '#fff' }}>
              <tr>
                <th>Comparator</th><th>FOXRED1</th><th>Comparator</th><th>Key Test</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_matrix || []).map((row, i) => (
                <tr key={i} style={{ background: row.comparator.includes('ACAD9') ? '#fff8e1' : undefined }}>
                  <td className="fw-bold small">{row.comparator}</td>
                  <td className="text-muted small">{row.foxred1}</td>
                  <td className="text-muted small">{row.comparator_val}</td>
                  <td className="fw-bold small" style={{ color: '#1565c0' }}>{row.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🏗️ CI Assembly Module Comparison" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: '#fff8e1', borderLeft: '4px solid #f57f17' }}>
          <strong>FOXRED1 = N-Module.</strong> Three major CI assembly classes with different BN-PAGE intermediates:
          Class 1 (MCIA/ND2-ND5: ACAD9, NDUFAF1, ECSIT, TMEM126B),
          Class 3 (ND1-module: NDUFAF3/4/5, TIMMDC1),
          N-Module area (FOXRED1, NDUFAF2, NDUFA12). WES is mandatory to distinguish within each class.
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>Factor</th><th>Module</th><th>FAD-binding</th><th>Riboflavin Response</th><th>HCM</th><th>Chromosome</th>
              </tr>
            </thead>
            <tbody>
              {[
                { factor: 'FOXRED1', module: 'N-module (chaperone)', fad: 'Yes', ribo: '0% (none)', hcm: '~10%', chr: '11q24.2', highlight: true },
                { factor: 'ACAD9', module: 'MCIA/ND2-ND5 (Class 1)', fad: 'Yes (ACAD superfamily)', ribo: '50-60% (Level B)', hcm: '55-65%', chr: '3q21.3' },
                { factor: 'NDUFAF1', module: 'MCIA/ND2-ND5 (Class 1)', fad: 'No', ribo: '0%', hcm: '<20%', chr: '15q11.2-q13' },
                { factor: 'NDUFAF3', module: 'ND1-module (Class 3)', fad: 'No', ribo: '0%', hcm: '15-25%', chr: '2q33.1' },
                { factor: 'TIMMDC1', module: 'ND1-module (Class 3)', fad: 'No', ribo: '0%', hcm: '>80%', chr: '3q25.1' },
                { factor: 'NDUFAF2', module: 'N-Q module area', fad: 'No', ribo: '0%', hcm: '<15%', chr: '5q12.1' },
                { factor: 'NDUFV1', module: 'N-module (structural)', fad: 'Yes (FMN)', ribo: '0%', hcm: '<15%', chr: '11q13.2' },
              ].map(r => (
                <tr key={r.factor} style={{ background: r.highlight ? LIGHT : undefined }}>
                  <td className="fw-bold" style={{ color: r.highlight ? COLOR : undefined }}>
                    {r.factor}
                    {r.highlight && <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>THIS</span>}
                  </td>
                  <td className="small text-muted">{r.module}</td>
                  <td className={r.fad.startsWith('Yes') ? 'fw-bold' : 'text-muted'}>{r.fad}</td>
                  <td>
                    <span className="badge" style={{
                      background: r.ribo.includes('50-60%') ? '#1565c0' : '#c62828',
                      fontSize: '0.65rem'
                    }}>{r.ribo}</span>
                  </td>
                  <td className={r.hcm.includes('>80%') ? 'text-danger fw-bold' : r.hcm.includes('55-65%') ? 'fw-bold' : ''}>{r.hcm}</td>
                  <td className="font-monospace small">{r.chr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔍 Same Chromosome 11 — FOXRED1 vs NDUFV1" borderColor="#c62828">
        <div className="alert mb-3" style={{ background: '#ffebee', borderLeft: '4px solid #c62828' }}>
          <strong>FOXRED1 (11q24.2) and NDUFV1 (11q13.2) are on the SAME chromosome 11.</strong>
          WES is mandatory — different bands on the long arm. Leukodystrophy (40-50% in NDUFV1, 0% in FOXRED1)
          is the key MRI discriminator. NDUFV1 is a structural N-module subunit; FOXRED1 is an N-module chaperone.
        </div>
        <div className="row g-3">
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: COLOR }}>FOXRED1 (11q24.2)</div>
            <ul className="list-unstyled small">
              <li>• N-module CI assembly <strong>chaperone</strong> (FAD-oxidoreductase)</li>
              <li>• NOT present in mature CI holoenzyme</li>
              <li>• Leukodystrophy: <strong className="text-success">0%</strong></li>
              <li>• HCM: ~10% (low)</li>
              <li>• Riboflavin response: 0%</li>
              <li>• Long arm band 24.2 (distal)</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: '#c62828' }}>NDUFV1 (11q13.2)</div>
            <ul className="list-unstyled small">
              <li>• N-module CI <strong>structural subunit</strong> (51 kDa, FMN-binding, NADH oxidation)</li>
              <li>• Present in mature CI holoenzyme</li>
              <li>• Leukodystrophy: <strong className="text-danger">40-50%</strong></li>
              <li>• HCM: &lt;15%</li>
              <li>• Riboflavin response: 0%</li>
              <li>• Long arm band 13.2 (proximal)</li>
            </ul>
          </div>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <>
      <SectionCard title="📖 Gene & Disease Definition">
        <p className="small mb-2">{data.gene_definition}</p>
        <p className="small mb-0">{data.disease_definition}</p>
      </SectionCard>

      <SectionCard title="🔬 Module Definitions">
        {(data.module_definitions || []).map(c => (
          <div key={c.term} className="mb-3 border-bottom pb-2">
            <div className="fw-bold small" style={{ color: COLOR }}>{c.term}</div>
            <p className="small text-muted mb-0">{c.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Parameter</th><th>Threshold</th><th>Significance</th></tr></thead>
            <tbody>
              {(data.clinical_thresholds || []).map(t => (
                <tr key={t.parameter}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td className="font-monospace text-danger">{t.threshold}</td>
                  <td className="text-muted">{t.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🧬 Inheritance">
        <p className="small mb-0">{data.inheritance_definition}</p>
      </SectionCard>

      <SectionCard title="📋 Standards & Guidelines">
        {(data.standards || []).map(s => (
          <div key={s.code} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="font-monospace fw-bold" style={{ color: COLOR }}>{s.code}</span>
            <span className="text-muted">{s.title}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 References">
        {(data.references || []).map(r => (
          <div key={r.id} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <p className="small fw-bold mb-1">{r.citation}</p>
            <p className="small text-muted mb-0"><em>{r.relevance}</em></p>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function FOXRED1Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/foxred1/overview`).then(r => r.json()),
      fetch(`${API}/api/foxred1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/foxred1/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 FOXRED1 — Complex I Deficiency (N-Module Chaperone)</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>N-Module Assembly</span>
        <span className="badge ms-1" style={{ background: '#e65100' }}>FAD-Binding — No Riboflavin Response</span>
        <span className="badge ms-1 bg-success">HCM ~10% (Low)</span>
        <span className="badge ms-1 bg-secondary">11q24.2</span>
        <span className="badge ms-1 bg-secondary">OMIM *613622</span>
        <span className="badge ms-1 bg-secondary">AR</span>
      </div>

      {err && <div className="alert alert-danger small">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <NModuleTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
