'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Treatments & Genetics', 'Definitions'];

// CPT1A colour scheme — deep teal/cyan (carnitine shuttle; liver; Arctic populations)
const ACCENT  = '#006064';   // deep teal — CPT1A / carnitine shuttle / hepatic
const ACCENT2 = '#00838f';   // medium teal — carnitine / C0 elevated marker
const ACCENT3 = '#1b5e20';   // deep green — KEY POSITIVES (C0 elevated, MCT therapeutic)
const ACCENT4 = '#b71c1c';   // deep red — ABSOLUTE CI (fasting, KD)
const ACCENT5 = '#4a148c';   // dark purple — Arctic founder / genetics
const ACCENT6 = '#e65100';   // deep orange — hyperammonemia (unique among FAO)
const ACCENT7 = '#37474f';   // dark slate — NBS / epidemiology
const ACCENT8 = '#880e4f';   // dark rose — exam traps (no cardiomyopathy)

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

function PctBar({ label, pct, color = ACCENT }) {
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function InfoBox({ title, children, color = ACCENT }) {
  return (
    <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold" style={{ color }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.72rem' }}>
      {text}
    </span>
  );
}

// ── Overview tab ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const b = data.biomarkers || {};
  const cf = data.clinical_features || {};
  const sd = data.severity_distribution || {};

  return (
    <div>
      {/* KPIs */}
      <div className="row g-3 mb-4">
        <KPI label="Patients" value={data.n_patients} color={ACCENT} />
        <KPI label="Seed" value={data.seed} color={ACCENT7} />
        <KPI label="OMIM Disease" value={data.omim_disease} color={ACCENT5} />
        <KPI label="OMIM Gene" value={data.omim_gene} color={ACCENT5} />
        <KPI label="Locus" value={data.locus} color={ACCENT2} />
        <KPI label="Inheritance" value="AR" color={ACCENT7} />
      </div>

      {/* High C0 — inverse marker banner */}
      <div className="alert mb-4" style={{ backgroundColor: '#e0f7fa', borderLeft: `5px solid ${ACCENT}` }}>
        <h6 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🔵 PRIMARY NBS MARKER: C0 (Free Carnitine) ELEVATED — THE INVERTED PROFILE
        </h6>
        <p className="mb-0 small">
          CPT1A deficiency presents the <strong>OPPOSITE</strong> of most FAO disorders: <strong>HIGH C0</strong> (≥60 μmol/L) with{' '}
          <strong>NORMAL/LOW C16</strong> and C18:1. The carnitine shuttle step 1 is blocked, so carnitine{' '}
          accumulates (unused) rather than being consumed to form acylcarnitines.{' '}
          <strong>C0/(C16+C18) ratio &gt;40</strong> = HIGHLY SPECIFIC discriminator.
        </p>
      </div>

      {/* Key biomarkers */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <InfoBox title="🔵 Average C0 (Free Carnitine)" color={ACCENT}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT }}>{b.avg_c0_umol} μmol/L</div>
            <div className="text-muted small">Normal: &lt;50 μmol/L · NBS flag: ≥60 μmol/L · CPT1A: often 60–160 μmol/L</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="🔵 Average C0/C16 Ratio" color={ACCENT2}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT2 }}>{b.avg_c0_c16_ratio}</div>
            <div className="text-muted small">Diagnostic &gt;40 · C16 NORMAL/LOW in CPT1A (not elevated like VLCAD)</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="⚠️ Average Ammonia at Crisis" color={ACCENT6}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT6 }}>{b.avg_ammonia_umol} μmol/L</div>
            <div className="text-muted small">UNIQUE among FAO disorders · Normal: &lt;50 μmol/L · CPT1A: 50–600 μmol/L</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="Average Glucose at Crisis" color={ACCENT4}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT4 }}>{b.avg_glucose_crisis_mmol} mmol/L</div>
            <div className="text-muted small">HYPOKETOTIC hypoglycaemia · Inappropriately low ketones despite low glucose</div>
          </InfoBox>
        </div>
      </div>

      {/* Severity distribution */}
      <InfoBox title="📊 Severity Distribution (40-Patient Cohort)" color={ACCENT7}>
        <div className="row">
          {Object.entries(sd).map(([grp, n]) => (
            <div key={grp} className="col-md-4 mb-2">
              <PctBar
                label={grp}
                pct={Math.round(n / data.n_patients * 100)}
                color={grp.startsWith('Severe') ? ACCENT4 : grp.startsWith('Mild') ? ACCENT3 : ACCENT2}
              />
              <div className="text-muted small">{n} patients</div>
            </div>
          ))}
        </div>
      </InfoBox>

      {/* Clinical features */}
      <InfoBox title="🏥 Clinical Features (40-Patient Cohort)" color={ACCENT7}>
        <div className="row">
          {Object.entries(cf).map(([feat, n]) => (
            <div key={feat} className="col-6 col-md-4 mb-2">
              <div className="d-flex justify-content-between small">
                <span style={{ color: feat === 'cardiomyopathy' ? ACCENT8 : 'inherit' }}>
                  {feat === 'cardiomyopathy' ? '❌ ' : ''}{feat.replace(/_/g, ' ')}
                </span>
                <span className="fw-bold" style={{ color: n === 0 ? ACCENT3 : ACCENT }}>
                  {n}/{data.n_patients}
                </span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{
                  width: `${n / data.n_patients * 100}%`,
                  backgroundColor: feat === 'cardiomyopathy' ? ACCENT8 : ACCENT,
                }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="NO cardiomyopathy (CPT1B serves heart — EXAM TRAP)" color={ACCENT8} />
          <Badge text="NO rhabdomyolysis (liver isoform)" color={ACCENT4} />
          <Badge text="Hyperammonemia UNIQUE among FAO disorders" color={ACCENT6} />
        </div>
      </InfoBox>

      {/* Key exam facts */}
      <InfoBox title="🎯 Highest-Yield Exam Facts" color={ACCENT}>
        <ol className="mb-0 small">
          {(data.key_exam_facts || []).map((f, i) => (
            <li key={i} className="mb-1"
              style={{ color: f.includes('UNIQUE') || f.includes('ABSOLUTE') ? ACCENT4 : f.includes('NEGATIVE') ? ACCENT8 : 'inherit' }}>
              {f}
            </li>
          ))}
        </ol>
      </InfoBox>

      {/* Pathway */}
      <InfoBox title="🔬 Carnitine Shuttle — Where CPT1A Acts (Step 1)" color={ACCENT2}>
        <div className="font-monospace small p-2 rounded" style={{ backgroundColor: '#e0f7fa' }}>
          <div><strong style={{ color: ACCENT4 }}>← CPT1A BLOCK (outer IMM, cytosolic face)</strong></div>
          <div>Step 1 (CPT1A): Long-chain acyl-CoA + Carnitine → Acylcarnitine + CoA-SH</div>
          <div className="text-muted">Step 2 (CACT/SLC25A20): Acylcarnitine (in) ↔ Carnitine (out) [antiport]</div>
          <div className="text-muted">Step 3 (CPT2): Acylcarnitine + CoA-SH → Acyl-CoA + Carnitine (matrix)</div>
          <div className="text-muted">Steps 4–7: Beta-oxidation (VLCAD → HADHA → HADHA → HADHB for long chain)</div>
          <div className="mt-1" style={{ color: ACCENT3 }}>
            ✅ MCT (C8/C10) BYPASSES CPT1A — medium-chain FA enter IMM directly → THERAPEUTIC
          </div>
          <div style={{ color: ACCENT5 }}>
            🧬 CPT1B (heart/muscle, 22q13.33) + CPT1C (brain, 19q13.33) = INTACT → no cardiac/muscle disease
          </div>
        </div>
      </InfoBox>

      {/* Arctic founder callout */}
      <InfoBox title="🌨️ Arctic Founder Variant — p.Pro479Leu (c.1436C>T)" color={ACCENT5}>
        <div className="small">
          <p className="mb-1">
            <strong>Most common CPT1A pathogenic variant globally by allele frequency</strong> — but confined to Arctic/subarctic populations.
          </p>
          <div className="row">
            <div className="col-md-6">
              <Badge text="Inuit (Canada/Greenland): up to 85% allele freq" color={ACCENT5} />
              <Badge text="Alaska Native: 40–70% allele freq" color={ACCENT5} />
              <Badge text="First Nations (northern Canada): 15–30%" color={ACCENT5} />
            </div>
            <div className="col-md-6">
              <Badge text="30–40% residual CPT1A activity retained" color={ACCENT2} />
              <Badge text="Hypomorphic — mostly asymptomatic if fast avoided" color={ACCENT3} />
              <Badge text="Controversial: pathogenic vs population-adapted variant" color={ACCENT7} />
            </div>
          </div>
        </div>
      </InfoBox>
    </div>
  );
}

// ── Patients & Biomarkers tab ─────────────────────────────────────────────────
function PatientsTab({ data }) {
  const [filter, setFilter] = useState('All');
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;

  const pts = data.patients || [];
  const bySev = data.by_severity || {};
  const nbs = data.nbs_profile_summary || {};

  const severities = ['All', ...new Set(pts.map(p => p.severity.split('(')[0].trim()))];
  const filtered = filter === 'All' ? pts : pts.filter(p => p.severity.startsWith(filter));

  return (
    <div>
      {/* NBS Profile Summary */}
      <InfoBox title="📊 NBS Profile Summary (CPT1A inverted pattern)" color={ACCENT}>
        <div className="row text-center">
          <div className="col-md-3 mb-2">
            <div className="fw-bold fs-5" style={{ color: ACCENT }}>{nbs.pct_c0_elevated_ge60}%</div>
            <div className="text-muted small">C0 ≥60 μmol/L (primary flag)</div>
          </div>
          <div className="col-md-3 mb-2">
            <div className="fw-bold fs-5" style={{ color: ACCENT2 }}>{nbs.pct_c0_c16_ratio_gt40}%</div>
            <div className="text-muted small">C0/C16 ratio &gt;40 (diagnostic)</div>
          </div>
          <div className="col-md-3 mb-2">
            <div className="fw-bold fs-5" style={{ color: ACCENT6 }}>{nbs.pct_hyperammonemia}%</div>
            <div className="text-muted small">Hyperammonemia &gt;80 μmol/L</div>
          </div>
          <div className="col-md-3 mb-2">
            <div className="fw-bold fs-5" style={{ color: ACCENT3 }}>0%</div>
            <div className="text-muted small">Cardiomyopathy (NONE — exam trap)</div>
          </div>
        </div>
      </InfoBox>

      {/* By severity summary */}
      <InfoBox title="📋 By Severity Group" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#e0f7fa' }}>
              <tr>
                <th>Severity</th><th>N</th><th>Avg C0</th><th>Avg C16</th>
                <th>Avg C0/C16</th><th>Avg Glucose</th><th>Avg NH3</th><th>Crisis %</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(bySev).map(([grp, s]) => (
                <tr key={grp}>
                  <td style={{ color: grp.startsWith('Severe') ? ACCENT4 : grp.startsWith('Mild') ? ACCENT3 : ACCENT }}>
                    {grp}
                  </td>
                  <td>{s.n}</td>
                  <td className="fw-bold" style={{ color: ACCENT }}>{s.avg_c0}</td>
                  <td>{s.avg_c16}</td>
                  <td className="fw-bold" style={{ color: ACCENT2 }}>{s.avg_c0_c16}</td>
                  <td style={{ color: ACCENT4 }}>{s.avg_glucose}</td>
                  <td style={{ color: ACCENT6 }}>{s.avg_ammonia}</td>
                  <td>{s.crisis_rate}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </InfoBox>

      {/* Patient filter */}
      <div className="d-flex gap-2 flex-wrap mb-3">
        {severities.map(s => (
          <button key={s} className="btn btn-sm" onClick={() => setFilter(s)}
            style={{ backgroundColor: filter === s ? ACCENT : '#e0f7fa', color: filter === s ? '#fff' : ACCENT }}>
            {s}
          </button>
        ))}
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead style={{ backgroundColor: '#e0f7fa' }}>
            <tr>
              <th>ID</th><th>Severity</th><th>Variant</th><th>Onset (mo)</th>
              <th>C0 μmol</th><th>C16 μmol</th><th>C0/C16</th>
              <th>Gluc</th><th>NH3</th><th>Hepato</th><th>Cardiac</th><th>Crisis</th><th>Arctic</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}>
                <td className="font-monospace text-muted">{p.id}</td>
                <td style={{ color: p.severity.startsWith('Severe') ? ACCENT4 : p.severity.startsWith('Mild') ? ACCENT3 : ACCENT }}>
                  {p.severity.split('(')[0].trim()}
                </td>
                <td className="font-monospace small" style={{ color: p.arctic_founder ? ACCENT5 : 'inherit', maxWidth: 200, overflow: 'hidden' }}>
                  {p.variant}
                </td>
                <td>{p.onset_age_months}</td>
                <td className="fw-bold" style={{ color: p.c0_umol >= 60 ? ACCENT : 'inherit' }}>{p.c0_umol}</td>
                <td>{p.c16_umol}</td>
                <td className="fw-bold" style={{ color: p.c0_c16_ratio > 40 ? ACCENT2 : 'inherit' }}>{p.c0_c16_ratio}</td>
                <td style={{ color: p.glucose_mmol < 2.5 ? ACCENT4 : 'inherit' }}>{p.glucose_mmol}</td>
                <td style={{ color: p.ammonia_umol > 80 ? ACCENT6 : 'inherit' }}>{p.ammonia_umol}</td>
                <td>{p.hepatomegaly ? '✓' : '–'}</td>
                <td style={{ color: ACCENT3 }}>{p.cardiomyopathy ? '⚠' : '✓ 0'}</td>
                <td style={{ color: p.metabolic_crisis ? ACCENT4 : 'inherit' }}>{p.metabolic_crisis ? '⚠' : '–'}</td>
                <td style={{ color: p.arctic_founder ? ACCENT5 : 'inherit' }}>{p.arctic_founder ? '🌨' : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Treatments & Genetics tab ─────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  const vc = data.variant_counts || {};
  const tr = data.treatment_summary || {};

  return (
    <div>
      {/* Treatment summary */}
      <InfoBox title="💊 Treatment Summary (40-Patient Cohort)" color={ACCENT}>
        <div className="row">
          {Object.entries(tr).map(([k, v]) => (
            <div key={k} className="col-md-4 mb-2 d-flex align-items-center gap-2">
              <span className="fw-bold fs-5" style={{ color: ACCENT }}>{v}</span>
              <span className="small text-muted">{k.replace(/_/g, ' ')}</span>
            </div>
          ))}
        </div>
      </InfoBox>

      {/* MCT therapeutic banner */}
      <div className="alert" style={{ backgroundColor: '#e8f5e9', borderLeft: `5px solid ${ACCENT3}` }}>
        <h6 className="fw-bold" style={{ color: ACCENT3 }}>✅ MCT OIL — THERAPEUTIC IN CPT1A (Level A)</h6>
        <p className="mb-0 small">
          Medium-chain fatty acids (C8 octanoate, C10 decanoate) <strong>do NOT require CPT1A</strong> to enter the
          mitochondrial matrix — they cross the inner mitochondrial membrane directly via FATP/acylcarnitine-independent
          pathways. Once inside, MCAD/SCAD oxidise them → acetyl-CoA → ketone bodies.
          This is fundamentally different from FASTING (which releases long-chain FA that need CPT1A).
          <strong> Unlike VLCAD/LCHAD where MCT is therapeutic for different reasons, CPT1A uniquely benefits because medium-chain
          FA bypass the blocked step entirely.</strong>
        </p>
      </div>

      {/* Contraindications */}
      <InfoBox title="🚫 Contraindications & High-Risk Treatments" color={ACCENT4}>
        <div className="row">
          <div className="col-md-6">
            <div className="fw-bold small mb-1" style={{ color: ACCENT4 }}>ABSOLUTE CONTRAINDICATIONS</div>
            <ul className="small mb-0">
              <li><strong>Fasting</strong> — ABSOLUTE CI (primary trigger; lipolysis → long-chain FA → cannot be oxidised)</li>
              <li><strong>Ketogenic Diet</strong> — ABSOLUTE CI (requires long-chain FA to generate ketones; CPT1A blocks entry)</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-1" style={{ color: ACCENT6 }}>HIGH RISK / NOT INDICATED</div>
            <ul className="small mb-0">
              <li><strong>Valproate (VPA)</strong> — HIGH RISK (inhibits FAO; depletes carnitine; avoid in all FAO disorders)</li>
              <li><strong>L-Carnitine</strong> — NOT ROUTINE (C0 already elevated; may accumulate toxic long-chain acylcarnitines)</li>
            </ul>
          </div>
        </div>
      </InfoBox>

      {/* Variant frequency chart */}
      <InfoBox title="🧬 Variant Distribution (Leading Allele per Patient)" color={ACCENT5}>
        {Object.entries(vc).sort((a,b) => b[1]-a[1]).map(([v, n]) => (
          <PctBar key={v} label={v} pct={Math.round(n / 40 * 100)} color={v.includes('Pro479') ? ACCENT5 : ACCENT} />
        ))}
      </InfoBox>

      {/* CPT1A vs other carnitine shuttle defects */}
      <InfoBox title="⚖️ Carnitine Shuttle Defects — Differential Diagnosis" color={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#e0f7fa' }}>
              <tr><th>Feature</th><th>CPT1A ↗</th><th>CACT (SLC25A20) ↗</th><th>CPT2 ↗</th></tr>
            </thead>
            <tbody>
              <tr><td>Step blocked</td><td>Step 1 (outer IMM)</td><td>Step 2 (translocase)</td><td>Step 3 (inner IMM)</td></tr>
              <tr><td>C0 (free carnitine)</td>
                <td className="fw-bold" style={{ color: ACCENT }}>HIGH ≥60</td>
                <td style={{ color: ACCENT4 }}>Low</td><td style={{ color: ACCENT4 }}>Low-Normal</td></tr>
              <tr><td>C16 (palmitoylcarnitine)</td>
                <td style={{ color: ACCENT3 }}>Normal/Low</td>
                <td className="fw-bold" style={{ color: ACCENT4 }}>HIGH</td>
                <td className="fw-bold" style={{ color: ACCENT4 }}>HIGH</td></tr>
              <tr><td>Cardiac disease</td>
                <td className="fw-bold" style={{ color: ACCENT3 }}>NONE (CPT1B intact)</td>
                <td style={{ color: ACCENT4 }}>Yes (severe)</td>
                <td style={{ color: ACCENT4 }}>Yes (severe form)</td></tr>
              <tr><td>Rhabdomyolysis</td>
                <td className="fw-bold" style={{ color: ACCENT3 }}>NONE</td>
                <td style={{ color: ACCENT4 }}>Yes</td>
                <td style={{ color: ACCENT4 }}>Yes (mild form)</td></tr>
              <tr><td>Hyperammonemia</td>
                <td className="fw-bold" style={{ color: ACCENT6 }}>YES (unique)</td>
                <td>Possible</td><td>Rare</td></tr>
              <tr><td>Primary organ</td><td>Liver</td><td>Heart/liver</td><td>Muscle/liver/heart</td></tr>
            </tbody>
          </table>
        </div>
      </InfoBox>

      {/* Compare to other FAO disorders */}
      <InfoBox title="🔬 CPT1A vs Other FAO Disorders — NBS Key Differences" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#e0f7fa' }}>
              <tr><th>Marker</th><th style={{ color: ACCENT }}>CPT1A</th><th>MCAD</th><th>VLCAD</th><th>LCHAD</th></tr>
            </thead>
            <tbody>
              <tr><td>C0 (free carnitine)</td>
                <td className="fw-bold" style={{ color: ACCENT }}>HIGH ↑↑</td>
                <td style={{ color: ACCENT4 }}>Low ↓</td><td style={{ color: ACCENT4 }}>Low ↓</td><td style={{ color: ACCENT4 }}>Low ↓</td></tr>
              <tr><td>C8 octanoylcarnitine</td>
                <td style={{ color: ACCENT3 }}>Normal</td>
                <td className="fw-bold" style={{ color: ACCENT4 }}>HIGH ↑↑</td>
                <td style={{ color: ACCENT3 }}>Normal</td><td style={{ color: ACCENT3 }}>Normal</td></tr>
              <tr><td>C14:1 tetradecenoyl</td>
                <td style={{ color: ACCENT3 }}>Normal</td>
                <td style={{ color: ACCENT3 }}>Normal</td>
                <td className="fw-bold" style={{ color: ACCENT4 }}>HIGH ↑↑</td>
                <td style={{ color: ACCENT3 }}>Normal</td></tr>
              <tr><td>C16-OH (3-hydroxy)</td>
                <td style={{ color: ACCENT3 }}>Normal</td>
                <td style={{ color: ACCENT3 }}>Normal</td>
                <td style={{ color: ACCENT3 }}>Normal</td>
                <td className="fw-bold" style={{ color: ACCENT4 }}>HIGH ↑↑</td></tr>
              <tr><td>Cardiomyopathy</td>
                <td className="fw-bold" style={{ color: ACCENT3 }}>NONE</td>
                <td style={{ color: ACCENT3 }}>NONE</td>
                <td style={{ color: ACCENT4 }}>YES</td>
                <td style={{ color: ACCENT4 }}>YES</td></tr>
              <tr><td>Hyperammonemia</td>
                <td className="fw-bold" style={{ color: ACCENT6 }}>YES (unique)</td>
                <td>Rare</td><td>Rare</td><td>Rare</td></tr>
              <tr><td>MCT therapeutic?</td>
                <td className="fw-bold" style={{ color: ACCENT3 }}>YES (bypass)</td>
                <td>No benefit</td>
                <td style={{ color: ACCENT3 }}>YES</td>
                <td style={{ color: ACCENT3 }}>YES</td></tr>
            </tbody>
          </table>
        </div>
      </InfoBox>
    </div>
  );
}

// ── Definitions tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;

  return (
    <div>
      <InfoBox title="🧬 Disease Overview" color={ACCENT}>
        <table className="table table-sm small mb-0">
          <tbody>
            <tr><td className="fw-bold text-muted" style={{ width: 180 }}>Disease</td><td>{data.disease_name}</td></tr>
            <tr><td className="fw-bold text-muted">Gene</td><td>{data.gene}</td></tr>
            <tr><td className="fw-bold text-muted">Locus</td><td>{data.locus}</td></tr>
            <tr><td className="fw-bold text-muted">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold text-muted">OMIM Disease</td><td>{data.omim_disease}</td></tr>
            <tr><td className="fw-bold text-muted">Inheritance</td><td>{data.inheritance}</td></tr>
          </tbody>
        </table>
      </InfoBox>

      <InfoBox title="⚗️ Enzymatic Function" color={ACCENT2}>
        <p className="small mb-0">{data.enzymatic_function}</p>
      </InfoBox>

      <InfoBox title="🔬 Protein" color={ACCENT2}>
        <p className="small mb-0">{data.protein}</p>
      </InfoBox>

      <InfoBox title="🚧 Metabolic Block" color={ACCENT4}>
        <p className="small mb-0">{data.metabolic_block}</p>
      </InfoBox>

      <InfoBox title="📊 NBS Primary Marker" color={ACCENT}>
        <p className="small mb-0">{data.nbs_marker}</p>
      </InfoBox>

      <InfoBox title="🧪 Key Biomarkers" color={ACCENT2}>
        <table className="table table-sm small mb-0">
          <tbody>
            {Object.entries(data.key_biomarkers || {}).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold text-muted font-monospace" style={{ width: 240 }}>{k.replace(/_/g, ' ')}</td>
                <td style={{ color: k.includes('C0') && !k.includes('neg') ? ACCENT : k === 'C8_octanoylcarnitine' || k === 'C14_1' || k === 'C16_OH' ? ACCENT3 : k === 'Ammonia' ? ACCENT6 : 'inherit' }}>
                  {v}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </InfoBox>

      <InfoBox title="🏥 Clinical Features" color={ACCENT7}>
        <table className="table table-sm small mb-0">
          <tbody>
            {Object.entries(data.clinical_features || {}).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold text-muted" style={{ width: 240 }}>{k.replace(/_/g, ' ')}</td>
                <td style={{ color: k.includes('NO_') ? ACCENT3 : k === 'Hyperammonemia' ? ACCENT6 : 'inherit' }}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </InfoBox>

      <InfoBox title="💊 Treatment" color={ACCENT3}>
        <table className="table table-sm small mb-0">
          <tbody>
            {Object.entries(data.treatment || {}).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold" style={{ width: 200, color: k.includes('CI') ? ACCENT4 : k.includes('THERAPEUTIC') ? ACCENT3 : k.includes('HIGH_RISK') ? ACCENT6 : ACCENT }}>{k.replace(/_/g, ' ')}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </InfoBox>

      <InfoBox title="🚫 Contraindications" color={ACCENT4}>
        <ul className="small mb-0">
          {(data.contraindications || []).map((c, i) => (
            <li key={i} style={{ color: c.includes('ABSOLUTE') ? ACCENT4 : ACCENT6 }}>{c}</li>
          ))}
        </ul>
      </InfoBox>

      <InfoBox title="🌨️ Genetics — Arctic Founder Variant" color={ACCENT5}>
        <p className="small mb-1"><strong>p.Pro479Leu (c.1436C>T)</strong></p>
        <p className="small mb-0">{(data.genetics || {}).population_note}</p>
      </InfoBox>

      <InfoBox title="⚖️ Key Distinguishing Facts" color={ACCENT2}>
        <ul className="small mb-0">
          {(data.key_distinguishing_facts || []).map((f, i) => <li key={i}>{f}</li>)}
        </ul>
      </InfoBox>

      <InfoBox title="📐 Malonyl-CoA Regulation" color={ACCENT7}>
        <p className="small mb-0">{data.malonyl_coa_regulation}</p>
      </InfoBox>

      <InfoBox title="⚖️ Comparison with CACT & CPT2" color={ACCENT2}>
        {Object.entries(data.comparison_table || {}).map(([k, v]) => (
          <div key={k} className="mb-2">
            <div className="small fw-bold" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </InfoBox>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function CPT1APage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview]  = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]          = useState(null);
  const [error, setError]        = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cpt1a/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Backend offline'));
    fetch(`${API}/api/cpt1a/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/cpt1a/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 d-flex align-items-center gap-2 flex-wrap">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Home</Link>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          🔵 CPT1A Epilepsy Dashboard
        </h4>
        <span className="badge" style={{ backgroundColor: ACCENT }}>CPT1A Deficiency</span>
        <span className="badge" style={{ backgroundColor: ACCENT5 }}>11q13.3</span>
        <span className="badge" style={{ backgroundColor: ACCENT7 }}>AR</span>
        <span className="badge" style={{ backgroundColor: ACCENT6 }}>Hyperammonemia Unique</span>
        <span className="badge" style={{ backgroundColor: ACCENT3 }}>NO Cardiomyopathy</span>
        <span className="badge" style={{ backgroundColor: ACCENT2 }}>C0 HIGH (Inverted Profile)</span>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`}
              style={tab === t ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview'               && <OverviewTab    data={overview}   />}
      {tab === 'Patients & Biomarkers'  && <PatientsTab    data={breakdown}  />}
      {tab === 'Treatments & Genetics'  && <TreatmentsTab  data={breakdown}  />}
      {tab === 'Definitions'            && <DefinitionsTab data={defs}       />}
    </div>
  );
}
