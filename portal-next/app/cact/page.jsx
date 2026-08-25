'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Treatments & Genetics', 'Definitions'];

// CACT colour scheme — deep amber/orange (carnitine translocase; cardiac; neonatal severe)
const ACCENT  = '#e65100';   // deep orange — CACT / acylcarnitine accumulation / C16 elevated
const ACCENT2 = '#bf360c';   // darker orange — C18:1 elevated / cardiac
const ACCENT3 = '#1b5e20';   // deep green — KEY POSITIVES / MCT therapeutic
const ACCENT4 = '#b71c1c';   // deep red — ABSOLUTE CI (fasting, KD, VPA)
const ACCENT5 = '#4a148c';   // dark purple — genetics
const ACCENT6 = '#006064';   // deep teal — C0 LOW marker (contrast: CPT1A has C0 HIGH)
const ACCENT7 = '#37474f';   // dark slate — NBS / epidemiology
const ACCENT8 = '#880e4f';   // dark rose — cardiomyopathy (HALLMARK — contrast CPT1A where absent)

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

      {/* Primary marker banner */}
      <div className="alert mb-4" style={{ backgroundColor: '#fbe9e7', borderLeft: `5px solid ${ACCENT}` }}>
        <h6 className="fw-bold mb-1" style={{ color: ACCENT }}>
          🟠 PRIMARY NBS MARKER: C16 (Palmitoylcarnitine) ELEVATED — NORMAL-DIRECTION FAO PROFILE
        </h6>
        <p className="mb-0 small">
          CACT deficiency presents with <strong>HIGH C16 + HIGH C18:1 + HIGH C18</strong> and <strong>LOW C0</strong> —
          the typical long-chain FAO disorder profile (opposite of CPT1A which has HIGH C0 + normal C16).{' '}
          The carnitine shuttle step 2 (translocase) is blocked: long-chain acylcarnitines ACCUMULATE in
          the cytoplasm/intermembrane space, CANNOT enter the mitochondrial matrix.{' '}
          <strong>C16-OH is NORMAL</strong> — KEY NEGATIVE vs LCHAD.
          <strong> Cardiomyopathy is the hallmark</strong> — unlike CPT1A where cardiac disease is absent (EXAM TRAP).
        </p>
      </div>

      {/* Key biomarkers */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <InfoBox title="🟠 Average C16 (Palmitoylcarnitine)" color={ACCENT}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT }}>{b.avg_c16_umol} μmol/L</div>
            <div className="text-muted small">Normal: &lt;0.6 μmol/L · NBS flag: ≥1.5 μmol/L · CACT: often 2–8 μmol/L</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="🔵 Average C0 (Free Carnitine) — LOW" color={ACCENT6}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT6 }}>{b.avg_c0_umol} μmol/L</div>
            <div className="text-muted small">NORMAL: 15–50 μmol/L · CACT: LOW (carnitine trapped as acylcarnitines) · OPPOSITE of CPT1A</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="❤️ Cardiomyopathy Rate" color={ACCENT8}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT8 }}>{cf.cardiomyopathy}/{data.n_patients}</div>
            <div className="text-muted small">HALLMARK of CACT · Dilated/hypertrophic · Life-threatening arrhythmias · EXAM TRAP: absent in CPT1A</div>
          </InfoBox>
        </div>
        <div className="col-md-6">
          <InfoBox title="⚡ Arrhythmia Rate" color={ACCENT2}>
            <div className="fs-3 fw-bold" style={{ color: ACCENT2 }}>{cf.arrhythmia}/{data.n_patients}</div>
            <div className="text-muted small">Ventricular arrhythmias · Potentially fatal · Urgent cardiac monitoring required</div>
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
                color={grp.includes('Severe') ? ACCENT4 : ACCENT3}
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
                <span style={{ color: feat === 'cardiomyopathy' ? ACCENT8 : feat === 'arrhythmia' ? ACCENT2 : 'inherit' }}>
                  {feat === 'cardiomyopathy' ? '❤️ ' : feat === 'arrhythmia' ? '⚡ ' : ''}{feat.replace(/_/g, ' ')}
                </span>
                <span className="fw-bold" style={{ color: ACCENT }}>
                  {n}/{data.n_patients}
                </span>
              </div>
              <div className="progress" style={{ height: 6 }}>
                <div className="progress-bar" style={{
                  width: `${n / data.n_patients * 100}%`,
                  backgroundColor: feat === 'cardiomyopathy' ? ACCENT8 : feat === 'arrhythmia' ? ACCENT2 : ACCENT,
                }} />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2">
          <Badge text="Cardiomyopathy HALLMARK (unlike CPT1A — EXAM TRAP)" color={ACCENT8} />
          <Badge text="Rhabdomyolysis present (unlike CPT1A where absent)" color={ACCENT4} />
          <Badge text="C0 LOW (unlike CPT1A where C0 is HIGH)" color={ACCENT6} />
          <Badge text="C16-OH NORMAL (KEY NEGATIVE vs LCHAD)" color={ACCENT3} />
        </div>
      </InfoBox>

      {/* Key exam facts */}
      <InfoBox title="🎯 Highest-Yield Exam Facts" color={ACCENT}>
        <ol className="mb-0 small">
          {(data.key_exam_facts || []).map((f, i) => (
            <li key={i} className="mb-1"
              style={{ color: f.includes('ABSOLUTE') || f.includes('HALLMARK') ? ACCENT4 : f.includes('NEGATIVE') || f.includes('NORMAL') ? ACCENT3 : f.includes('CARDIAC') || f.includes('cardiomyopathy') ? ACCENT8 : 'inherit' }}>
              {f}
            </li>
          ))}
        </ol>
      </InfoBox>

      {/* Pathway */}
      <InfoBox title="🔬 Carnitine Shuttle — Where CACT Acts (Step 2)" color={ACCENT2}>
        <div className="font-monospace small p-2 rounded" style={{ backgroundColor: '#fbe9e7' }}>
          <div className="text-muted">Step 1 (CPT1A): Long-chain acyl-CoA + Carnitine → Acylcarnitine + CoA-SH [outer IMM]</div>
          <div><strong style={{ color: ACCENT4 }}>← CACT/SLC25A20 BLOCK (inner IMM translocase)</strong></div>
          <div>Step 2 (CACT): Acylcarnitine IN ↔ Free carnitine OUT [antiport through IMM]</div>
          <div className="text-muted">Step 3 (CPT2): Acylcarnitine + CoA-SH → Acyl-CoA + Carnitine [matrix face]</div>
          <div className="text-muted">Steps 4–7: Beta-oxidation (VLCAD → MTP → MTP → MTP for long chain)</div>
          <div className="mt-1" style={{ color: ACCENT3 }}>
            ✅ MCT (C8/C10) BYPASSES CACT — medium-chain FA enter mitochondria via MCT1 directly → THERAPEUTIC
          </div>
          <div style={{ color: ACCENT6 }}>
            ⚠️ C0 LOW (carnitine recycling blocked) → L-Carnitine supplementation ESSENTIAL (unlike CPT1A)
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
      <InfoBox title="📊 NBS Profile Summary (CACT — high C16 + low C0 pattern)" color={ACCENT}>
        <div className="row text-center">
          <div className="col-md-3 mb-2">
            <div className="fw-bold fs-5" style={{ color: ACCENT }}>{nbs.pct_c16_elevated}%</div>
            <div className="text-muted small">C16 ≥1.5 μmol/L (primary flag)</div>
          </div>
          <div className="col-md-3 mb-2">
            <div className="fw-bold fs-5" style={{ color: ACCENT6 }}>{nbs.pct_c0_low}%</div>
            <div className="text-muted small">C0 &lt;15 μmol/L (carnitine depleted)</div>
          </div>
          <div className="col-md-3 mb-2">
            <div className="fw-bold fs-5" style={{ color: ACCENT8 }}>{nbs.pct_cardiomyopathy}%</div>
            <div className="text-muted small">Cardiomyopathy (hallmark)</div>
          </div>
          <div className="col-md-3 mb-2">
            <div className="fw-bold fs-5" style={{ color: ACCENT3 }}>0%</div>
            <div className="text-muted small">C16-OH elevated (NORMAL = key negative vs LCHAD)</div>
          </div>
        </div>
      </InfoBox>

      {/* By severity summary */}
      <InfoBox title="📋 By Severity Group" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#fbe9e7' }}>
              <tr>
                <th>Severity</th><th>N</th><th>Avg C16</th><th>Avg C0</th>
                <th>Avg C18:1</th><th>Avg Glucose</th><th>Cardiac %</th><th>Crisis %</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(bySev).map(([grp, s]) => (
                <tr key={grp}>
                  <td style={{ color: grp.includes('Severe') ? ACCENT4 : ACCENT3 }}>{grp}</td>
                  <td>{s.n}</td>
                  <td className="fw-bold" style={{ color: ACCENT }}>{s.avg_c16}</td>
                  <td style={{ color: ACCENT6 }}>{s.avg_c0}</td>
                  <td>{s.avg_c18_1}</td>
                  <td style={{ color: ACCENT4 }}>{s.avg_glucose}</td>
                  <td style={{ color: ACCENT8 }}>{s.cardiomyopathy_rate}%</td>
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
            style={{ backgroundColor: filter === s ? ACCENT : '#fbe9e7', color: filter === s ? '#fff' : ACCENT }}>
            {s}
          </button>
        ))}
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead style={{ backgroundColor: '#fbe9e7' }}>
            <tr>
              <th>ID</th><th>Severity</th><th>Variant</th><th>Onset (d)</th>
              <th>C16 μmol</th><th>C0 μmol</th><th>C18:1</th>
              <th>Gluc</th><th>Cardiac</th><th>Arrhyth</th><th>Rhabdo</th><th>Treatment</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}>
                <td className="font-monospace text-muted">{p.id}</td>
                <td style={{ color: p.severity.includes('Severe') ? ACCENT4 : ACCENT3 }}>
                  {p.severity.split('(')[0].trim()}
                </td>
                <td className="font-monospace small" style={{ color: ACCENT5, maxWidth: 200, overflow: 'hidden' }}>
                  {p.variant}
                </td>
                <td>{p.onset_age_days}d</td>
                <td className="fw-bold" style={{ color: p.c16 >= 1.5 ? ACCENT : 'inherit' }}>{p.c16}</td>
                <td style={{ color: p.c0 < 15 ? ACCENT6 : 'inherit' }}>{p.c0}</td>
                <td style={{ color: p.c18_1 >= 0.8 ? ACCENT2 : 'inherit' }}>{p.c18_1}</td>
                <td style={{ color: p.glucose < 2.5 ? ACCENT4 : 'inherit' }}>{p.glucose}</td>
                <td style={{ color: p.cardiomyopathy ? ACCENT8 : ACCENT3 }}>{p.cardiomyopathy ? '❤️' : '–'}</td>
                <td style={{ color: p.arrhythmia ? ACCENT2 : 'inherit' }}>{p.arrhythmia ? '⚡' : '–'}</td>
                <td style={{ color: p.rhabdomyolysis ? ACCENT4 : 'inherit' }}>{p.rhabdomyolysis ? '⚠' : '–'}</td>
                <td className="small text-muted">{p.primary_treatment}</td>
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

      {/* MCT + L-carnitine therapeutic banner */}
      <div className="alert" style={{ backgroundColor: '#e8f5e9', borderLeft: `5px solid ${ACCENT3}` }}>
        <h6 className="fw-bold" style={{ color: ACCENT3 }}>✅ MCT OIL + L-CARNITINE — BOTH THERAPEUTIC IN CACT (Level A)</h6>
        <p className="mb-0 small">
          <strong>MCT (C8/C10):</strong> Medium-chain FA bypass the carnitine shuttle — they enter mitochondria
          via MCT1 (monocarboxylate transporter), independent of CPT1/CACT/CPT2. Oxidised by MCAD/SCAD → ketones.{' '}
          <strong>L-Carnitine:</strong> C0 is LOW in CACT (carnitine recycling blocked by translocase); supplementation is ESSENTIAL
          to replenish free carnitine. <em>Contrast with CPT1A where L-carnitine is NOT routine (C0 is already HIGH).</em>
        </p>
      </div>

      {/* Contraindications */}
      <InfoBox title="🚫 Contraindications & High-Risk Treatments" color={ACCENT4}>
        <div className="row">
          <div className="col-md-6">
            <div className="fw-bold small mb-1" style={{ color: ACCENT4 }}>ABSOLUTE CONTRAINDICATIONS</div>
            <ul className="small mb-0">
              <li><strong>Fasting</strong> — ABSOLUTE CI (triggers lipolysis → long-chain FA → cannot enter mitochondria → crisis)</li>
              <li><strong>Ketogenic Diet</strong> — ABSOLUTE CI (requires long-chain fat; CACT blocks transport → worsens disease)</li>
              <li><strong>Valproate (VPA)</strong> — ABSOLUTE CI (inhibits FAO; severe carnitine depletion; especially dangerous with cardiomyopathy)</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-1" style={{ color: ACCENT8 }}>CARDIAC MANAGEMENT</div>
            <ul className="small mb-0">
              <li><strong>Cardiomyopathy:</strong> Digoxin/ACE inhibitors/beta-blockers under cardiology guidance</li>
              <li><strong>Arrhythmia:</strong> Antiarrhythmic monitoring; urgent Holter; ICU for neonatal form</li>
              <li><strong>Triheptanoin (C7):</strong> Level B — anaplerotic odd-chain entry into TCA cycle</li>
            </ul>
          </div>
        </div>
      </InfoBox>

      {/* Variant frequency chart */}
      <InfoBox title="🧬 Variant Distribution (Leading Allele per Patient)" color={ACCENT5}>
        {Object.entries(vc).sort((a,b) => b[1]-a[1]).map(([v, n]) => (
          <PctBar key={v} label={v} pct={Math.round(n / 40 * 100)} color={v.includes('199-10') ? ACCENT5 : ACCENT} />
        ))}
      </InfoBox>

      {/* Carnitine shuttle differential */}
      <InfoBox title="⚖️ Carnitine Shuttle Defects — Differential Diagnosis" color={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#fbe9e7' }}>
              <tr><th>Feature</th><th>CPT1A (Step 1)</th><th style={{ color: ACCENT }}>CACT (Step 2) ↗</th><th>CPT2 (Step 3)</th></tr>
            </thead>
            <tbody>
              <tr><td>Gene</td><td>CPT1A</td><td className="fw-bold" style={{ color: ACCENT }}>SLC25A20</td><td>CPT2</td></tr>
              <tr><td>Step blocked</td><td>Step 1 (outer IMM, rate-limiting)</td><td className="fw-bold" style={{ color: ACCENT }}>Step 2 (translocase)</td><td>Step 3 (inner IMM, matrix)</td></tr>
              <tr><td>C0 (free carnitine)</td>
                <td style={{ color: ACCENT3 }}>HIGH ↑↑ (≥60)</td>
                <td className="fw-bold" style={{ color: ACCENT6 }}>LOW ↓ (&lt;15)</td>
                <td style={{ color: ACCENT6 }}>Low-Normal</td></tr>
              <tr><td>C16 palmitoylcarnitine</td>
                <td style={{ color: ACCENT3 }}>Normal/Low</td>
                <td className="fw-bold" style={{ color: ACCENT }}>HIGH ↑↑</td>
                <td style={{ color: ACCENT }}>HIGH ↑↑</td></tr>
              <tr><td>Cardiomyopathy</td>
                <td className="fw-bold" style={{ color: ACCENT3 }}>NONE (CPT1B intact)</td>
                <td className="fw-bold" style={{ color: ACCENT8 }}>YES — HALLMARK</td>
                <td style={{ color: ACCENT8 }}>YES (severe form)</td></tr>
              <tr><td>Rhabdomyolysis</td>
                <td className="fw-bold" style={{ color: ACCENT3 }}>NONE (liver isoform)</td>
                <td style={{ color: ACCENT4 }}>Yes (all forms)</td>
                <td style={{ color: ACCENT4 }}>Yes (hallmark — mild/adult)</td></tr>
              <tr><td>Hyperammonemia</td>
                <td style={{ color: ACCENT6 }}>YES (UNIQUE in FAO)</td>
                <td>Possible (moderate)</td><td>Rare</td></tr>
              <tr><td>L-Carnitine use</td>
                <td>NOT routine (C0 HIGH)</td>
                <td className="fw-bold" style={{ color: ACCENT3 }}>ESSENTIAL (C0 LOW)</td>
                <td style={{ color: ACCENT3 }}>Yes (C0 low)</td></tr>
              <tr><td>Primary phenotype</td><td>Hepatic / hypoketotic</td><td className="fw-bold" style={{ color: ACCENT }}>Neonatal severe cardiac</td><td>Rhabdomyolysis / adult</td></tr>
            </tbody>
          </table>
        </div>
      </InfoBox>

      {/* CACT vs LCHAD */}
      <InfoBox title="🔬 CACT vs LCHAD — Key Distinctions (both have C16 elevated + cardiac)" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#fbe9e7' }}>
              <tr><th>Feature</th><th style={{ color: ACCENT }}>CACT (SLC25A20)</th><th>LCHAD (HADHA)</th></tr>
            </thead>
            <tbody>
              <tr><td>C16 palmitoylcarnitine</td><td style={{ color: ACCENT }}>HIGH ↑↑</td><td style={{ color: ACCENT }}>HIGH ↑↑ (similar)</td></tr>
              <tr><td>C16-OH (3-hydroxy)</td><td className="fw-bold" style={{ color: ACCENT3 }}>NORMAL — KEY NEGATIVE</td><td className="fw-bold" style={{ color: ACCENT4 }}>HIGH ↑↑ — PRIMARY MARKER</td></tr>
              <tr><td>Retinal degeneration</td><td className="fw-bold" style={{ color: ACCENT3 }}>ABSENT</td><td style={{ color: ACCENT4 }}>YES — progressive RPE degeneration</td></tr>
              <tr><td>Peripheral neuropathy</td><td className="fw-bold" style={{ color: ACCENT3 }}>ABSENT</td><td style={{ color: ACCENT4 }}>YES — axonal neuropathy</td></tr>
              <tr><td>Maternal AFLP/HELLP</td><td className="fw-bold" style={{ color: ACCENT3 }}>ABSENT</td><td style={{ color: ACCENT4 }}>YES — maternal complication</td></tr>
              <tr><td>DHA supplementation</td><td>NOT indicated</td><td style={{ color: ACCENT3 }}>YES — retinal/neural protection</td></tr>
              <tr><td>Cardiomyopathy</td><td style={{ color: ACCENT8 }}>YES (dilated/hypertrophic)</td><td style={{ color: ACCENT8 }}>YES (dilated)</td></tr>
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
                <td style={{
                  color: k.includes('C16') && !k.includes('OH') ? ACCENT
                       : k.includes('C18') ? ACCENT2
                       : k.includes('C0') ? ACCENT6
                       : k.includes('C16_OH') ? ACCENT3
                       : k.includes('C8') ? ACCENT3
                       : 'inherit'
                }}>
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
                <td style={{
                  color: k.includes('Cardiomyopathy') ? ACCENT8
                       : k.includes('Arrhythmia') ? ACCENT2
                       : k.includes('NO_') ? ACCENT3
                       : 'inherit'
                }}>{v}</td>
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
                <td className="fw-bold" style={{ width: 200, color: k.includes('CI') || k.includes('ABSOLUTE') ? ACCENT4 : k.includes('THERAPEUTIC') || k.includes('ESSENTIAL') ? ACCENT3 : k.includes('HIGH_RISK') ? ACCENT6 : ACCENT }}>
                  {k.replace(/_/g, ' ')}
                </td>
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

      <InfoBox title="🧬 Genetics — Key Variants" color={ACCENT5}>
        <table className="table table-sm small mb-0">
          <tbody>
            {Object.entries((data.genetics || {}).key_variants || {}).map(([v, desc]) => (
              <tr key={v}>
                <td className="fw-bold font-monospace text-muted" style={{ width: 240 }}>{v}</td>
                <td>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {(data.genetics || {}).population_note && (
          <p className="small mt-2 mb-0 text-muted">{data.genetics.population_note}</p>
        )}
      </InfoBox>

      <InfoBox title="⚖️ Key Distinguishing Facts" color={ACCENT2}>
        <ul className="small mb-0">
          {(data.key_distinguishing_facts || []).map((f, i) => <li key={i}>{f}</li>)}
        </ul>
      </InfoBox>

      <InfoBox title="📐 Carnitine Shuttle Context" color={ACCENT7}>
        <pre className="small mb-0" style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
          {data.carnitine_shuttle_context}
        </pre>
      </InfoBox>

      <InfoBox title="⚖️ Comparison Table" color={ACCENT2}>
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
export default function CACTPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview]  = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]          = useState(null);
  const [error, setError]        = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cact/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Backend offline'));
    fetch(`${API}/api/cact/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/cact/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 d-flex align-items-center gap-2 flex-wrap">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Home</Link>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          🟠 CACT Epilepsy Dashboard
        </h4>
        <span className="badge" style={{ backgroundColor: ACCENT }}>CACT / SLC25A20 Deficiency</span>
        <span className="badge" style={{ backgroundColor: ACCENT5 }}>3p21.31</span>
        <span className="badge" style={{ backgroundColor: ACCENT7 }}>AR</span>
        <span className="badge" style={{ backgroundColor: ACCENT8 }}>Cardiomyopathy HALLMARK</span>
        <span className="badge" style={{ backgroundColor: ACCENT6 }}>C0 LOW</span>
        <span className="badge" style={{ backgroundColor: ACCENT }}>C16 HIGH</span>
        <span className="badge" style={{ backgroundColor: ACCENT3 }}>C16-OH Normal (vs LCHAD)</span>
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
