'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'DYNC2H1 Dynein-2 & Retrograde IFT', 'Definitions'];

// SRTD3 colour scheme — DYNC2H1 / retrograde IFT / Jeune / narrow thorax
const ACCENT  = '#1a237e';   // deep navy — DYNC2H1 retrograde IFT motor; primary mechanism
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory failure; severity
const ACCENT3 = '#1b5e20';   // deep green — renal TIN; secondary renal disease; ESRD
const ACCENT4 = '#4a148c';   // deep purple — retinal dystrophy; rod-cone; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic; ductal plate malformation
const ACCENT6 = '#37474f';   // dark slate — molecular architecture; AAA+ ring; dynein-2 structure
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; VEPTR surgery; diagnostic EM
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; EVC differential

const SEED = 381;

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

function Alert({ color, children }) {
  return (
    <div className="alert mb-2" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return <span className="badge me-1" style={{ background: color, fontSize: '0.72em' }}>{text}</span>;
}

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 8, borderRadius: 4 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color, borderRadius: 4 }} />
      </div>
    </div>
  );
}

export default function SRTD3Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/srtd3/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd3/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading SRTD3 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; DYNC2H1 Short-Rib Thoracic Dysplasia 3 (SRTD3) — Jeune ATD3 · Retrograde IFT Motor
            </h4>
            <div className="small text-muted mb-1">
              <strong>DYNC2H1</strong> · 11q22.3 · 4,307 aa · AAA+ ATPase ring + stalk + MTBD ·
              cytoplasmic dynein-2 retrograde IFT motor · MOST COMMON SRTD gene (~50%) ·
              narrow thorax (primary) · polydactyly ~{ov.pct_polydactyly}% ·
              renal secondary · retinal ~{ov.pct_retinal_dystrophy}% · CHF ~{ov.pct_hepatic_chf}% ·
              NO situs inversus
            </div>
            <div className="small">
              <Badge text="OMIM *603297" color={ACCENT} />
              <Badge text="#613091 SRTD3/ATD3" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="11q22.3" color={ACCENT6} />
              <Badge text="Retrograde IFT motor" color={ACCENT} />
              <Badge text="Most common SRTD (~50%)" color={ACCENT2} />
              <Badge text={`Polydactyly ~${ov.pct_polydactyly}%`} color={ACCENT8} />
              <Badge text={`Retinal ~${ov.pct_retinal_dystrophy}%`} color={ACCENT4} />
              <Badge text={`CHF ~${ov.pct_hepatic_chf}%`} color={ACCENT5} />
              <Badge text="NO situs inversus" color={ACCENT3} />
              <Badge text="VEPTR surgical Rx" color={ACCENT7} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              Neonatal vent {ov.pct_neonatal_ventilator}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT8, fontSize: '0.8em' }}>
              Polydactyly {ov.pct_polydactyly}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              Any renal {ov.pct_any_renal}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT4, fontSize: '0.8em' }}>
              Retinal {ov.pct_retinal_dystrophy}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              CHF {ov.pct_hepatic_chf}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT}>
        <strong>&#x1f3cb; DYNC2H1 IS THE MOST COMMON SRTD GENE (~50% OF MOLECULARLY CONFIRMED SRTD) — FIRST GENE TO TEST IN SKELETAL CILIOPATHY:</strong> DYNC2H1
        encodes the 4,307 aa AAA+ ATPase heavy chain of cytoplasmic dynein-2, the <em>retrograde intraflagellar
        transport (IFT) motor</em>. Dynein-2 powers retrograde movement of IFT trains from the ciliary
        tip back to the cell body. Loss of DYNC2H1 → retrograde IFT failure → IFT-B subunits accumulate
        at ciliary tips (&quot;bulging/club cilia tip&quot; on EM) → Hedgehog signalling impaired in chondrocytes
        → severely short ribs + narrow thorax + short limbs ± polydactyly. DYNC2H1 is ALWAYS the
        first gene on any skeletal ciliopathy panel.
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x1f9b4; NARROW THORAX IS THE PRIMARY PATHOGNOMONIC FEATURE — NOT A SECONDARY COMPLICATION:</strong> The
        narrow, horizontally-oriented short-rib thorax is the defining, earliest, and most life-threatening
        feature of SRTD3 — in contrast to NPHP1–20 where renal TIN is primary and skeletal involvement
        is absent or minor. Neonatal respiratory failure from thoracic restriction is the primary lethal
        mechanism in SRPS3 (biallelic null alleles). Severe thorax: {ov.pct_neonatal_ventilator}% of this cohort
        required neonatal ventilator support. VEPTR/MAGEC thoracic expansion surgery is the primary
        mechanical treatment. Thoracic circumference assessment at every clinical visit.
      </Alert>
      <Alert color={ACCENT8}>
        <strong>&#x270b; POLYDACTYLY ~{ov.pct_polydactyly}% — POSTAXIAL MOST COMMON — DISTINGUISH FROM EVC (CHD) AND BBS (OBESITY):</strong> Postaxial
        polydactyly (extra digit, ulnar/fibular side) in ~55% of SRTD3 — a key clinical clue at
        birth pointing toward skeletal ciliopathy. Key differential: Ellis-van Creveld (EVC) has
        congenital heart defect (CHD 50–60%, ASD/AVSD) — ABSENT in SRTD3. Bardet-Biedl (BBS) has
        truncal obesity + hypogonadism + no narrow thorax. Echo mandatory in all narrow-thorax +
        polydactyly neonates. SRTD3: no CHD, no obesity, no ectodermal features.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1f6ab; NO SITUS INVERSUS — PRIMARY CILIA (9+0) NOT NODAL MOTILE CILIA (9+2) — DYNC2H1 ≠ PCD DYNEIN:</strong> DYNC2H1
        is the retrograde IFT motor of non-motile PRIMARY CILIA (9+0 axoneme). Nodal cilia (which
        determine left-right lateralisation) are MOTILE and use outer/inner dynein arm complexes
        (DNAH5, DNAH11, DNAI1 — PCD genes). DYNC2H1 does NOT affect nodal cilia → zero situs
        inversus in SRTD3. If situs inversus present with skeletal dysplasia, consider PCD
        co-occurrence. The word &quot;dynein&quot; is shared by PCD and SRTD3 — the mechanisms are entirely distinct.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; RENAL + RETINAL + HEPATIC ARE SECONDARY FEATURES IN SRTD3 SURVIVORS — MANDATORY SURVEILLANCE:</strong> Any
        renal involvement: {ov.pct_any_renal}% of cohort. ESRD/transplant: {ov.pct_esrd_or_transplant}%. Retinal dystrophy:
        {ov.pct_retinal_dystrophy}%. CHF: {ov.pct_hepatic_chf}%. Annual renal USS + GFR from age 5. Annual ERG + fundoscopy from
        diagnosis. Annual liver USS + APRI index. Renal transplant CURATIVE (donor kidney has
        functional DYNC2H1 → no TIN recurrence). Retinal + hepatic are cell-autonomous — NOT
        corrected by transplant.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Cohort N" value={ov.cohort_n} color={ACCENT} />
            <KPI label="Neonatal ventilator" value={`${ov.pct_neonatal_ventilator}%`} color={ACCENT2} />
            <KPI label="Polydactyly" value={`${ov.pct_polydactyly}%`} color={ACCENT8} />
            <KPI label="Any renal" value={`${ov.pct_any_renal}%`} color={ACCENT3} />
            <KPI label="ESRD/Tx" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT3} />
            <KPI label="Retinal" value={`${ov.pct_retinal_dystrophy}%`} color={ACCENT4} />
            <KPI label="CHF (hepatic)" value={`${ov.pct_hepatic_chf}%`} color={ACCENT5} />
            <KPI label="VEPTR surgery" value={`${ov.pct_veptr_surgery}%`} color={ACCENT7} />
            <KPI label="Biallelic null" value={`${ov.pct_biallelic_null}%`} color={ACCENT2} />
            <KPI label="Deceased" value={`${ov.pct_deceased}%`} color={ACCENT6} />
            <KPI label="Median GFR" value={`${ov.median_gfr} ml/min`} color={ACCENT3} />
            <KPI label="Median age Dx" value={`${ov.median_age_dx_yr}yr`} color={ACCENT} />
          </div>

          <Section title="SRTD3 Mechanism Overview — Retrograde IFT Failure" color={ACCENT}>
            <div className="row g-3">
              <div className="col-md-6">
                <div className="p-3 rounded" style={{ background: ACCENT + '08', border: `1px solid ${ACCENT}30` }}>
                  <div className="fw-bold small mb-2" style={{ color: ACCENT }}>&#x2699; Dynein-2 Retrograde IFT Motor</div>
                  <div className="small">
                    <div>&#x2713; DYNC2H1 (4,307 aa) — heavy chain; AAA+ ATPase ring (6 modules)</div>
                    <div>&#x2713; Powers retrograde IFT: ciliary tip → cell body</div>
                    <div>&#x2713; Without DYNC2H1: IFT-B subunits pile up at tip = &quot;bulging cilia&quot;</div>
                    <div>&#x2713; Hedgehog pathway impaired: PTCH1/SMO/Gli3 trafficking fails</div>
                    <div>&#x2713; Chondrocytes: Ihh/Shh fails → short ribs, narrow thorax, short limbs</div>
                    <div>&#x2713; Renal tubular cells: Wnt/mTOR/Notch fails → TIN (secondary)</div>
                    <div>&#x2713; Distinct from DNAH5/PCD (motile cilia dynein) — different molecule, different axoneme</div>
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="p-3 rounded" style={{ background: ACCENT2 + '08', border: `1px solid ${ACCENT2}30` }}>
                  <div className="fw-bold small mb-2" style={{ color: ACCENT2 }}>&#x1f9b4; Clinical Hierarchy (Primary → Secondary)</div>
                  <div className="small">
                    <div><strong>PRIMARY (congenital, pathognomonic):</strong></div>
                    <div>1. Narrow thorax + short horizontal ribs → respiratory failure</div>
                    <div>2. Short limbs (rhizomelia/mesomelia)</div>
                    <div>3. Polydactyly (postaxial ~38%, preaxial ~10%)</div>
                    <br />
                    <div><strong>SECONDARY (develops in survivors):</strong></div>
                    <div>4. Renal TIN → ESRD (childhood–adolescence)</div>
                    <div>5. Retinal rod-cone dystrophy (~18%)</div>
                    <div>6. Congenital hepatic fibrosis/CHF (~14%)</div>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Cohort Sample (8 of 40 patients, seed 381)" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead>
                  <tr>
                    <th>ID</th><th>Ethnicity</th><th>Thorax</th><th>Polydactyly</th>
                    <th>Renal</th><th>Retinal</th><th>Allele class</th><th>Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.patients || []).map(p => (
                    <tr key={p.id}>
                      <td><code>{p.id}</code></td>
                      <td className="small">{p.ethnicity?.split('(')[0].trim()}</td>
                      <td className="small">{p.thorax_severity?.split('—')[0].trim().slice(0, 30)}</td>
                      <td className="small">{p.polydactyly?.split('(')[0].trim()}</td>
                      <td className="small">{p.renal_status?.split('—')[0].trim().slice(0, 25)}</td>
                      <td className="small">{p.retinal_status?.split('(')[0].trim()}</td>
                      <td className="small">{p.allele_class?.split('(')[0].trim().slice(0, 30)}</td>
                      <td className="small">{p.outcome?.split('—')[0].trim().slice(0, 25)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Dynein-2 Complex Subunits" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead style={{ background: ACCENT6 + '15' }}>
                  <tr>
                    <th>Subunit</th><th>Gene</th><th>Chromosome</th><th>Role</th><th>SRTD Disease</th>
                  </tr>
                </thead>
                <tbody className="small">
                  <tr style={{ background: ACCENT + '10' }}>
                    <td><strong>Heavy chain ★</strong></td><td>DYNC2H1</td><td>11q22.3</td>
                    <td>AAA+ ATPase motor; retrograde force generation</td>
                    <td><strong>SRTD3</strong> (MOST COMMON ~50%)</td>
                  </tr>
                  <tr>
                    <td>Light int. chain</td><td>DYNC2LI1</td><td>2p21</td>
                    <td>Complex assembly; links heavy chain to adaptors</td>
                    <td>SRTD15</td>
                  </tr>
                  <tr>
                    <td>Int. chain 1</td><td>WDR34</td><td>9q34.11</td>
                    <td>Bridges heavy chain to light chains; IFT-A contact</td>
                    <td>SRTD11</td>
                  </tr>
                  <tr>
                    <td>Int. chain 2</td><td>WDR60</td><td>7q36.3</td>
                    <td>IFT-A adaptor; DYNC2H1 complex stabiliser</td>
                    <td>SRTD8 (2nd most common dynein-2 SRTD)</td>
                  </tr>
                  <tr>
                    <td>Light chain 1</td><td>TCTEX1D2</td><td>3q22.1</td>
                    <td>Cargo adaptor; Tctex-1 family domain</td>
                    <td>SRTD17</td>
                  </tr>
                  <tr>
                    <td>Light chain 2</td><td>DYNLRB1/DYNLRB2</td><td>20q13/16q24</td>
                    <td>Roadblock family; complex integrity</td>
                    <td>Rare SRTD</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <Section title="Thorax Severity Distribution" color={ACCENT2}>
                {Object.entries(bk.thorax_severity_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={SEED > 0 ? 40 : 1} color={ACCENT2} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Polydactyly Distribution" color={ACCENT8}>
                {Object.entries(bk.polydactyly_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT8} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Renal Status Distribution" color={ACCENT3}>
                {Object.entries(bk.renal_status_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT3} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="CKD Stage Tiers" color={ACCENT3}>
                {Object.entries(bk.ckd_stage_tiers || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT3} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Retinal Status Distribution" color={ACCENT4}>
                {Object.entries(bk.retinal_status_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT4} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Hepatic / CHF Status" color={ACCENT5}>
                {Object.entries(bk.hepatic_status_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT5} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Allele Class Distribution" color={ACCENT6}>
                {Object.entries(bk.allele_class_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT6} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="VEPTR Surgery Distribution" color={ACCENT7}>
                {Object.entries(bk.veptr_surgery_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT7} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="First Presentation" color={ACCENT}>
                {Object.entries(bk.first_presentation_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Prior Misdiagnosis Distribution" color={ACCENT7}>
                {Object.entries(bk.prior_misdiagnosis_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT7} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Outcome Distribution" color={ACCENT2}>
                {Object.entries(bk.outcome_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT2} />
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="Ethnicity Distribution" color={ACCENT6}>
                {Object.entries(bk.ethnicity_distribution || {}).map(([k, v]) => (
                  <Bar key={k} label={k} value={v} max={40} color={ACCENT6} />
                ))}
              </Section>
            </div>
          </div>

          <Section title="SRTD3 vs Other Skeletal Ciliopathies — Comparison Table" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>Feature</th>
                    <th>SRTD3 / DYNC2H1</th>
                    <th>SRTD4/ATD4 / TTC21B</th>
                    <th>SRTD5/CED1 / WDR19</th>
                    <th>EVC / EVC1-EVC2</th>
                    <th>BBS1 / Bardet-Biedl</th>
                  </tr>
                </thead>
                <tbody className="small">
                  <tr>
                    <td>Mechanism</td>
                    <td><strong>Retrograde IFT motor (dynein-2)</strong></td>
                    <td>IFT-A retrograde adaptor</td>
                    <td>IFT-A largest subunit</td>
                    <td>Hh pathway effector (EVC zone)</td>
                    <td>BBSome IFT cargo adaptor</td>
                  </tr>
                  <tr>
                    <td>Narrow thorax</td>
                    <td style={{ background: '#c8e6c9' }}><strong>YES — pathognomonic (primary)</strong></td>
                    <td>~7–10% (null only)</td>
                    <td>~8% (null only)</td>
                    <td>YES</td>
                    <td>NO</td>
                  </tr>
                  <tr>
                    <td>Polydactyly</td>
                    <td><strong>~55%</strong></td>
                    <td>~12%</td>
                    <td>~15%</td>
                    <td>YES (&gt;90%)</td>
                    <td>YES (~70%) postaxial</td>
                  </tr>
                  <tr>
                    <td>CHD (heart)</td>
                    <td style={{ background: '#c8e6c9' }}><strong>NO</strong></td>
                    <td>NO</td>
                    <td>NO</td>
                    <td style={{ background: '#ffcdd2' }}><strong>YES 50–60% (ASD/AVSD)</strong></td>
                    <td>Rare (&lt;5%)</td>
                  </tr>
                  <tr>
                    <td>Ectodermal</td>
                    <td>NO</td>
                    <td>NO</td>
                    <td style={{ background: '#fff9c4' }}><strong>YES (hypotrichosis, hyponychia)</strong></td>
                    <td>YES (hypodontia, nail)</td>
                    <td>NO</td>
                  </tr>
                  <tr>
                    <td>Renal TIN</td>
                    <td>40–60% (secondary, survivors)</td>
                    <td>~90% (primary)</td>
                    <td>~80% (primary)</td>
                    <td>~25% cysts</td>
                    <td>~25% cysts</td>
                  </tr>
                  <tr>
                    <td>Retinal</td>
                    <td>~18%</td>
                    <td>~0% (TTC21B not photoreceptors)</td>
                    <td>~25%</td>
                    <td>NO</td>
                    <td>~99% rod-cone</td>
                  </tr>
                  <tr>
                    <td>Obesity</td>
                    <td>NO</td>
                    <td>NO</td>
                    <td>NO</td>
                    <td>NO</td>
                    <td>YES (universal by age 5)</td>
                  </tr>
                  <tr>
                    <td>Situs inversus</td>
                    <td>NO</td>
                    <td>NO</td>
                    <td>NO</td>
                    <td>NO</td>
                    <td>NO</td>
                  </tr>
                  <tr>
                    <td>Gene prevalence in SRTD</td>
                    <td style={{ background: '#c8e6c9' }}><strong>~50% (most common)</strong></td>
                    <td>~5%</td>
                    <td>~5%</td>
                    <td>n/a (not SRTD)</td>
                    <td>n/a (not SRTD)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 2: DYNC2H1 Architecture & Retrograde IFT Biology ── */}
      {tab === 2 && df && (
        <div>
          <Section title="DYNC2H1 Protein Architecture — 4,307 aa AAA+ Motor" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr>
                    <th>Domain</th><th>Residues</th><th>Function</th><th>Allele-Phenotype Correlation</th>
                  </tr>
                </thead>
                <tbody className="small">
                  <tr>
                    <td><strong>N-terminal stem/tail</strong></td><td>~1–1,300</td>
                    <td>Complex assembly scaffold; binds DYNC2LI1, WDR34, WDR60, TCTEX1D2, DYNLRB1</td>
                    <td>Missense → moderate SRTD3 (partial complex; reduced retrograde speed)</td>
                  </tr>
                  <tr>
                    <td><strong>AAA1 (ATP hydrolysis)</strong></td><td>~1,300–1,700</td>
                    <td>Principal ATP hydrolysis site; main power stroke; nucleotide-binding P-loop</td>
                    <td>Truncating → SRPS3 or severe SRTD3; missense → moderate</td>
                  </tr>
                  <tr>
                    <td><strong>AAA2–AAA4</strong></td><td>~1,700–2,800</td>
                    <td>AAA3/AAA4: mechanical coupling between ring and stalk; regulate power output</td>
                    <td>Null variants → complete retrograde failure; missense → partial activity</td>
                  </tr>
                  <tr>
                    <td><strong>Stalk (coiled-coil)</strong></td><td>~2,800–3,200</td>
                    <td>Contacts IFT-A complex at ciliary tip; provides directionality; dynein stepping</td>
                    <td>Stalk missense → moderate SRTD3; stalk truncating → severe</td>
                  </tr>
                  <tr>
                    <td><strong>AAA5–AAA6</strong></td><td>~3,200–3,800</td>
                    <td>Buttress; stabilises stalk-MTBD geometry; coordinates AAA+ ring conformations</td>
                    <td>Hypomorphic C-terminal missense → mildest SRTD3 (partial motor activity)</td>
                  </tr>
                  <tr>
                    <td><strong>MTBD (microtubule-binding)</strong></td><td>~3,800–4,307</td>
                    <td>Anchors dynein-2 to B-tubule axoneme; retrograde stepping track</td>
                    <td>C-terminal hypomorphic → near-normal thorax; renal-predominant adult phenotype</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Key DYNC2H1 Disease-Causing Variants" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-hover">
                <thead style={{ background: ACCENT6 + '15' }}>
                  <tr><th>Variant</th><th>Class</th></tr>
                </thead>
                <tbody className="small">
                  {(df.genetic_architecture?.key_variants || []).map((v, i) => (
                    <tr key={i}>
                      <td><code>{v.split('—')[0].trim()}</code></td>
                      <td>{v.split('—').slice(1).join('—').trim()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Retrograde IFT Pathway — Hedgehog Signalling Connection" color={ACCENT}>
            <div className="row g-3">
              <div className="col-md-6">
                <div className="p-3 rounded" style={{ background: ACCENT + '08', border: `1px solid ${ACCENT}30` }}>
                  <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Normal Retrograde IFT (DYNC2H1 intact)</div>
                  <div className="small">
                    <div>1. Kinesin-2 (KIF3A/KIF3B/KAP) → anterograde IFT trains (base → tip)</div>
                    <div>2. IFT-B cargoes delivered to ciliary tip</div>
                    <div>3. DYNC2H1 dynein-2 powers IFT-A train return (tip → base)</div>
                    <div>4. PTCH1 cleared from tip → SMO activated → ciliary compartment</div>
                    <div>5. Gli3 full-length processed to Gli3A (activator) → Hh targets ON</div>
                    <div>6. Chondrocytes: Ihh → growth plate organised → normal ribs + limbs</div>
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="p-3 rounded" style={{ background: ACCENT2 + '08', border: `1px solid ${ACCENT2}30` }}>
                  <div className="fw-bold small mb-2" style={{ color: ACCENT2 }}>DYNC2H1 Loss → Retrograde Failure</div>
                  <div className="small">
                    <div>1. Anterograde still runs (kinesin-2 intact)</div>
                    <div>2. IFT-B subunits ACCUMULATE at tip → &quot;bulging/club cilia&quot; on EM</div>
                    <div>3. Retrograde halted — PTCH1, SMO, Gli3 trapped in cilia</div>
                    <div>4. Gli3 full-length not processed → Hh targets suppressed</div>
                    <div>5. Chondrocytes: Ihh signalling fails → severely short ribs + narrow thorax</div>
                    <div>6. Renal/retinal/hepatic: downstream cilia dysfunction (secondary)</div>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          <Section title="Differential Diagnosis Table" color={ACCENT7}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead style={{ background: ACCENT7 + '15' }}>
                  <tr><th>DDx Pair</th><th>Key Distinction</th></tr>
                </thead>
                <tbody className="small">
                  {Object.entries(df.ddx_table || {}).map(([pair, distinction]) => (
                    <tr key={pair}>
                      <td><strong>{pair}</strong></td>
                      <td>{distinction}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="SRTD Family Comparison" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-bordered">
                <thead style={{ background: ACCENT + '15' }}>
                  <tr><th>Disease / Gene</th><th>Key Features</th></tr>
                </thead>
                <tbody className="small">
                  {Object.entries(df.srtd_comparison || {}).map(([disease, features]) => (
                    <tr key={disease} style={disease.startsWith('★') ? { background: ACCENT + '10' } : {}}>
                      <td><strong>{disease}</strong></td>
                      <td>{features}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="Treatment Summary" color={ACCENT3}>
            {Object.entries(df.treatment || {}).map(([title, text]) => (
              <div key={title} className="mb-3 p-3 rounded" style={{ background: ACCENT3 + '08', border: `1px solid ${ACCENT3}30` }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT3 }}>{title}</div>
                <div className="small">{text}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && df && (
        <div>
          <Section title="Disease Definition" color={ACCENT}>
            <div className="p-3 rounded" style={{ background: ACCENT + '08', border: `1px solid ${ACCENT}30` }}>
              <div className="small">{df.disease}</div>
            </div>
          </Section>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="table-responsive">
                <table className="table table-sm table-bordered">
                  <tbody className="small">
                    <tr><td><strong>OMIM Gene</strong></td><td>{df.omim_gene}</td></tr>
                    <tr><td><strong>OMIM Disease</strong></td><td>{df.omim_disease}</td></tr>
                    <tr><td><strong>Chromosome</strong></td><td>{df.chromosome}</td></tr>
                    <tr><td><strong>Inheritance</strong></td><td>{df.inheritance}</td></tr>
                    <tr><td><strong>Prevalence</strong></td><td>{df.prevalence}</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <Section title="Molecular Mechanism" color={ACCENT}>
            <div className="small p-3 rounded" style={{ background: ACCENT + '05', border: `1px solid ${ACCENT}20` }}>
              {df.mechanism}
            </div>
          </Section>
          <Section title="Key Clinical Features" color={ACCENT2}>
            {Object.entries(df.key_clinical_features || {}).map(([feat, text]) => (
              <div key={feat} className="mb-3 p-3 rounded" style={{ background: ACCENT2 + '08', border: `1px solid ${ACCENT2}30` }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>{feat}</div>
                <div className="small">{text}</div>
              </div>
            ))}
          </Section>
          <Section title="Genetic Architecture" color={ACCENT6}>
            {Object.entries(df.genetic_architecture || {}).filter(([k]) => k !== 'key_variants').map(([k, v]) => (
              <div key={k} className="mb-2">
                <span className="fw-bold small" style={{ color: ACCENT6 }}>{k}: </span>
                <span className="small">{typeof v === 'object' ? JSON.stringify(v) : v}</span>
              </div>
            ))}
          </Section>
          <Section title="Diagnostic Criteria" color={ACCENT}>
            {Object.entries(df.diagnostic_criteria || {}).map(([criterion, text]) => (
              <div key={criterion} className="mb-3 p-3 rounded" style={{ background: ACCENT + '05', border: `1px solid ${ACCENT}20` }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{criterion}</div>
                <div className="small">{text}</div>
              </div>
            ))}
          </Section>
          <Section title="Prognosis" color={ACCENT3}>
            <div className="small p-3 rounded" style={{ background: ACCENT3 + '08', border: `1px solid ${ACCENT3}30` }}>
              {df.prognosis}
            </div>
          </Section>
          <Section title="Cohort Note" color={ACCENT6}>
            <div className="small text-muted">{df.cohort_note}</div>
          </Section>
        </div>
      )}
    </div>
  );
}
