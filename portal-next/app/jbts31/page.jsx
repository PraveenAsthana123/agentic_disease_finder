'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Centriole Elongation Pearls', 'Definitions'];

// JBTS31 colour scheme — CEP120 / Centriole Elongation / Daughter Centriole / Hypomorphic Alleles
// Deep violet for centriole scaffold; slate-blue for CPAP-binding; amber for short cilia
const ACCENT   = '#4527a0';   // deep violet — CEP120 centriole scaffold identity
const ACCENT2  = '#1565c0';   // royal blue — CPAP/CENPJ binding axis
const ACCENT3  = '#e65100';   // deep orange-amber — short cilia / basal body failure
const ACCENT4  = '#2e7d32';   // forest green — no MKS tier / all liveborn / no thorax
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#c62828';   // crimson — JBTS31 vs SRTD19 alert / allele class
const ACCENT7  = '#00695c';   // dark teal — renal NPHP-like
const ACCENT8  = '#6a1b9a';   // purple — retinal rod-cone
const ACCENT9  = '#1b5e20';   // dark green — cerebellar ataxia
const ACCENT10 = '#795548';   // brown — intellectual disability / oculomotor

const SEED = 479;
const N_COHORT = 40;

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
    <div className="alert mb-3" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
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

function DataTable({ headers, rows, accent }) {
  return (
    <div className="table-responsive mb-3">
      <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
        <thead style={{ background: accent + '22' }}>
          <tr>{headers.map(h => <th key={h} style={{ color: accent }}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function JBTS31Page() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/jbts31/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts31/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts31/definitions`).then(r => r.json()),
    ];
    Promise.all(endpoints)
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const kpis = overview?.key_kpis || {};
  const prevPct = breakdown?.phenotype_prevalence || {};

  return (
    <div style={{ fontFamily: 'Inter, system-ui, sans-serif', background: '#f8f9fa', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT2} 100%)`, color: '#fff', padding: '18px 24px 14px' }}>
        <div className="d-flex align-items-center gap-2 mb-1">
          <Link href="/" style={{ color: '#ffffffb0', fontSize: 12 }}>← Home</Link>
          <span style={{ color: '#ffffff60' }}>·</span>
          <span style={{ fontSize: 12, color: '#ffffffb0' }}>Expert Dashboards</span>
        </div>
        <h4 className="fw-bold mb-0" style={{ color: '#fff', fontSize: 18 }}>
          🧬 CEP120 Joubert Syndrome Type 31 (JBTS31)
        </h4>
        <div style={{ fontSize: 12, color: '#ffffffc0', marginTop: 3 }}>
          CEP120 · Daughter Centriole Elongation Scaffold · Hypomorphic Alleles · No Thorax · No MKS Tier ·
          OMIM Gene <strong style={{color:'#fff'}}>*613446</strong> · Disease <strong style={{color:'#fff'}}>#617761</strong> ·
          5q23.2 · <em>Allelic: SRTD19 (moderate alleles) · NPHP20 (renal-dominant) · SRPS2B (biallelic null)</em>
        </div>
        <div style={{ fontSize: 11, color: '#ffffffa0', marginTop: 2 }}>
          40-patient cohort · seed-{SEED} · 3 endpoints /api/jbts31/overview|breakdown|definitions
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', borderBottom: '1px solid #dee2e6', padding: '0 24px' }}>
        <div className="d-flex gap-0">
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{
                border: 'none', background: 'none', padding: '10px 16px', fontSize: 13,
                cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
                color: tab === t ? ACCENT : '#495057',
                borderBottom: tab === t ? `2px solid ${ACCENT}` : '2px solid transparent',
              }}>
              {t}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: '20px 24px' }}>
        {loading && <div className="text-center py-4 text-muted">Loading JBTS31 data…</div>}
        {error && <div className="alert alert-danger">API error: {error}</div>}

        {/* ── OVERVIEW TAB ────────────────────────────────────────────── */}
        {tab === 'Overview' && overview && (
          <div>
            {/* Critical allele-class banner */}
            <Alert color={ACCENT6}>
              <strong>⚠ Allele Class — JBTS31 = Hypomorphic CEP120 only</strong><br />
              <span style={{ fontSize: 12 }}>
                JBTS31 requires <em>very hypomorphic</em> biallelic CEP120 alleles (≥30% residual function).
                Moderate alleles → <strong>SRTD19</strong> (narrow thorax, short ribs).
                Biallelic null → <strong>SRPS2B</strong> (perinatal lethal).
                The chest X-ray is the first discriminator: <strong>normal CXR = JBTS31</strong>.
              </span>
            </Alert>

            {/* JBTS31 vs SRTD19 vs NPHP20 panel */}
            <Alert color={ACCENT}>
              <strong>JBTS31 / NPHP20 / SRTD19 — same gene CEP120, different allele classes & phenotypes</strong><br />
              <span style={{ fontSize: 12 }}>
                <strong>JBTS31</strong>: mild missense → MTS + ataxia; NO thorax finding; CEP120 residual ~30–40% →
                short cilia (50–70% WT) → cerebellar Shh failure → MTS (renal 30%, retinal 20%).<br />
                <strong>NPHP20</strong>: same OMIM #617761; renal-dominant; ESRD adolescent-adult; MTS in ~55%.<br />
                <strong>SRTD19</strong>: moderate alleles; narrow thorax + short ribs + polydactyly; VEPTR intervention.
              </span>
            </Alert>

            <Alert color={ACCENT4}>
              <strong>✓ No MKS Tier · All Liveborn · No Thoracic Cage Involvement</strong><br />
              <span style={{ fontSize: 12 }}>
                CEP120/JBTS31 is NOT a TZ-diffusion-barrier gene → no MKS-tier lethal allele class (contrast MKS1/JBTS28,
                B9D2/JBTS34). JBTS31 biallelic null → SRPS2B not MKS1. All JBTS31 patients (hypomorphic class) are
                liveborn with normal respiratory function at birth — no early chest complication.
              </span>
            </Alert>

            {/* KPI cards */}
            <Section title="Phenotype Prevalence (40-patient cohort, seed-479)" color={ACCENT}>
              <div className="row g-2">
                <KPI label="MTS on MRI" value="100%" color={ACCENT} />
                <KPI label="Cerebellar Ataxia" value={kpis.cerebellar_ataxia_pct} color={ACCENT9} />
                <KPI label="Neonatal Hypotonia" value={kpis.neonatal_hypotonia_pct} color={ACCENT2} />
                <KPI label="Oculomotor Apraxia" value={kpis.oculomotor_apraxia_pct} color={ACCENT3} />
                <KPI label="Breathing Dysreg" value={kpis.breathing_dysreg_pct} color={ACCENT5} />
                <KPI label="Intellectual Dis." value={kpis.intellectual_disability} color={ACCENT10} />
                <KPI label="Retinal Rod-Cone" value={kpis.retinal_pct} color={ACCENT8} />
                <KPI label="Renal NPHP-like" value={kpis.renal_nphp_pct} color={ACCENT7} />
                <KPI label="Polydactyly" value={kpis.polydactyly_pct} color={ACCENT3} />
                <KPI label="Thorax Normal" value="100%" color={ACCENT4} />
                <KPI label="All Liveborn" value="100%" color={ACCENT4} />
                <KPI label="No MKS Tier" value="✓" color={ACCENT4} />
              </div>
            </Section>

            {/* Protein domain architecture */}
            <Section title="CEP120 Protein Domain Architecture (1,007 aa — Centriole Elongation Scaffold)" color={ACCENT2}>
              <div style={{ fontFamily: 'monospace', fontSize: 11, background: '#f0f4ff', borderRadius: 6, padding: 12, lineHeight: 1.7 }}>
                <div>
                  <span style={{color: ACCENT2, fontWeight: 700}}>aa   1– 200</span>
                  <span style={{color: ACCENT2}}> │ CPAP/CENPJ-binding domain (ARM fold contacts; Arg200 boundary; JBTS31: Pro240Leu mild)</span>
                </div>
                <div>
                  <span style={{color: ACCENT, fontWeight: 700}}>aa 201– 550</span>
                  <span style={{color: ACCENT}}> │ Central coiled-coil 1 / CEP135 recruiter (SSNA1/NA14 interaction; self-oligomerisation; Ala501Val)</span>
                </div>
                <div>
                  <span style={{color: ACCENT3, fontWeight: 700}}>aa 551– 750</span>
                  <span style={{color: ACCENT3}}> │ Central coiled-coil 2 / TULP3 contact (JBTS30 link; Glu562Lys; ciliary transport coordination)</span>
                </div>
                <div>
                  <span style={{color: ACCENT5, fontWeight: 700}}>aa 751–1007</span>
                  <span style={{color: ACCENT5}}> │ C-terminal HEAT/ARM repeats / PCM anchor (subdistal appendage positioning; Ala813Pro; Leu1019Pro)</span>
                </div>
              </div>
            </Section>

            {/* Cilia phenotype */}
            <Section title="JBTS31 Cilia Phenotype — SHORT but PRESENT (IF/EM Diagnostic Pattern)" color={ACCENT3}>
              <DataTable
                accent={ACCENT3}
                headers={["IF Marker", "JBTS31 (CEP120 hypomorphic)", "JBTS27 (ARMC9 null)", "SRTD3 (DYNC2H1 dynein-2)", "JBTS28 (MKS1 TZ gate)"]}
                rows={[
                  ["Acetylated α-tubulin", "FEW, SHORT cilia (50–70% WT)", "ABSENT cilia", "Present, normal length", "Present, abnormal TZ"],
                  ["GT335 (glutamylation)", "Reduced (short axoneme)", "Absent", "Present + tip club", "Reduced (Y-link absent)"],
                  ["ARL13B ciliary IF", "Weak short signal", "Absent", "Tip accumulation", "Reduced"],
                  ["CP110 persistence", "NOT present (CP110 cleared)", "PRESENT (CP110 persists)", "Not present", "Not present"],
                  ["EM tip morphology", "Short axoneme, no club/bulge", "No axoneme", "Club/bulge tip", "Short axoneme, no Y-link"],
                ]}
              />
            </Section>

            {/* Allele spectrum */}
            <Section title="CEP120 Allele-Phenotype Spectrum (Three Disease Thresholds)" color={ACCENT6}>
              <DataTable
                accent={ACCENT6}
                headers={["Allele Class", "CEP120 Residual Function", "Disease", "Key Feature", "Outcome"]}
                rows={[
                  ["Biallelic null (truncating + truncating)", "<5%", "SRPS2B", "Perinatal lethal skeletal dysplasia", "Perinatal death"],
                  ["Missense + null / moderate missense", "10–30%", "SRTD19 (#617895)", "Narrow thorax + short ribs; VEPTR", "Liveborn; respiratory Rx"],
                  ["Mild missense (both alleles, residual ≥30%)", "30–45%", "JBTS31 (#617761)", "MTS only; normal CXR; all liveborn", "Normal thorax; cerebellar"],
                ]}
              />
              <div style={{ fontSize: 11, color: '#555' }}>
                The threshold for thoracic cage narrowing is <strong>~&lt;30% CEP120 function</strong>. JBTS31 alleles
                retain enough CEP120 to avoid this threshold. Cerebellar cilia are more sensitive to centriole length
                reduction → MTS appears at lower Hh impairment than skeletal phenotype.
              </div>
            </Section>

            {/* DDx pearls */}
            <Section title="Diagnostic Pearls — JBTS31 vs Allelic Diseases" color={ACCENT5}>
              {(overview.ddx_pearls || []).map((pearl, i) => (
                <Alert key={i} color={ACCENT5}>
                  <span style={{ fontSize: 12 }}>• {pearl}</span>
                </Alert>
              ))}
            </Section>

            {/* Ethnic distribution */}
            <Section title="Ethnic Distribution (40-patient cohort)" color={ACCENT2}>
              <div className="row g-2">
                {Object.entries(overview.ethnic_breakdown || {}).map(([eth, n]) => (
                  <div key={eth} className="col-6 col-md-4 mb-2">
                    <div className="d-flex justify-content-between align-items-center px-2 py-1 rounded" style={{ background: ACCENT2 + '12', fontSize: 12 }}>
                      <span>{eth}</span>
                      <span className="fw-bold" style={{ color: ACCENT2 }}>{n}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          </div>
        )}

        {/* ── BREAKDOWN TAB ─────────────────────────────────────────────── */}
        {tab === 'Diagnostic Breakdown' && breakdown && (
          <div>
            <Section title="Phenotype Prevalence (%) — 40-patient cohort" color={ACCENT}>
              <DataTable
                accent={ACCENT}
                headers={["Phenotype", "N / 40", "Prevalence (%)"]}
                rows={Object.entries(prevPct).map(([k, v]) => [
                  k.replace(/_/g, ' ').replace(/pct/g, '%'),
                  Math.round(v * N_COHORT / 100),
                  `${v}%`,
                ])}
              />
            </Section>

            <Section title="MTS Severity Distribution" color={ACCENT}>
              <DataTable
                accent={ACCENT}
                headers={["MTS Severity", "Count", "%"]}
                rows={Object.entries(breakdown.mts_severity_distribution || {}).map(([k, v]) => [
                  k, v, `${Math.round(v / N_COHORT * 100)}%`
                ])}
              />
            </Section>

            <Section title="Allele Class Distribution" color={ACCENT2}>
              <DataTable
                accent={ACCENT2}
                headers={["Allele Class", "Count", "%"]}
                rows={Object.entries(breakdown.allele_class_distribution || {}).map(([k, v]) => [
                  k, v, `${Math.round(v / N_COHORT * 100)}%`
                ])}
              />
            </Section>

            <Section title="Renal Distribution" color={ACCENT7}>
              <DataTable
                accent={ACCENT7}
                headers={["Renal Finding", "Count", "%"]}
                rows={Object.entries(breakdown.renal_distribution || {}).map(([k, v]) => [
                  k, v, `${Math.round(v / N_COHORT * 100)}%`
                ])}
              />
            </Section>

            <Section title="Retinal Distribution" color={ACCENT8}>
              <DataTable
                accent={ACCENT8}
                headers={["Retinal Finding", "Count", "%"]}
                rows={Object.entries(breakdown.retinal_distribution || {}).map(([k, v]) => [
                  k, v, `${Math.round(v / N_COHORT * 100)}%`
                ])}
              />
            </Section>

            <Section title="Key Pathogenic Variants (JBTS31 spectrum)" color={ACCENT6}>
              <DataTable
                accent={ACCENT6}
                headers={["Variant", "Domain", "Population", "Severity"]}
                rows={(breakdown.key_variants || []).map(v => [
                  v.variant, v.domain, v.population, v.severity
                ])}
              />
            </Section>

            <Section title="Patient Cohort (40 patients, seed-479)" color={ACCENT5}>
              <DataTable
                accent={ACCENT5}
                headers={["ID", "Ethnicity", "MTS", "Ataxia", "Retinal", "Renal", "Polydactyly", "Thorax", "Variant 1"]}
                rows={(breakdown.cohort_table || []).map(p => [
                  p.id, p.ethnicity, p.mts, p.ataxia, p.retinal, p.renal, p.polydactyly, p.thorax, p.variant_1
                ])}
              />
            </Section>
          </div>
        )}

        {/* ── CENTRIOLE ELONGATION PEARLS TAB ──────────────────────────── */}
        {tab === 'Centriole Elongation Pearls' && overview && (
          <div>
            <Alert color={ACCENT}>
              <strong>CEP120 / JBTS31 — Centriole Elongation, Short Cilia, Allele-Threshold Disease</strong><br />
              <span style={{ fontSize: 12 }}>
                CEP120 biallelic hypomorphic LOF → short centrioles → dysfunctional basal bodies →
                SHORT cilia (50–70% WT) → Hedgehog impaired in cerebellum → MTS. The disease is defined
                by allele severity class: only very hypomorphic alleles produce JBTS31 without thoracic cage involvement.
              </span>
            </Alert>

            <Section title="Centriole Elongation Pathway (CEP120 Step)" color={ACCENT2}>
              <div style={{ fontSize: 12, lineHeight: 1.9, background: '#f0f4ff', borderRadius: 6, padding: 12 }}>
                <div>① CPAP/CENPJ seeds the procentriole cartwheel at the proximal end of the growing daughter centriole</div>
                <div>② <strong>CEP120</strong> is recruited to the distal tip of the procentriole via CPAP-binding domain (aa 1–200)</div>
                <div>③ CEP120 recruits CEP135 + SSNA1/NA14 → stabilises the distal centriole wall scaffold</div>
                <div>④ Procentriole elongates to full ~450 nm length under CEP120/CEP135/SSNA1 coordination</div>
                <div>⑤ CP110/CEP97 caps the distal tip; TTBK2 phosphorylates CEP164 → CP110 removed → axoneme extension begins</div>
                <div>⑥ Mature basal body docks to plasma membrane via distal appendages (CEP83, SCLT1, FBF1, LRRC45) → ciliogenesis</div>
                <div style={{ color: ACCENT6, marginTop: 6 }}>
                  ⚠ In JBTS31 (hypomorphic CEP120): steps ②–④ are <em>partially impaired</em> →
                  centrioles elongate to ~70% normal length → basal body docking reduced efficiency →
                  cilia form but remain SHORT (50–70% WT). Cerebellar threshold: Shh impaired → MTS.
                  Thoracic threshold: NOT reached (JBTS31 alleles retain ~30–40% CEP120 function).
                </div>
              </div>
            </Section>

            <Section title="JBTS31 Molecular Phenotype — Tissue-Specific Thresholds" color={ACCENT3}>
              <DataTable
                accent={ACCENT3}
                headers={["Tissue", "Cilia Phenotype", "JBTS31 Consequence", "Prevalence", "Management"]}
                rows={[
                  ["Cerebellar granule cell progenitors", "Short cilia → Shh impaired", "Vermis hypoplasia → MTS (100%)", "100%", "Physiotherapy; occupational therapy; anti-epileptics"],
                  ["Renal tubular epithelium", "Short cilia → TIN / cysts", "NPHP-like / CKD → ESRD in ~6%", "~30%", "Annual creatinine/eGFR; transplant for CKD 5"],
                  ["Photoreceptor connecting cilium", "Short connecting cilium", "Rod-cone dystrophy (progressive)", "~20%", "Annual ERG/VF; gene therapy trial eligibility"],
                  ["Costal chondrocytes (thorax)", "Short cilia → mild Hh impairment", "THRESHOLD NOT REACHED → normal CXR", "0%", "CXR at diagnosis to exclude SRTD19"],
                  ["Hepatic cholangiocytes", "Mild cilia shortening", "Very rarely CHF (<5%)", "~5%", "Annual LFTs; biopsy if fibrosis markers"],
                ]}
              />
            </Section>

            <Section title="CPAP-CEP120-CEP135 Centriole Elongation Complex" color={ACCENT2}>
              <DataTable
                accent={ACCENT2}
                headers={["Protein", "Gene", "JBTS/SRTD Allele", "Role in CEP120 Elongation Complex", "CEP120 Contact"]}
                rows={[
                  ["CPAP/SAS-4", "CENPJ", "JBTS9", "Procentriole seed; recruits CEP120 to distal tip", "N-term CPAP-binding domain (aa 1–200)"],
                  ["CEP135", "CEP135", "—", "Distal centriole scaffold; stabilises elongation platform", "CC1 (aa 201–550)"],
                  ["SSNA1/NA14", "SSNA1", "—", "Centriole ring assembly; co-operates with CEP135", "CC1 interface"],
                  ["TULP3", "TULP3", "JBTS30", "IFT-A cargo adaptor; CEP120 CC2 contact; ciliary import", "CC2 (aa 551–750)"],
                  ["CP110", "CP110", "(N/A)", "Centriole distal cap; CEP120 co-operates with cap removal", "Indirect (TTBK2/ARMC9 axis)"],
                ]}
              />
            </Section>

            <Section title="Mandatory Co-Sequencing: CENPJ (CPAP, JBTS9)" color={ACCENT6}>
              <Alert color={ACCENT6}>
                <strong>CPAP/CENPJ is CEP120's obligate binding partner</strong><br />
                <span style={{ fontSize: 12 }}>
                  Digenic CEP120 + CENPJ (compound heterozygous across two genes) documented in one JBTS31 kindred.
                  CENPJ biallelic LOF alone → JBTS9 (Joubert Syndrome Type 9; OMIM #612285).
                  For all JBTS31 patients: <strong>CENPJ MUST be co-sequenced</strong> — single-gene CEP120 panel
                  is insufficient. Order whole-exome sequencing (WES) or a JBTS/ciliopathy panel that includes CENPJ.
                </span>
              </Alert>
            </Section>

            <Section title="JBTS31 Renal Transplant — Curative, Cell-Autonomous" color={ACCENT7}>
              <Alert color={ACCENT7}>
                <strong>Renal transplant is curative for the renal component of JBTS31</strong><br />
                <span style={{ fontSize: 12 }}>
                  The centriole elongation defect is cell-autonomous: donor kidney tubular cells carry
                  intact CEP120 → no disease recurrence post-transplant. ~6% of JBTS31 patients reach CKD 5/ESRD.
                  Transplant planning should begin at CKD 3–4. Transplant does NOT improve cerebellar, retinal,
                  or intellectual disability components (established during neurodevelopment).
                </span>
              </Alert>
            </Section>

            <Section title="Surveillance Protocol for JBTS31" color={ACCENT5}>
              <DataTable
                accent={ACCENT5}
                headers={["Interval", "Investigation", "Rationale"]}
                rows={[
                  ["At diagnosis", "Brain MRI (axial + sagittal through PC); CXR; renal US; ophthalmic ERG/VF; ECG", "Confirm MTS; exclude SRTD19; baseline renal and retinal"],
                  ["At diagnosis", "CENPJ co-sequencing (WES panel); parental carrier testing", "Digenic risk; family counselling"],
                  ["Annual", "Serum creatinine + eGFR; urine protein/creatinine; renal USS", "Monitor for CKD progression"],
                  ["Annual", "ERG + visual field (ophthalmic)", "Monitor rod-cone dystrophy progression"],
                  ["Annual", "LFTs (ALT/GGT/ALP); USS liver", "Screen for hepatic fibrosis (<5%)"],
                  ["Every 2–3 yr", "Brain MRI (repeat if new cerebellar symptoms)", "Track cerebellar atrophy"],
                  ["Ongoing", "Physiotherapy + occupational therapy + speech therapy (if dysarthria)", "Cerebellar ataxia management"],
                ]}
              />
            </Section>
          </div>
        )}

        {/* ── DEFINITIONS TAB ───────────────────────────────────────────── */}
        {tab === 'Definitions' && definitions && (
          <div>
            <Alert color={ACCENT}>
              <strong>JBTS31 / CEP120 — Key Terms &amp; Concepts</strong>
            </Alert>
            {(definitions.definitions || []).map((d, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: '#fff', border: `1px solid ${ACCENT}30` }}>
                <div className="fw-bold mb-1" style={{ color: ACCENT }}>{d.term}</div>
                <div style={{ fontSize: 13, lineHeight: 1.7, color: '#333' }}>{d.definition}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
