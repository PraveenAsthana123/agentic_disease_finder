'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT81 Architecture & IFT-B Complex', 'Definitions'];

// NPHP19 colour scheme — IFT81 / IFT-B anterograde / JBTS35 / tubulin-binding
const ACCENT  = '#1a237e';   // deep indigo — IFT81 / IFT-B core; ultra-rare; primary
const ACCENT2 = '#880e4f';   // deep pink — Joubert syndrome (JBTS35); MTS; cerebellar
const ACCENT3 = '#e65100';   // burnt orange — retinal dystrophy; rod-cone; LCA-like
const ACCENT4 = '#1b5e20';   // deep green — IFT-B complex hierarchy; anterograde transport
const ACCENT5 = '#b71c1c';   // deep red — ESRD; CKD progression
const ACCENT6 = '#37474f';   // dark slate — molecular architecture; IFT-B scaffold
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; 12q arm confusion; digenic
const ACCENT8 = '#004d40';   // dark teal — IFT train assembly; ciliary biology

const SEED = 377;
const _COHORT_SIZE = 40;

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

export default function NPHP19Page() {
  const [tab, setTab]   = useState(0);
  const [ov, setOv]     = useState(null);
  const [bk, setBk]     = useState(null);
  const [df, setDf]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/nphp19/overview`).then(r => r.json()),
      fetch(`${API}/api/nphp19/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nphp19/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBk(b); setDf(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading NPHP19 cohort…</div>;
  if (error)   return <div className="container py-5 text-danger">Error: {error}</div>;
  if (!ov)     return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', border: `2px solid ${ACCENT}` }}>
        <div className="d-flex align-items-start gap-3 flex-wrap">
          <div style={{ flex: 1, minWidth: 260 }}>
            <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
              &#x1f9ec; Nephronophthisis Type 19 / Joubert Syndrome 35 (NPHP19/JBTS35) — IFT81 · IFT-B Anterograde Core
            </h4>
            <div className="small text-muted mb-1">
              <strong>IFT81</strong> · 12q23.1 · ~698 aa · IFT-B anterograde core bridge ·
              obligate heterodimer with IFT74 · tubulin-binding module · JBTS35 in ~65% ·
              retinal ~50–60% · ultra-rare (&lt;20 families 2026)
            </div>
            <div className="small">
              <Badge text="OMIM *605489" color={ACCENT} />
              <Badge text="#617302 JBTS35/NPHP19" color={ACCENT} />
              <Badge text="AR biallelic LOF" color={ACCENT6} />
              <Badge text="12q23.1" color={ACCENT8} />
              <Badge text="IFT-B anterograde" color={ACCENT4} />
              <Badge text="JBTS35 ~65%" color={ACCENT2} />
              <Badge text="Retinal ~50–60%" color={ACCENT3} />
              <Badge text="IFT74 co-sequence" color={ACCENT7} />
              <Badge text="Ultra-rare &lt;20 families" color={ACCENT5} />
            </div>
          </div>
          <div className="d-flex gap-2 flex-wrap">
            <span className="badge px-3 py-2" style={{ background: ACCENT, fontSize: '0.8em' }}>
              Ultra-rare &lt;20 families
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT2, fontSize: '0.8em' }}>
              JBTS35 {ov.pct_jbts35_confirmed}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT3, fontSize: '0.8em' }}>
              Retinal {ov.pct_retinal_dystrophy}%
            </span>
            <span className="badge px-3 py-2" style={{ background: ACCENT5, fontSize: '0.8em' }}>
              ESRD/Tx {ov.pct_esrd_or_transplant}%
            </span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert color={ACCENT4}>
        <strong>&#x1f9ec; IFT81 IS THE CORE BRIDGE OF THE IFT-B ANTEROGRADE COMPLEX:</strong> IFT81 forms an
        obligate heterodimer with IFT74 via N-terminal calponin homology (CH) domains — this
        IFT81/IFT74 module is the <strong>tubulin-binding engine of IFT-B1</strong>, importing alpha/beta-tubulin
        into growing cilia for axoneme assembly. IFT81 also bridges IFT-B1 (IFT88, IFT52, IFT46,
        IFT70) to IFT-B2 (IFT172, IFT57, IFT80, IFT38, IFT54, IFT20) as the structural scaffold.
        Loss of IFT81 → IFT-B core collapse → anterograde train failure → cilia absent →
        NPHP19 + JBTS35. Mechanistically DISTINCT from IFT-A retrograde subtypes
        (NPHP12/TTC21B; NPHP13/WDR19) and from distal appendage subtypes (NPHP18/CEP83; NPHP15/CEP164).
      </Alert>
      <Alert color={ACCENT2}>
        <strong>&#x1f9e0; JOUBERT SYNDROME 35 (JBTS35) — MOLAR TOOTH SIGN IN ~{ov.pct_jbts35_confirmed}% — BRAIN MRI MANDATORY:</strong> Biallelic
        IFT81 LOF causes Joubert syndrome (Molar Tooth Sign on axial MRI: cerebellar vermis
        hypoplasia + SCP elongation) in ~65% — higher penetrance than NPHP18/CEP83 (~55%).
        Oculomotor apraxia (OMA), neonatal hypotonia, episodic breathing irregularity
        (self-resolves ~2–3yr), cerebellar ataxia, developmental delay. Brain MRI MANDATORY
        at diagnosis. IFT81 is the dominant IFT-B gene causing JBTS; MTS identifies JBTS35 alleles.
      </Alert>
      <Alert color={ACCENT3}>
        <strong>&#x1f441;&#xfe0f; RETINAL DYSTROPHY IN ~{ov.pct_retinal_dystrophy}% — HIGHER THAN NPHP18 — ANNUAL ERG MANDATORY:</strong> Rod–cone
        dystrophy in ~48–55% (ERG abnormal) and LCA-like severe early in ~8–10% (ERG flat,
        neonatal nystagmus; null×null alleles). Retinal penetrance (~50–60%) exceeds NPHP18/CEP83
        (~30–40%) but is less than NPHP5/IQCB1 (&gt;95%). Retinal does NOT improve post-renal
        transplant (cell-autonomous photoreceptor connecting cilium defect). Annual ERG + fundoscopy
        mandatory from diagnosis in ALL IFT81 patients.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>&#x26a0;&#xfe0f; IFT74 OBLIGATE HETERODIMER — CO-SEQUENCING MANDATORY + CHROMOSOME 12q CONFUSION:</strong> IFT81 (12q23.1)
        and IFT74 (9p21.2) form an obligate heterodimer. Digenic ciliopathy (single heterozygous
        IFT81 + single heterozygous IFT74) is documented. IFT74 MUST be co-sequenced in all
        IFT81 cases. Additionally, IFT81 at 12q23.1 shares the same chromosome arm as
        CEP83 (12q22, NPHP18) and CEP290 (12q21.32, NPHP6) — targeted single-gene panels for
        CEP83 or CEP290 do NOT cover IFT81. NPHP1 MLPA misses IFT81. WES mandatory.
        {ov.pct_misdiagnosed_cep290}% of cohort initially tested CEP83/CEP290 (negative) before IFT81 identified.
      </Alert>
      <Alert color={ACCENT}>
        <strong>&#x2705; RENAL TRANSPLANT CURATIVE · RETINAL/CEREBELLAR CELL-AUTONOMOUS — ULTRA-RARE (&lt;20 FAMILIES):</strong> Donor
        kidney with functional IFT81 → intact IFT-B → normal tubular cilia → no TIN recurrence.
        Excellent renal outcomes. Retinal dystrophy and JBTS35 cerebellar features are
        cell-autonomous and do NOT improve post-transplant. Ultra-rare status (&lt;20 families 2026):
        IFT81 may be absent from targeted NPHP/JBTS gene panels; WES is the only reliable
        approach. Multi-disciplinary team mandatory for JBTS35 cases (nephrology + ophthalmology +
        developmental paediatrics + physiotherapy + OT + speech therapy).
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
            <KPI label="Cohort (n)" value={ov.cohort_n} color={ACCENT} />
            <KPI label="Median GFR" value={`${ov.median_gfr} ml/min`} color={ACCENT} />
            <KPI label="Median Hb" value={`${ov.median_hb} g/dL`} color={ACCENT6} />
            <KPI label="Median age Dx" value={`${ov.median_age_renal_dx}yr`} color={ACCENT} />
            <KPI label="ESRD/Transplant" value={`${ov.pct_esrd_or_transplant}%`} color={ACCENT5} />
            <KPI label="JBTS35 (MTS)" value={`${ov.pct_jbts35_confirmed}%`} color={ACCENT2} />
            <KPI label="Retinal dystrophy" value={`${ov.pct_retinal_dystrophy}%`} color={ACCENT3} />
            <KPI label="LCA-like" value={`${ov.pct_lca_like}%`} color={ACCENT3} />
            <KPI label="Situs inversus" value="<2%" color={ACCENT} />
            <KPI label="NPHP1 MLPA misdiag" value={`${ov.pct_misdiagnosed_nphp1}%`} color={ACCENT7} />
            <KPI label="12q arm misdiag" value={`${ov.pct_misdiagnosed_cep290}%`} color={ACCENT7} />
            <KPI label="JBTS-unknown misdiag" value={`${ov.pct_misdiagnosed_jbts_unknown}%`} color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="&#x1f9ec; IFT81 — IFT-B Anterograde Core Bridge (12q23.1)" color={ACCENT6}>
                <div className="small text-muted mb-2">
                  IFT81 (~698 aa) is a structural core subunit of the IFT-B anterograde
                  transport complex. Forms an obligate heterodimer with IFT74 via N-terminal
                  CH domains — the IFT81/IFT74 module is the tubulin-binding engine of IFT-B1.
                  IFT81 also bridges IFT-B1 to IFT-B2 subcomplexes within the IFT-B supercomplex.
                </div>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>IFT81 (Intraflagellar Transport 81 homolog)</td></tr>
                    <tr><td className="fw-bold">Location</td><td>12q23.1 (same arm as CEP83/NPHP18 at 12q22 and CEP290/NPHP6 at 12q21.32)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>~698 aa · IFT-B core bridge · CH domain (tubulin binding) · anterograde complex</td></tr>
                    <tr><td className="fw-bold">IFT-B hierarchy</td><td>IFT81/IFT74 heterodimer → IFT-B1 (IFT88, IFT52, IFT46, IFT70) → IFT-B2 (IFT172, IFT57, IFT80…)</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*605489</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#617302 (JBTS35 / NPHP19)</td></tr>
                    <tr><td className="fw-bold">Key partners</td><td>IFT74 (obligate heterodimer; 9p21.2; co-sequence mandatory); IFT88; IFT52; kinesin-2 (KIF3A/KIF3B/KAP)</td></tr>
                    <tr><td className="fw-bold">Phenotypes</td><td>NPHP19 (renal TIN) + JBTS35 (Joubert ~65%) + retinal (~50–60%) · no situs · no CHF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/2,000,000–5,000,000; &lt;20 published families (2026); ultra-rare</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive · biallelic LOF · digenic (IFT81 + IFT74) documented</td></tr>
                  </tbody>
                </table>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="&#x1f6a8; NPHP19 / JBTS35 Hallmark Features" color={ACCENT}>
                {[
                  ['IFT81 is the IFT-B anterograde core bridge — obligate IFT74 heterodimer; tubulin-binding engine of IFT-B1', ACCENT4,
                   'IFT81 forms an obligate heterodimer with IFT74 via N-terminal CH domains. This IFT81/IFT74 module is the primary tubulin-binding unit of IFT-B1, importing alpha/beta-tubulin dimers into growing cilia for axoneme assembly. IFT81 also bridges IFT-B1 to IFT-B2 as structural scaffold. Loss = IFT-B core disassembly → anterograde IFT train failure → cilia absent. Mechanistically distinct from IFT-A retrograde (NPHP12/TTC21B; NPHP13/WDR19) and DA subtypes (NPHP18/CEP83)'],
                  ['JBTS35 — Molar Tooth Sign in ~65% — Brain MRI mandatory at diagnosis', ACCENT2,
                   'Cerebellar vermis hypoplasia + SCP elongation (MTS) on axial T1/T2 MRI in ~65% of biallelic IFT81 cases — higher penetrance than NPHP18/CEP83 (~55%). Oculomotor apraxia (OMA), neonatal hypotonia, episodic breathing irregularity (self-resolves 2–3yr), cerebellar ataxia, developmental delay. MRI at diagnosis determines: JBTS35 alleles vs pure renal NPHP19 alleles; guides developmental paediatrics and ophthalmology surveillance'],
                  [`Retinal dystrophy ~${ov.pct_retinal_dystrophy}% — higher than NPHP18 (~30–40%); ERG annual mandatory`, ACCENT3,
                   'Rod-cone dystrophy in ~48–55% (ERG abnormal, progressive). LCA-like severe early in ~8–10% (ERG flat, neonatal nystagmus; null×null alleles). Higher retinal penetrance than NPHP18/CEP83 (~30–40%) reflecting broader IFT-B involvement in photoreceptor connecting cilium. Retinal does NOT improve post-transplant. Annual ERG + fundoscopy from diagnosis mandatory. Low-vision aids, mobility training as retinal disease progresses'],
                  ['IFT74 (9p21.2) obligate heterodimer — co-sequencing mandatory; digenic ciliopathy documented', ACCENT7,
                   'IFT81 and IFT74 form an obligate CH-domain heterodimer. Single heterozygous IFT81 + single heterozygous IFT74 constitutes a digenic ciliopathy pair (each in trans). IFT74 (9p21.2) must always be co-sequenced alongside IFT81. Digenic cases may be missed if only IFT81 is sequenced and one heterozygous variant is found — always check IFT74. WES captures both loci automatically'],
                  ['12q chromosome arm confusion — IFT81 at 12q23.1; CEP83 at 12q22; CEP290 at 12q21.32 — all different genes', ACCENT7,
                   'Three NPHP/JBTS genes on the same chromosome arm 12q: CEP290 (12q21.32, NPHP6), CEP83 (12q22, NPHP18), IFT81 (12q23.1, NPHP19). Targeted single-gene panels for any of these do NOT cover the others. NPHP1 MLPA (290kb) misses all three. Gene-specific: LCA10 panel (CEP290) + CEP83 panel (NPHP18) both MISS IFT81. WES is the only reliable diagnostic method. ~12% of cohort had 12q gene confusion before IFT81 identified'],
                  ['Ultra-rare (<20 families 2026) — absent from many NPHP/JBTS targeted gene panels', ACCENT5,
                   'IFT81 ultra-rare status means it is absent from many targeted NPHP and JBTS gene panels (designed around more common subtypes). Gene-unknown Joubert with renal + retinal: always consider IFT81 on WES. Digenic IFT81+IFT74 adds further complexity. Ultra-rarity limits precise genotype-phenotype delineation; caution in prognosis without specific literature search for identified variants'],
                  ['NPHP19 renal — TIN + corticomedullary cysts + ESRD (adolescent to adult); transplant curative', ACCENT,
                   'Tubulointerstitial nephritis + corticomedullary cysts + concentrating defect. ESRD onset variable (adolescent to early adult — later onset than NPHP1 ~13yr median; more variable given ultra-rare status and allele diversity). Bilateral small echogenic kidneys ± cysts on USS. Renal transplant CURATIVE: donor kidney has functional IFT81 → intact IFT-B → no TIN recurrence. Pre-emptive transplant preferred when donor available'],
                  ['Renal transplant curative · retinal/cerebellar cell-autonomous · multi-disciplinary team for JBTS35', ACCENT,
                   'Excellent post-transplant renal outcomes. JBTS35 cerebellar hypoplasia and retinal dystrophy are cell-autonomous — NOT corrected by renal transplant. Multi-disciplinary team mandatory for JBTS35 cases: nephrology + developmental paediatrics + ophthalmology + physiotherapy + OT + speech therapy. Pure renal NPHP19 (~18%): focused nephrology post-transplant only'],
                ].map(([title, color, sub], i) => (
                  <div key={i} className="mb-2 p-2 rounded" style={{ background: color + '12', borderLeft: `3px solid ${color}` }}>
                    <div className="fw-bold small" style={{ color }}>{title}</div>
                    <div className="text-muted" style={{ fontSize: '0.75em' }}>{sub}</div>
                  </div>
                ))}
              </Section>
            </div>
          </div>

          {/* Sample patients */}
          <Section title={`&#x1f4cb; Sample Patients (first 8 of ${_COHORT_SIZE} · seed=${SEED})`} color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead>
                  <tr style={{ background: ACCENT + '22' }}>
                    <th>ID</th><th>Ethnicity</th><th>CKD Stage</th>
                    <th>Age Dx (yr)</th><th>GFR</th>
                    <th>JBTS35</th><th>Retinal</th><th>Misdiagnosis</th>
                  </tr>
                </thead>
                <tbody>
                  {ov.patients.map(p => (
                    <tr key={p.id}>
                      <td><span className="badge" style={{ background: ACCENT }}>{p.id}</span></td>
                      <td style={{ fontSize: '0.72em' }}>{p.ethnicity.split('(')[0].trim()}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.ckd_stage.split('(')[0].trim()}</td>
                      <td>{p.age_renal_dx_yr}</td>
                      <td>{p.gfr_now_ml_min}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.jbts35_status.split('(')[0].trim().slice(0, 22)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.retinal_status.split('(')[0].trim().slice(0, 22)}</td>
                      <td style={{ fontSize: '0.72em' }}>{p.prior_misdiagnosis.split('(')[0].trim().slice(0, 26)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ── */}
      {tab === 1 && bk && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="CKD Stage / Renal Status Distribution" color={ACCENT}>
              {Object.entries(bk.ckd_stage_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="JBTS35 Status (Joubert — Molar Tooth Sign)" color={ACCENT2}>
              {Object.entries(bk.jbts35_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Retinal Involvement" color={ACCENT3}>
              {Object.entries(bk.retinal_status).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Urine Osmolality Tiers (Tubular Concentrating Defect)" color={ACCENT}>
              {Object.entries(bk.urine_osmolality_tiers).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Prior Misdiagnosis Distribution" color={ACCENT7}>
              {Object.entries(bk.prior_misdiagnosis).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
            <Section title="First Presenting Symptom" color={ACCENT}>
              {Object.entries(bk.first_symptom_distribution).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Kidney USS Phenotype" color={ACCENT8}>
              {Object.entries(bk.kidney_phenotype).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
              ))}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {Object.entries(bk.ethnicity).map(([k, v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 2: IFT81 Architecture & IFT-B Complex ── */}
      {tab === 2 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="&#x1f9ec; IFT81 Architecture, IFT-B Complex & Anterograde Transport" color={ACCENT4}>
              {df.genetic_architecture && Object.entries(df.genetic_architecture).filter(([k]) => k !== 'key_variants').map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT4}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{typeof v === 'string' ? v : JSON.stringify(v)}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="&#x1f9ea; Key Variants (IFT81 NPHP19/JBTS35)" color={ACCENT}>
              {df.genetic_architecture && df.genetic_architecture.key_variants && df.genetic_architecture.key_variants.map((v, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="&#x1f4ca; NPHP Subtype Comparison" color={ACCENT6}>
              {df.nphp_comparison && Object.entries(df.nphp_comparison).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded"
                  style={{
                    background: k.includes('★') ? ACCENT + '18' : ACCENT6 + '08',
                    borderLeft: `3px solid ${k.includes('★') ? ACCENT : ACCENT6}`
                  }}>
                  <div className="fw-bold small" style={{ color: k.includes('★') ? ACCENT : ACCENT6 }}>{k}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-12">
            <Section title="&#x1fa7a; Differential Diagnosis Table" color={ACCENT7}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT7 + '18' }}>
                    <tr><th>Condition</th><th>Key Distinguishing Features from NPHP19/JBTS35</th></tr>
                  </thead>
                  <tbody>
                    {df.ddx_table && Object.entries(df.ddx_table).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold" style={{ color: ACCENT7, minWidth: 200 }}>{k}</td>
                        <td className="small">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && df && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Disease Definition" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {['disease','omim_gene','omim_disease','chromosome','inheritance','prevalence'].map(k => (
                    <tr key={k}>
                      <td className="fw-bold text-capitalize" style={{ width: 140 }}>{k.replace(/_/g,' ')}</td>
                      <td>{df[k]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="Mechanism" color={ACCENT2}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.mechanism}
              </div>
            </Section>
            <Section title="&#x1f3e5; Treatment" color={ACCENT}>
              {df.treatment && Object.entries(df.treatment).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Key Clinical Features" color={ACCENT3}>
              {df.key_clinical_features && Object.entries(df.key_clinical_features).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT3 + '08', borderLeft: `3px solid ${ACCENT3}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="Diagnostic Criteria" color={ACCENT6}>
              {df.diagnostic_criteria && Object.entries(df.diagnostic_criteria).map(([k, v]) => (
                <div key={k} className="mb-2 p-2 rounded" style={{ background: ACCENT6 + '08', borderLeft: `3px solid ${ACCENT6}` }}>
                  <div className="fw-bold small" style={{ color: ACCENT6 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{v}</div>
                </div>
              ))}
            </Section>
            <Section title="Prognosis" color={ACCENT}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT + '06', lineHeight: 1.7 }}>
                {df.prognosis}
              </div>
            </Section>
            <Section title="Cohort Note" color={ACCENT6}>
              <div className="small text-muted p-2 rounded" style={{ background: ACCENT6 + '06', lineHeight: 1.6 }}>
                {df.cohort_note}
              </div>
            </Section>
          </div>
        </div>
      )}

      <div className="mt-4 pt-3 border-top">
        <Link href="/" className="btn btn-sm btn-outline-secondary me-2">&#x2190; Portal Home</Link>
        <Link href="/nphp18" className="btn btn-sm btn-outline-primary">&#x2190; NPHP18 (CEP83/DA Foundation)</Link>
      </div>
    </div>
  );
}
