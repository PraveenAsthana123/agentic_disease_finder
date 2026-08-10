'use client';
import Link from 'next/link';
import { ROLES, FEATURES, AGENTIC } from '../../lib/nav';
import { OPS } from '../../lib/ops';
export default function SiteNav() {
  const dd = (title, items, base) => (
    <li className="nav-item dropdown">
      <a className="nav-link dropdown-toggle text-white" href="#" data-bs-toggle="dropdown">{title}</a>
      <ul className="dropdown-menu dropdown-menu-dark">
        {items.map(i => <li key={i.id}><Link className="dropdown-item" href={`${base}/${i.id}`}>{i.icon?i.icon+' ':''}{i.name}</Link></li>)}
      </ul>
    </li>);
  return (
    <nav className="navbar navbar-expand-lg navbar-dark" style={{background:'#0b1f3a'}}>
      <div className="container-fluid">
        <Link className="navbar-brand fw-bold" href="/">🧠 NeuroAI</Link>
        <ul className="navbar-nav me-auto">
          <li className="nav-item"><Link className="nav-link text-white" href="/dashboard">Dashboard</Link></li>
          {dd('Departments', ROLES, '/role')}
          {dd('Features', FEATURES, '')}
          {dd('Agentic AI', AGENTIC, '/agentic')}
          {dd('Ops & Governance', OPS, '/ops')}
          <li className="nav-item"><Link className="nav-link text-white" href="/drift">📉 Drift</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognition-link">🔗 Cognition</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-scales">📋 Scales</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuro-scales">🩺 Neuro Scales</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-types">AI Types</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-type-coverage">🤖 AI Coverage</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/epilepsy-nurse">💉 Epilepsy Nurse</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication">&#x1f48a; Medication</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/treatment-efficacy">&#x1f4c9; Treatment Efficacy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication-refills">&#x1f504; Refills</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-severity">&#x26a1; Seizure Severity</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-timeline">&#x23f1;&#xfe0f; Seizure Timeline</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mri-review">&#x1f9e0; MRI Review</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ilae-classification">&#x1f9ec; ILAE Classification</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-metadata">&#x1f9e0; Seizure Metadata</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/incident-management">&#x1f6a8; Incidents</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/inbox">&#x2709;&#xfe0f; Inbox</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/token-cost">&#x1f4b0; Token/Cost</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/vector-db">&#x1f5c4;&#xfe0f; Vector DB</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/knowledge-graph">&#x1f578;&#xfe0f; Knowledge Graph</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-risk">&#x26a0;&#xfe0f; AI Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-control-tower">&#x1f5fc; AI Control Tower</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-governance">&#x1f3db;&#xfe0f; AI Governance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-observability">&#x1f52d; AI Observability</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/responsible-ai-dashboard">&#x2696;&#xfe0f; Responsible AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/global-approval-policy">&#x2705; Approval Policy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/executive-scorecard">&#x1f4ca; Exec Scorecard</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/workflow">Workflow</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mcp-security">MCP Security</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sec-ops">&#x1f512; SecOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/shadow-ai">&#x1f575;&#xfe0f; Shadow AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/feature-flags">&#x1f6a9; Feature Flags</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/change-management">&#x1f504; Change Mgmt</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-registry">&#x1f5c2;&#xfe0f; Model Registry</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-retirement">&#x1f4e6; Model Retire</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-finops">&#x1f4b8; AI FinOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/billing-claims">&#x1f4b3; Billing &amp; Claims</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ssep">SSEP</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/vep">VEP</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hrv">HRV</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/abpm">ABPM/Holter</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cloud-ops">&#x2601;&#xfe0f; Cloud Ops</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-ops">&#x1f4e6; DataOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/observability">&#x1f441;&#xfe0f; Observability</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/traces">&#x1f4e1; HTTP Traces</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/llmops">&#x1f4ac; LLMOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/stack">Tech Stack</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-ai-stack">&#x1f9e0; EEG AI Stack</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/classify">Classify</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/xai">XAI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/interpretable-ai">Interpretable AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/decision-ai">Decision AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/recovery-trajectory">&#x1f4c8; Recovery Trajectory</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/reporting-ai">&#x1f4cb; Reporting AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/video-eeg">&#x1f4f9; Video EEG</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/safety-network">&#x1f6e1;&#xfe0f; Safety Network</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/camera-monitoring">&#x1f4f9; Camera Monitor</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pnes-screening">&#x1f9e0; PNES Screening</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-comparison">&#x1f4ca; Model Compare</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-comparison">&#x1f500; Patient Compare</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hospitalization">&#x1f3e5; Hospitalization</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-diary">&#x1f4d3; Seizure Diary</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-triggers">&#x26a1; Seizure Triggers</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-forecasting">&#x1f52e; Seizure Forecasting</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-prediction">&#x1f4c8; Seizure Prediction</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-ai-rag-pipeline">&#x1f9ec; EEG→AI→RAG Pipeline</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/automatic-pipelines">&#x2699;&#xfe0f; Auto Pipelines</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scheduled-jobs">&#x23f0; Scheduled Jobs</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/onboarding">&#x1f4cb; Onboarding</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pharmacogenomics">&#x1f9ec; Pharmacogenomics</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognitive-decline">&#x1f9e0; Cognitive Decline</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/adl">&#x1f3c3; ADL</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/quality-of-life">&#x1f31f; Quality of Life</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-documents">&#x1f4c4; Patient Docs</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/consent-management">&#x1f4dc; Consent</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-decisions">&#x2696;&#xfe0f; Clinical Decisions</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/workbench">&#x1f9ea; Workbench</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neurolab-readiness">&#x1f3e5; NeuroLab Readiness</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/iot-devices">&#x1f4e1; IoT Devices</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/iot-fleet">&#x1f6f0;&#xfe0f; IoT Fleet</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/iot-gateway">&#x1f4f6; IoT Gateway</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-band">&#x26a1; Seizure Band</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/wearable-devices">&#x231a; Wearable Devices</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/wearable-readings">&#x1f4c8; Wearable Readings</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/smartwatch">&#x231a; Smartwatch</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/emotiv-wearable">&#x1fa7a; Emotiv Wearable</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/emotiv-epoc-x">&#x1f9e0; Emotiv EPOC X</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/emotiv-insight">&#x1f9e0; Emotiv Insight</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ecg-patch">&#x1fa7a; ECG Patch</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/device-mode">&#x1f4e1; Device Mode</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-mobile">&#x1f4f1; Patient Mobile App</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/device-telemetry">&#x1f4f6; Device Telemetry</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/therapy">&#x1f9d8; Therapy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/rehab-plans">&#x1f9b4; Rehab Plans</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/daily-plans">&#x1f4c5; Daily Plans</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/multimodal-fusion">&#x1f52c; Multimodal Fusion</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/phase2-multimodal">&#x1f4ca; Phase 2 Multimodal</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-appointments">&#x1f4c5; Appointments</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/surgical-outcomes">&#x1fa7a; Surgical Outcomes</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/presurgical-evaluation">&#x1f52c; Pre-Surgical Eval</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/caregiver-readiness">&#x1f91d; Caregiver Readiness</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/caregiver-app">&#x1f4f1; Caregiver App</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/r-ipo">&#x1f504; Role IPO</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/inference-testing">&#x1f9ea; Inference Testing</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cross-patient-benchmark">&#x1f9ec; Cross-Patient</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/loso">&#x1f4ca; LOSO CV</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/bootstrap-ci">&#x1f4c9; Bootstrap CI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/regulatory-compliance">&#x1f3db; Regulatory</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-requirements">&#x1f4ca; Data Requirements</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/production-issues">&#x1f6a8; Production Issues</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-risk-stratification">&#x1f6a8; Clinical Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuro-advancements">&#x1f9e0; Neuro Advancements</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/enterprise-pipelines">&#x1f3ed; Enterprise Pipelines</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-manager">&#x1f4cb; Data Manager</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/annotation">&#x1f3f7;&#xfe0f; Annotation QC</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neurologist">&#x1f9e0; Neurologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/human-evaluation">&#x1f9d1;&#x200d;&#x2695;&#xfe0f; HITL Review</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/alerts">&#x1f6a8; Alerts</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-dark-factory">&#x1f3ed; Dark Factory</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mobile-alerts">&#x1f4f1; Mobile Alerts</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/emergency-sos">&#x1f6a8; Emergency SOS</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/lsss">&#x1f4ca; LSSS</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/phq9-dashboard">&#x1f4cb; PHQ-9</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mood-comorbidity">&#x1f9e0; Mood-Comorbidity</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/psychiatrist">&#x1f4ac; Psychiatrist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/radiologist">&#x1f9b4; Radiologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-psychologist">&#x1f9e0; Clin. Psychologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuropsychologist">&#x1f9e0; Neuropsychologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/occupational-therapist">&#x1f590;&#xfe0f; OT</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dietitian">&#x1f957; Dietitian</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/social-worker">&#x1f91d; Social Worker</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/program-coordinator">&#x1f9ed; Prog. Coordinator</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/consultant-matrix">&#x1f465; Consultant Matrix</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-analysis-results">&#x1f52c; EEG Analysis Results</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dataset-coverage">&#x1f5c2;&#xfe0f; Dataset Coverage</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hallucination">&#x1f9e0; Hallucination Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/deep-learning">&#x1f9e0; Deep Learning</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/time-frequency">&#x1f4ca; Time-Frequency (TFR)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/temporal-approval">&#x231b; Temporal Approval</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-technician">&#x1f4e1; EEG Technician</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/is-sop">&#x1f4d1; IS-SOP Compliance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/irb-reviewer">&#x2696;&#xfe0f; IRB Reviewer</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuro-ai-ecosystem">&#x1f9ec; Neuro AI Ecosystem</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/fairness">&#x2696;&#xfe0f; Fairness &amp; Bias</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/openclaw">&#x1f9be; OpenClaw Agents</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sleep-staging">&#x1f4a4; Sleep Staging</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/icd10-coding">&#x1f3f7;&#xfe0f; ICD-10 Coding</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/channel-quality">&#x1f4f6; Channel Quality</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/secure-messaging">&#x1f4ac; Secure Messaging</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/referrer-notify">&#x1f4e8; Referrer Notify</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/referral-triage">&#x1f3e5; Referral Triage</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/guided-assessment">&#x1f4dd; Guided Assessments</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/business-workflows">&#x2699;&#xfe0f; Business Workflows</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/consultant-workflows">&#x1f465; Consultant Workflows</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-flowcharts">&#x1f500; Clinical Flowcharts</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/role-process-flows">&#x1f9ed; Role Process Flows</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/role-dashboards">&#x1f4cb; Role Dashboards</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognitive-tests">&#x1f9e0; Cognitive Tests</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-demographics">&#x1f465; Demographics</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/population-health">&#x1f30d; Population Health</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/models">&#x1f5c2;&#xfe0f; Model Registry</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/audit">&#x1f4dc; Audit Trail</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-steward">&#x1f6e1;&#xfe0f; Data Steward</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/causal-ai">&#x1f517; Causal AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/federated-learning">&#x1f310; Federated Learning</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/grounding-gate">&#x1f6e1;&#xfe0f; Grounding Gate</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-artifact-analysis">&#x1f4ca; EEG Artifacts</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-viewer">&#x1f4c9; EEG Viewer</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/iot-engineer">&#x1f4e1; IoT Engineer</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-security">&#x1f6e1;&#xfe0f; AI Security</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-federation">&#x1f310; AI Federation</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/telehealth">&#x1f4f9; Telehealth</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/voice-ai">&#x1f399;&#xfe0f; Voice AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/edge-deploy">&#x1f680; Edge Deploy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/operator-requests">&#x1f4e5; Operator Requests</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuro-tests-catalog">&#x1f9ea; Neuro Tests Catalog</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/education-modules">&#x1f4da; Education Modules</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/validation-studies">&#x1f52c; Validation Studies</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/epilepsy-challenges">&#x26a1; Epilepsy Challenges</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/simulations">&#x1f504; Process Simulations</Link></li>
        </ul>
        <span className="navbar-text text-info small">● System Online · SSR</span>
      </div>
    </nav>);
}
