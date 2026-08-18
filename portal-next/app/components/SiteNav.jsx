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
          <li className="nav-item"><Link className="nav-link text-white" href="/model-drift">🎯 Model Drift</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognition-link">🔗 Cognition</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-scales">📋 Scales</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuro-scales">🩺 Neuro Scales</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-types">AI Types</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-type-coverage">🤖 AI Coverage</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/stories-tests">&#x1f4d6; Stories &amp; Tests</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/qa-test-suite">&#x1f9ea; QA Test Suite</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/epilepsy-nurse">💉 Epilepsy Nurse</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pharmacist">&#x1f48a; Pharmacist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication">&#x1f48a; Medication</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication-adherence">&#x2705; Med Adherence</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/treatment-efficacy">&#x1f4c9; Treatment Efficacy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication-refills">&#x1f504; Refills</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication-interaction">&#x26a0;&#xfe0f; Drug Interactions</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-severity">&#x26a1; Seizure Severity</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-timeline">&#x23f1;&#xfe0f; Seizure Timeline</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mri-review">&#x1f9e0; MRI Review</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dicom-viewer">&#x1f5bc;&#xfe0f; DICOM Viewer</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ilae-classification">&#x1f9ec; ILAE Classification</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-metadata">&#x1f9e0; Seizure Metadata</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/incident-management">&#x1f6a8; Incidents</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/inbox">&#x2709;&#xfe0f; Inbox</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/token-cost">&#x1f4b0; Token/Cost</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/vector-db">&#x1f5c4;&#xfe0f; Vector DB</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/knowledge-graph">&#x1f578;&#xfe0f; Knowledge Graph</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-risk">&#x26a0;&#xfe0f; AI Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-lifecycle">&#x1f504; AI Lifecycle</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-control-tower">&#x1f5fc; AI Control Tower</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/system-health">&#x1f5a5;&#xfe0f; System Health</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/admin-users">&#x1f465; Admin Users</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/advisor-issues">&#x1f50d; Advisor Issues</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/root-cause-analysis">&#x1f50d; RCA Center</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-governance">&#x1f3db;&#xfe0f; AI Governance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-governance">&#x1f3db;&#xfe0f; Model Governance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-observability">&#x1f52d; AI Observability</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/responsible-ai-dashboard">&#x2696;&#xfe0f; Responsible AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/global-approval-policy">&#x2705; Approval Policy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/executive-scorecard">&#x1f4ca; Exec Scorecard</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/executive-ai">&#x1f916; Executive AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/workflow">Workflow</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mcp-security">MCP Security</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sec-ops">&#x1f512; SecOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/shadow-ai">&#x1f575;&#xfe0f; Shadow AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/feature-flags">&#x1f6a9; Feature Flags</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/change-management">&#x1f504; Change Mgmt</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-registry">&#x1f5c2;&#xfe0f; Model Registry</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-retirement">&#x1f4e6; Model Retire</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-finops">&#x1f4b8; AI FinOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-roi">&#x1f4ca; AI ROI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/carbon-tracker">&#x1f331; Carbon Tracker</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/billing-claims">&#x1f4b3; Billing &amp; Claims</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ssep">SSEP</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/vep">VEP</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/rns">&#x26a1; RNS</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/bera">&#x1f442; BERA (ABR)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mep">&#x26a1; MEP (TMS)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/emg">&#x1f4aa; EMG</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sfemg">&#x1f9ec; SFEMG</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ncv">&#x26a1; NCV</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/blink-reflex">&#x1f441;&#xfe0f; Blink Reflex</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ssr">&#x1f4a6; SSR</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hrv">HRV</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/abpm">ABPM/Holter</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/abpm-holter">&#x1f493; ABPM-Holter Combined</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cloud-ops">&#x2601;&#xfe0f; Cloud Ops</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-ops">&#x1f4e6; DataOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/observability">&#x1f441;&#xfe0f; Observability</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/traces">&#x1f4e1; HTTP Traces</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/transaction-log">&#x1f4cb; Transaction Log</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/conversation-log">&#x1f4ac; Conv Log</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/llmops">&#x1f4ac; LLMOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/stack">Tech Stack</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-ai-stack">&#x1f9e0; EEG AI Stack</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/classify">Classify</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/xai">XAI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/interpretable-ai">Interpretable AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/decision-ai">Decision AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/recovery-trajectory">&#x1f4c8; Recovery Trajectory</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/reporting-ai">&#x1f4cb; Reporting AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/report-layout">&#x1f4c4; Report Layout</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/video-eeg">&#x1f4f9; Video EEG</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/video-correlation">&#x1f3a5; Video Correlation</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-video">&#x1f3a5; Patient Video</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/safety-network">&#x1f6e1;&#xfe0f; Safety Network</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/camera-monitoring">&#x1f4f9; Camera Monitor</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/comorbidities">&#x1f9e0; Comorbidities</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pnes-screening">&#x1f9e0; PNES Screening</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pnes-differential">&#x1f9e0; PNES Differential</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-comparison">&#x1f4ca; Model Compare</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-calibration">&#x1f4d0; Model Calibration</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-comparison">&#x1f500; Patient Compare</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hospitalization">&#x1f3e5; Hospitalization</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/discharge-planning">&#x1f4cb; Discharge Planning</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/longitudinal-timeline">&#x1f4c5; Longitudinal Timeline</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-diary">&#x1f4d3; Seizure Diary</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/trigger-logs">&#x1f4c5; Trigger Logs</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-triggers">&#x26a1; Seizure Triggers</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-forecasting">&#x1f52e; Seizure Forecasting</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-horizon">&#x23f1;&#xfe0f; Seizure Horizon Analysis</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-prediction">&#x1f4c8; Seizure Prediction</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/raw-eeg-waveform">&#x1f9e0; Raw EEG Waveform</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-clinical-panel">&#x1f4c8; EEG Clinical Signal Panel</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-channel-quality-map">&#x1f4f6; EEG Channel Quality Map</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-ai-rag-pipeline">&#x1f9ec; EEG→AI→RAG Pipeline</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/automatic-pipelines">&#x2699;&#xfe0f; Auto Pipelines</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scheduled-jobs">&#x23f0; Scheduled Jobs</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/onboarding">&#x1f4cb; Onboarding</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/onboarding-intake">&#x1f4dd; Intake Classification</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pharmacogenomics">&#x1f9ec; Pharmacogenomics</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/aed-side-effects">&#x26a0;&#xfe0f; AED Side Effects</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/aed-compliance">&#x2705; AED Compliance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/aed-polypharmacy">&#x1f48a; AED Polypharmacy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-assessments">&#x1f4ca; Clinical Assessments</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/assessment-catalog">&#x1f4cb; Assessment Catalog</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/battery-scoring">&#x1f52c; Battery Scoring</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognitive-decline">&#x1f9e0; Cognitive Decline</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/adl">&#x1f3c3; ADL</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pro-outcomes">&#x1f4cb; PRO Outcomes</Link></li>
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
          <li className="nav-item"><Link className="nav-link text-white" href="/wearables-digital-twin">&#x1f9ec; Digital Twin</Link></li>
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
          <li className="nav-item"><Link className="nav-link text-white" href="/tele-rehab">&#x1f4f9; Tele-Rehab</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/copm-fim">&#x1f4cb; COPM/FIM Instruments</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/rehab-goals">&#x1f3af; Rehab Goal Tracking</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/home-program">&#x1f3e0; Home Program Builder</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/daily-plans">&#x1f4c5; Daily Plans</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/multimodal-fusion">&#x1f52c; Multimodal Fusion</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/phase2-multimodal">&#x1f4ca; Phase 2 Multimodal</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-appointments">&#x1f4c5; Appointments</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/appointments">&#x1f4cb; Clinic Appointments</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-outcomes">&#x1f4ca; Clinical Outcomes</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/surgical-outcomes">&#x1fa7a; Surgical Outcomes</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/presurgical-evaluation">&#x1f52c; Pre-Surgical Eval</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/caregiver-readiness">&#x1f91d; Caregiver Readiness</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/caregiver-app">&#x1f4f1; Caregiver App</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/caregivers">&#x1fac2; Caregivers</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/r-ipo">&#x1f504; Role IPO</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/inference-testing">&#x1f9ea; Inference Testing</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cross-patient-benchmark">&#x1f9ec; Cross-Patient</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/bonn">&#x1f3db; Bonn Ext. Validation</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/loso">&#x1f4ca; LOSO CV</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/bootstrap-ci">&#x1f4c9; Bootstrap CI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/accuracy-options">&#x1f4ca; Accuracy Methods</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/regulatory-compliance">&#x1f3db; Regulatory</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-requirements">&#x1f4ca; Data Requirements</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/production-issues">&#x1f6a8; Production Issues</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/production-monitoring">&#x1f6e1;&#xfe0f; Production Monitoring</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/agent-loop">&#x1f501; Agent Loop Monitor</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/knowledge-management">&#x1f5c2;&#xfe0f; Knowledge Mgmt</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-risk-stratification">&#x1f6a8; Clinical Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuro-advancements">&#x1f9e0; Neuro Advancements</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/enterprise-pipelines">&#x1f3ed; Enterprise Pipelines</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-manager">&#x1f4cb; Data Manager</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-uploads">&#x1f4e4; EEG Uploads</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/annotation">&#x1f3f7;&#xfe0f; Annotation QC</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neurologist">&#x1f9e0; Neurologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/esignature">&#x1f58a;&#xfe0f; E-Signature Reports</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/human-evaluation">&#x1f9d1;&#x200d;&#x2695;&#xfe0f; HITL Review</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/component-findings">&#x1f52c; Component Findings</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/alerts">&#x1f6a8; Alerts</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-dark-factory">&#x1f3ed; Dark Factory</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mobile-alerts">&#x1f4f1; Mobile Alerts</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/emergency-sos">&#x1f6a8; Emergency SOS</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/lsss">&#x1f4ca; LSSS</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/phq9-dashboard">&#x1f4cb; PHQ-9</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cssrs-dashboard">&#x26a0;&#xfe0f; C-SSRS</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/moca-dashboard">&#x1f9e9; MoCA</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/nddi-e">&#x1f9e0; NDDI-E</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mood-comorbidity">&#x1f9e0; Mood-Comorbidity</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/psychiatrist">&#x1f4ac; Psychiatrist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/radiologist">&#x1f9b4; Radiologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-psychologist">&#x1f9e0; Clin. Psychologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuropsychologist">&#x1f9e0; Neuropsychologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/slp">&#x1f5e3;&#xfe0f; SLP</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/p300-erp">&#x1f9e0; P300/ERP</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mmn">&#x1f9e0; MMN</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuropsych-battery">&#x1f9e0; Neuropsych Battery</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neurophysiologist">&#x1f4e1; Neurophysiologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-technologist">&#x1f9ea; EEG Technologist</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-advisor">&#x1f916; AI/ML Advisor</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/governance-advisor">&#x2696;&#xfe0f; Governance Advisor</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/methodology-advisor">&#x1f393; Methodology Advisor</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/biostatistician">&#x1f4ca; Biostatistician</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/signal-quality">&#x1f4e1; Signal Quality</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/abpm-holter">&#x2764;&#xfe0f; ABPM/Holter</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/occupational-therapist">&#x1f590;&#xfe0f; OT</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/fim-dashboard">&#x1f3cb;&#xfe0f; FIM</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/copm-dashboard">&#x1f3af; COPM</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dietitian">&#x1f957; Dietitian</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/social-worker">&#x1f91d; Social Worker</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/program-coordinator">&#x1f9ed; Prog. Coordinator</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/consultant-matrix">&#x1f465; Consultant Matrix</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-analysis-results">&#x1f52c; EEG Analysis Results</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dataset-coverage">&#x1f5c2;&#xfe0f; Dataset Coverage</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/datasets">&#x1f5c4;&#xfe0f; Datasets Registry</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hallucination">&#x1f9e0; Hallucination Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/deep-learning">&#x1f9e0; Deep Learning</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/transfer-learning">&#x1f504; Transfer Learning</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hybrid-cnn">&#x1f9ec; Hybrid CNN-LSTM</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/feature-gaps">&#x1f50d; DL Review Gap Analysis</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/time-frequency">&#x1f4ca; Time-Frequency (TFR)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/topomap">&#x1f9e0; 10-20 Topomap</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-data-formats">&#x1f4c2; EEG Data Formats</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/temporal-approval">&#x231b; Temporal Approval</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-technician">&#x1f4e1; EEG Technician</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/is-sop">&#x1f4d1; IS-SOP Compliance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/irb-reviewer">&#x2696;&#xfe0f; IRB Reviewer</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neuro-ai-ecosystem">&#x1f9ec; Neuro AI Ecosystem</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/fairness">&#x2696;&#xfe0f; Fairness &amp; Bias</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/openclaw">&#x1f9be; OpenClaw Agents</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sleep-staging">&#x1f4a4; Sleep Staging</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sleep-stage-analysis">&#x1f6cc; Sleep Stage Analysis</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/icd10-coding">&#x1f3f7;&#xfe0f; ICD-10 Coding</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/channel-quality">&#x1f4f6; Channel Quality</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-acquisition">&#x1f50c; EEG Acquisition</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/recording-conditions">&#x1f4f9; Recording Conditions</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hv-photic">&#x26a1; HV/Photic</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/secure-messaging">&#x1f4ac; Secure Messaging</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/secure-messages">&#x1f4e9; Secure Messages</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/referrer-notify">&#x1f4e8; Referrer Notify</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/referral-triage">&#x1f3e5; Referral Triage</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/referral-records">&#x1f4c1; Referral Records</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/guided-assessment">&#x1f4dd; Guided Assessments</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/business-workflows">&#x2699;&#xfe0f; Business Workflows</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/consultant-workflows">&#x1f465; Consultant Workflows</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-flowcharts">&#x1f500; Clinical Flowcharts</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/role-process-flows">&#x1f9ed; Role Process Flows</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/role-dashboards">&#x1f4cb; Role Dashboards</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/role-specs">&#x1f4c4; Role Specs Registry</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/role-challenges">&#x26a1; Role Challenges &amp; AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/role-tests">&#x1f9ea; Role Tests</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognitive-tests">&#x1f9e0; Cognitive Tests</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-demographics">&#x1f465; Demographics</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/population-health">&#x1f30d; Population Health</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/models">&#x1f5c2;&#xfe0f; Model Registry</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/audit">&#x1f4dc; Audit Trail</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/regulatory-audit-trail">&#x1f4cb; Reg Audit Trail</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/regulatory-submissions">&#x1f4c4; Reg Submissions</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-compliance">&#x2696;&#xfe0f; AI Compliance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-steward">&#x1f6e1;&#xfe0f; Data Steward</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/causal-ai">&#x1f517; Causal AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/federated-learning">&#x1f310; Federated Learning</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/grounding-gate">&#x1f6e1;&#xfe0f; Grounding Gate</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-artifact-analysis">&#x1f4ca; EEG Artifacts</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-viewer">&#x1f4c9; EEG Viewer</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eeg-viz">&#x1f9e0; EEG Viz Platform</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hitl">&#x1f9d1;&#x200d;&#x2695;&#xfe0f; HITL Evaluation</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/iot-engineer">&#x1f4e1; IoT Engineer</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-incident">&#x1f6a8; AI Incidents</Link></li>
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
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-burden">&#x26a1; Seizure Burden</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/presurgical-eval">&#x1f52c; Pre-Surgical Eval</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-journey">&#x1f6e4;&#xfe0f; Patient Journey</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-freedom">&#x1f3c6; Seizure Freedom</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sudep-risk">&#x26a1; SUDEP Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/drug-resistant-epilepsy">&#x1f48a; Drug-Resistant Epilepsy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/status-epilepticus">&#x1f6a8; Status Epilepticus</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ceeg-monitoring">&#x1f4e1; cEEG Monitoring</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neurosurgeon">&#x1f9e0; Neurosurgeon / Epilepsy Surgery</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/workflow-efficiency">&#x2699;&#xfe0f; Clinical Workflow Efficiency</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/genetic-epilepsy">&#x1f9ec; Genetic Epilepsy Syndromes</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dravet">&#x1f9ec; Dravet Syndrome (SCN1A)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/tsc">&#x1f9ec; Tuberous Sclerosis Complex (TSC)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/lgs">&#x1f9e0; Lennox-Gastaut Syndrome (LGS)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/west-syndrome">&#x1f476; West Syndrome (Infantile Spasms)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/jme">&#x1f9e0; Juvenile Myoclonic Epilepsy (JME)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/tle">&#x1f9e0; Temporal Lobe Epilepsy (TLE)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/fle">&#x1f9e0; Frontal Lobe Epilepsy (FLE)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cae">&#x1f9d2; Childhood Absence Epilepsy (CAE)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ole">&#x1f441;&#xfe0f; Occipital Lobe Epilepsy (OLE)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ple">&#x1f9e0; Parietal Lobe Epilepsy (PLE)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pme">&#x1f9ec; Progressive Myoclonic Epilepsy (PME)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/bects">&#x1f9d2; SeLECTS / BECTS (Rolandic Epilepsy)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/doose">&#x1fa96; Doose Syndrome (MAE)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/rasmussen">&#x1f9e0; Rasmussen&#39;s Encephalitis</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/fires">&#x1f525; FIRES (Febrile Infection-Related Epilepsy)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/glut1">&#x1f9ec; GLUT1 Deficiency Syndrome (De Vivo)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/angelman">&#x1f9ec; Angelman Syndrome (UBE3A / 15q11)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/rett">&#x1f9ec; Rett Syndrome (MECP2 / Xq28)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cdkl5">&#x1f9ec; CDKL5 Deficiency Disorder (CDD / Xp22)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnq2">&#x26a1; KCNQ2 Encephalopathy (Kv7.2 / 20q13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/stxbp1">&#x1f9e0; STXBP1 Encephalopathy (Munc18-1 / 9q34)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scn2a">&#x26a1; SCN2A Encephalopathy (Nav1.2 / 2q24)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scn8a">&#x26a1; SCN8A Encephalopathy (Nav1.6 / 12q13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnt1">&#x26a1; KCNT1 Encephalopathy (KNa1.1 / EIMFS / 9q34)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnt2">&#x26a1; KCNT2 Epilepsy (DEE57 / West Syndrome / KNa1.2-Slick / Quinidine-No-Evidence / 1q31.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna1h">&#x26a1; CACNA1H Epilepsy (GGE / CAE / JME / Cav3.2 T-type Ca&#178;&#8314; / Ethosuximide Precision / 16p13.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna1g">&#x26a1; CACNA1G Epilepsy (GGE / CAE / JME / Cav3.1 T-type Ca&#178;&#8314; / TC-LTCS Primary / 17q21.33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna1i">&#x26a1; CACNA1I Epilepsy (GGE / CAE / JME / Cav3.3 T-type Ca&#178;&#8314; / TRN-Dominant / ETX-Level-B / 22q13.1)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pcdh19">&#x1f9ec; PCDH19 Clustering Epilepsy (Protocadherin-19 / Xq22)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/grin2a">&#x1f9e0; GRIN2A Epilepsy-Aphasia Spectrum (GluN2A / CSWS / LKS / 2q33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/syngap1">&#x1f9ec; SYNGAP1 Encephalopathy (SYNGAPathy / RasGAP / 6p21)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/depdc5">&#x1f9ec; DEPDC5 Focal Epilepsy (GATOR1 Complex / FFEVF / mTOR / 22q12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/prrt2">&#x26a1; PRRT2 Epilepsy Spectrum (BFIE / PKD / ICCA / 16p11.2)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/slc6a1">&#x1f9ec; SLC6A1 Epilepsy (GAT-1 / MAE / Doose Syndrome / 3p25)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcna2">&#x26a1; KCNA2 Epilepsy (Kv1.2 / GOF-LOF-DEE / 4-AP Precision / 1p13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcna1">&#x26a1; KCNA1 Epilepsy / EA1 (Kv1.1 / Episodic Ataxia / Myokymia / 12p13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/tsc1">&#x1f331; TSC1 Epilepsy (Tuberous Sclerosis / mTOR / Hamartin / Everolimus / 9q34)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/tsc2">&#x1f33f; TSC2 Epilepsy (Tuberous Sclerosis / Tuberin / mTOR / Severe-70% / 16p13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/wwox">&#x1f9ec; WWOX Epilepsy / WOREE (DEE28 / FRA16D / WW-Oxidoreductase / AR / 16q23)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/lgi1">&#x1f9e0; LGI1 Epilepsy / ADLTE (Auditory Aura / ADAM22-ADAM23 / AD-70%-penetrance / 10q23)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/rorb">&#x1f9ec; RORB Epilepsy / GGE (Thalamocortical TF / Absence-Myoclonic-MAE-DEE / 9q21)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sptan1">&#x1f9ec; SPTAN1 Epilepsy / DEE5 (Alpha-II Spectrin / AIS Cytoskeleton / West→LGS / 9q34)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/chd2">&#x1f9ec; CHD2 Epilepsy / GGE-Photo (Chromatin Remodeling / H3.3 / PPR-75% / MEI / 15q26)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/calm">&#x1f9ec; CALM1/2/3 Calmodulinopathy (DEE + Long-QT / CPVT5 / ICD / 3-gene-1-protein / 14q·2p·19q)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/nprl2">&#x1f9ec; NPRL2/NPRL3 Epilepsy (GATOR1 Complex / FFEVF / FCD IIb / mTOR Everolimus / 3p24·8q24)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/aldh7a1">&#x1f9ec; ALDH7A1 Epilepsy (Pyridoxine-Dependent / Antiquitin / PDE / 5q23)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pnpo">&#x1f9ec; PNPO Epilepsy (PLP-Dependent / Pyridoxamine-5-phosphate-Oxidase / Neonatal-EE / NOT-Pyridoxine / VPA-INH-TGB-CI / 17q21.32)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/eef1a2">&#x1f9ec; EEF1A2 Epilepsy (DEE-5 / Translation-Elongation / Postnatal-Switch-Diagnostic-Clock / Polymicrogyria / West-LGS / PHT-CBZ-CI / TGB-ABSOLUTE / 20q13.33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabbr2">&#x1f9ec; GABBR2 Epilepsy (DEE-59 / GABA-B Receptor Subunit 2 / Metabotropic-GABA / GOF-Constitutive-Gi / LOF-Autoreceptor-Loss / Baclofen-Precision-LOF / TGB-ABSOLUTE-NCSE / West-LGS / 22q12.2)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/shank3">&#x1f9ec; SHANK3 Epilepsy (Phelan-McDermid-Syndrome / PSD-Scaffold / mGluR5-AMPA-NMDA / IGF-1-Precision / SHANK3-Regression-Syndrome / VGB-AVOID / Post-Anaesthetic-Regression / 22q13.33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dyrk1a">&#x1f9ec; DYRK1A Epilepsy (DYRK1A-Syndrome / DEE-Microcephaly / NRSF-Nav1.1-Pathway / PHT-CBZ-HIGH-RISK / DYRK1A-Inhibitors-ABSOLUTE-CI / Folinic-Acid-FRA1 / VPA-HDAC-Epigenetic / 21q22.13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mef2c">&#x1f9ec; MEF2C Epilepsy (MEF2C-Haploinsufficiency-Syndrome / MHS / MADS-Box-Transcription-Factor / GABAergic-Interneuron-LOF / Rett-Like-Without-MECP2 / PHT-CBZ-HIGH-RISK / TGB-ABSOLUTE-NCSE / LTG-Mono-Myoclonic-Aggravation / Photosensitivity-35pct / VPA-HDAC-Epigenetic / 5q14.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabbr1">&#x1f9ec; GABBR1 Epilepsy (GABA-B-Receptor-Subunit-1 / Venus-Flytrap-Ligand-Binding / GABBR1a-Presynaptic-Sushi / GABBR1b-Postsynaptic-GIRK / GEFS+-Focal-Absence / Baclofen-Precision-LOF / TGB-ABSOLUTE-NCSE / LTG-Mono-Myoclonic-HIGH / 6p22.1)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/iqsec2">&#x1f9ec; IQSEC2 Epilepsy (X-Linked-DEE / ArfGEF-BRAG1 / AMPAR-Trafficking / Myoclonic-Encephalopathy / IS-West / PHT-CBZ-HIGH-RISK / TGB-ABSOLUTE-NCSE / LEV-Behavioural-XLID-Caution / VGB-ERG-Mandatory / Xp11.22)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pten">&#x1f3af; PTEN Epilepsy (PHTS / mTORopathy / Cowden / BRRS / ASD-Macrocephaly / Everolimus-mTOR-Precision / Cancer-Surveillance-NCCN / 10q23.31)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/nhlrc1">&#x1f9e0; NHLRC1 Epilepsy (Lafora Disease Type 2 / EPM2B / Malin-E3-Ubiquitin-Ligase / Progressive-Myoclonic-Epilepsy / CBZ-OXC-PHT-ABSOLUTE-CI / Metformin-AMPK-Disease-Modifying / 6p22.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/epm2a">&#x1f9ec; EPM2A Epilepsy (Lafora Disease Type 1 / Laforin / Dual-Specificity-Glucan-Phosphatase / CBM-Domain-W32G-Basque-Founder / DSP-Domain-C266-Phosphatase / CBZ-OXC-PHT-ABSOLUTE-CI / Metformin-KD-Disease-Modifying / 6q24.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cstb">&#x1f9ec; CSTB Epilepsy (Unverricht-Lundborg Disease / EPM1 / Cystatin-B / Dodecamer-Repeat-Expansion-CCCCGCCCCGCG / Cathepsin-Inhibitor / Piracetam-Level-A-Action-Myoclonus / CBZ-OXC-PHT-ABSOLUTE-CI / GBP-Worsen-Myoclonus / Non-Fatal-PME / Finnish-Baltic-Founder / 21q22.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scarb2">&#x1fac0; SCARB2 Epilepsy (Action-Myoclonus-Renal-Failure-Syndrome / EPM4 / AMRF / LIMP-2 / Lysosomal-GBA1-Transport-LOF / FSGS-Proteinuria-Pathognomonic / GBP-PGB-Double-CI-Renal / ACE-ARB-Mandatory / Transplant-NOT-CNS-Disease-Modifying / Israeli-Arab-Roma-Founder / 4q21.1)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gosr2">&#x1f30a; GOSR2 Epilepsy (North-Sea-PME / EPM6 / Golgi-SNARE-v-SNARE / ER-to-Golgi-Transport-LOF / Gly144Trp-North-Sea-Founder / Scoliosis-Virtually-Universal / GBP-Orthopedic-Trap / CBZ-OXC-PHT-ABSOLUTE-CI / TGB-ABSOLUTE-NCSE / Non-Fatal-PME / 17q21.32)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/polg">&#x1f9ec; POLG Epilepsy (Alpers-Huttenlocher / mtDNA Depletion / VPA-CI / 15q26)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/slc2a1">&#x1f9ec; SLC2A1 Epilepsy (GLUT1-DS / De Vivo Disease / KD Precision / 1p34)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabrb3">&#x1f9ec; GABRB3 Epilepsy (DEE28 / West Syndrome → LGS / GABA-A β3 / 15q12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hcn1">&#x26a1; HCN1 Epilepsy (DEE24 / Ih Channelopathy / GOF-LOF Dual / Fever-Sensitive / 5p12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hcn2">&#x26a1; HCN2 Epilepsy (GEFS+ / Febrile Seizures / CAE / Ih TC-Dominant LOF / LTG-CI-LOF / 19p13.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/foxg1">&#x1f9e0; FOXG1 Syndrome (Congenital Rett Variant / DEE / Dyskinesias / 14q12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gnao1">&#x26a1; GNAO1 Encephalopathy (DEE17 / Ohtahara / G&#945;o GOF-LOF / 16q13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabrg2">&#x1f9ec; GABRG2 Epilepsy (DEE11 / GEFS+ / GABA-A &#947;2 / BDZ Hyposensitivity / 5q34)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/grin1">&#x1f9e0; GRIN1 Epilepsy (DEE / GluN1 Obligatory NMDA Subunit / D-serine LOF / Memantine GOF / 9q34.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/grin2b">&#x1f9e0; GRIN2B Epilepsy (DEE27 / GluN2B / NMDA Subunit 2B / Memantine Precision / 12p12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/grin2d">&#x1f9e0; GRIN2D Epilepsy (DEE / GluN2D / Extrasynaptic-Subcortical / Movement-Disorder / Memantine-GOF / 19q13.33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mecp2">&#x1f9ec; MECP2-Related Disorders (Rett Syndrome / MDS / X-Linked / Trofinetide / Xq28)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabra1">&#x1f9ec; GABRA1 Epilepsy (DEE19 / CAE / JME / GABA-A &#945;1 / BDZ-Rescue-Adjusted / 5q34)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabra2">&#x1f9ec; GABRA2 Epilepsy (GGE / GEFS+ / Alcohol-Sensitive / GABA-A &#945;2 / AIS Inhibition / 4p12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabra5">&#x1f9ec; GABRA5 Epilepsy (DEE65 / Hippocampal Tonic-Inhibition / BZD-Insensitive-&#945;5 / ACTH / 15q12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabrd">&#x1f9ec; GABRD Epilepsy (GGE / GEFS+ / Catamenial / GABA-A &#948; Subunit / Tonic Inhibition / Ganaxolone / 1p36.33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/slc13a5">&#x1f9b7; SLC13A5 Epilepsy (Citrate Transporter Deficiency / NAFE / EIEE25 / NaCT / Triheptanoin / AR / 17p13.1)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/atp1a3">&#x1f9ec; ATP1A3 Epilepsy (AHC / CAPOS / RDP / DEE-ATP1A3 / Na+/K+-ATPase &#945;3 / Flunarizine / 19q13.2)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/stx1b">&#x1f9ec; STX1B Epilepsy (GEFS+ Spectrum / Febrile Seizures Plus / Focal Epilepsy of Infancy / Syntaxin-1B / t-SNARE / 16p11.2)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gnb1">&#x1f9ec; GNB1 Epilepsy (DEE / Infantile Spasms / West Syndrome / G&#946;1 G-protein &#946;1 Subunit / GIRK / ACTH / 1p36.33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clcn2">&#x1f9ec; CLCN2 Epilepsy (GGE / JME / CAE / GTCS-Alone / CLC-2 Cl&#8315; Channel / Acetazolamide Precision / 3q26.1)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dnm1">&#x1f9ec; DNM1 Epilepsy (DEE31 / Dynamin-1 GTPase / Synaptic Vesicle Endocytosis / PV+ Interneuron / ACTH / KD / 9q34.11)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnb1">&#x26a1; KCNB1 Epilepsy (DEE26 / Kv2.1 Channelopathy / Delayed-Rectifier I&#8336; / GOF-LOF / 20q13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnc1">&#x26a1; KCNC1 Epilepsy (EPM7 / Progressive Myoclonic Epilepsy 7 / Kv3.1 Shaw K&#8314; / PV+ Fast-Spiking / R320H-Founder / CBZ-LTG-ABSOLUTE-CI / 21q22.13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnc2">&#x26a1; KCNC2 Epilepsy (GGE / Focal / DEE / Kv3.2 Shaw K&#8314; / GOF-LOF-Dual / PV+-TRN / CBZ-LTG-HIGH-RISK-LOF / 12q21.32)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna1a">&#x1f9e0; CACNA1A Epilepsy (DEE42 / Cav2.1 P/Q-Type Ca&#178;&#8314; / EA2 / FHM1 / 19p13.13)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna1b">&#x1f9ec; CACNA1B Epilepsy (DEE / NDMSB / Cav2.2 N-type HVA Ca&#178;&#8314; / LEV-Level-B / Hyperkinetic / 9q34.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna1e">&#x1f9ec; CACNA1E Epilepsy (DEE69 / Cav2.3 R-type HVA Ca&#178;&#8314; / No-Precision-Blocker / ACTH-Level-A / 1q25.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna1c">&#x2764;&#xfe0f; CACNA1C Epilepsy (Timothy Syndrome / LQTS8 / Cav1.2 L-type HVA Ca&#178;&#8314; / Verapamil-Precision / 12p13.33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna1d">&#x1f9ec; CACNA1D Epilepsy (SANDD / DEE+Autism+Aldosteronism / Cav1.3 Low-Threshold L-type / Isradipine-Precision / 3p14.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cacna2d2">&#x1f9ec; CACNA2D2 Epilepsy (EECAT / &#x3b1;2-&#x3b4;-2 Auxiliary Subunit / Gabapentinoid-Binding Protein / Cerebellar Atrophy / AR-LOF / 3p21.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/tbc1d24">&#x1f9ec; TBC1D24 Epilepsy (DOORS Syndrome / FHEIG / DEE16 / RAB35-Rab-GAP / TLDc-Oxidative-Stress / AR-LOF / 16p13.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/chrna4">&#x1f9ec; CHRNA4 Epilepsy (ADNFLE / Nocturnal Frontal Lobe / nAChR &#x3b1;4 GOF / CBZ-First-Line / HLA-B1502 / 20q13.33)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnma1">&#x26a1; KCNMA1 Epilepsy (BK-Channel / MaxiK-Slo1 / Epilepsy+Paroxysmal-Dyskinesia GOF / Liang-Wang-Syndrome LOF / Quinidine-Precision / 10q22.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/chrnb2">&#x1f9ec; CHRNB2 Epilepsy (ADNFLE3 / nAChR &#x3b2;2 Subunit / &#x28;&#x3b1;4&#x29;&#x2082;&#x28;&#x3b2;2&#x29;&#x2083; GOF / Psychiatric-Comorbidity-40pct / V287M-Cognitive / HLA-B1502 / 1q21.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/chrna2">&#x1f9ec; CHRNA2 Epilepsy (ADNFLE2 / nAChR &#x3b1;2 Subunit / Habenulo-Interpeduncular / Rarest-ADNFLE / GOF-I279N-I304N / CBZ-First-Line / HLA-B1502 / 8p21.2)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mtor">&#x1f3af; MTOR Epilepsy (mTOR Kinase / mTORopathy Apex / GOF-Somatic-Mosaic / FCD-IIb+HME+MCAP+Smith-Kingsmore / Everolimus-Direct / 1p36.22)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cntnap2">&#x1f9ec; CNTNAP2 Epilepsy (CASPR2 / Cortical-Dysplasia-Focal-Epilepsy-CDFE / Pitt-Hopkins-like-1 / Juxtaparanodal-Kv1.1 / Bumetanide-NKCC1 / Largest-Human-Gene / 7q35-36.1)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scn1b">&#x2764;&#xfe0f; SCN1B Epilepsy (GEFS+ / Dravet-like DEE / Brugada Type 5 / Nav-&#946;1 / 19q13.12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/arx">&#x1f9ec; ARX Epilepsy (X-linked DEE / Ohtahara / West / XLAG / Partington / Xp21.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnq3">&#x26a1; KCNQ3 Epilepsy (BFNS-3 / DEE-KCNQ3 / Kv7.3 M-Current Partner / 11q23.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/kcnq5">&#x26a1; KCNQ5 Epilepsy (DEE / ID-Epilepsy / Kv7.5 M-Current / Interneuron-Enriched / LOF-Paradox / 6q14.1)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabrb2">&#x1f9e0; GABRB2 Epilepsy (GEFS+ / CAE / DEE-Dravet-like / GABA-A &#946;2 Subunit / 5q34)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/gabrb1">&#x1f9ec; GABRB1 Epilepsy (DEE / GEFS+ / GABA-A &#946;1 Subunit / Limbic-Hippocampal / Perampanel-AMPA / 4p12)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scn3a">&#x26a1; SCN3A Epilepsy (DEE67 / NaV1.3 / Focal Epilepsy of Infancy / R357Q-PMG / 2q24.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scn1a">&#x1f9ec; SCN1A Epilepsy (Dravet Syndrome / GEFS+ / SMEI / NaV1.1 PV-Interneuron / 2q24.3)</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/autoimmune-epilepsy">&#x1f9eb; Autoimmune Epilepsy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pediatric-epilepsy">&#x1f9d2; Pediatric Epilepsy Syndromes</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neonatal-eeg">&#x1f476; Neonatal EEG</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/vns-therapy">&#x26a1; VNS Therapy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/research-coordinator">&#x1f4cb; Research Coordinator</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/dba-research">&#x1f393; DBA Research KPIs</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/research-publication">&#x1f4c4; Research Publication Readiness</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/iec-irb-tracker">&#x1f4cb; IEC/IRB 173-Doc Tracker</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/insurance-preauth">&#x1f3e5; Insurance Pre-Authorization</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/epilepsy-in-women">&#x2640;&#xfe0f; Epilepsy in Women</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/catamenial">&#x1f534; Catamenial Epilepsy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-augmentation">&#x1f9ec; Data Augmentation</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-semiology">&#x1f9e9; Seizure Semiology</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/snn-neuromorphic">&#x26a1; SNN Neuromorphic</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hfo">&#x1f4a5; HFO Biomarkers</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pms">&#x1f4ca; AI Post-Market Surveillance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/expert-dashboards-catalog">&#x1f4da; Dashboard Catalog</Link></li>
        </ul>
        <span className="navbar-text text-info small">● System Online · SSR</span>
      </div>
    </nav>);
}
