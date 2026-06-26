export const ROLES = [
  { id:'eeg-tech', name:'EEG Technician', icon:'🧠', focus:'Signal acquisition, montage QC, artifact flags' },
  { id:'neurologist', name:'Neurologist', icon:'🩺', focus:'Seizure review, per-subject LOSO, clinical decision' },
  { id:'psychiatrist', name:'Psychiatrist', icon:'🧩', focus:'Schizophrenia/depression screening, behavioral markers' },
  { id:'ot', name:'Occupational Therapist', icon:'🤝', focus:'Rehab tracking, motor-imagery/BCI progress' },
  { id:'researcher', name:'Researcher', icon:'🔬', focus:'Model training, statistics, fairness, XAI' },
  { id:'admin', name:'Admin', icon:'⚙️', focus:'Users, audit, model registry, system health' },
  { id:'epilepsy-nurse', name:'Epilepsy Nurse', icon:'💉', focus:'Seizure diary, SUDEP risk, AED adherence, action plans' },
];
export const FEATURES = [
  { id:'dashboard', name:'Dashboard' }, { id:'classify', name:'Classify (EDF)' },
  { id:'eeg-viewer', name:'EEG Signal Viewer' }, { id:'loso', name:'Per-Subject LOSO' },
  { id:'xai', name:'Explainability (SHAP)' }, { id:'fairness', name:'Fairness / RAI' },
  { id:'models', name:'Model Registry' }, { id:'audit', name:'Audit Trail' },
];
export const AGENTIC = [
  { id:'agentic-stack', name:'10-Layer Agentic Stack' }, { id:'council', name:'Council of Agents' },
  { id:'status-agents', name:'Status Agents' }, { id:'llm-gateway', name:'LLM Gateway' },
  { id:'control-tower', name:'AI Control Tower' },
];
export const DISEASES = ['Epilepsy','Schizophrenia','Depression','Autism','Parkinson','Stress'];
