# EEG Project Strategy Framework

## Complete 11-Phase Methodology for EEG-Based ML Projects

---

# PHASE 1: Project Framing + Success Criteria

| Item | What You Do | Options/Techniques | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|-------------------|------|--------|--------|---------------|------------|
| 1) Use-case definition | Define the decision/label | Stress vs calm, seizure event, MI class, sleep stage, cognitive load | Keep label objective + measurable | Vague labels like "mood" without ground truth | Problem statement + label schema | Label agreement ≥ 0.8 | Label noise → add rules + adjudication |
| 2) Data scope | Decide datasets, channels, sampling rate, duration, subjects | Public (DEAP, SEED, PhysioNet), private, hybrid | Start with 1 main + 1 benchmark dataset | Mixing many datasets too early | Data inventory sheet | Coverage: #subjects, #sessions, class balance | Dataset mismatch → harmonize montage + resample |
| 3) Evaluation target | Choose primary metric & constraints | Accuracy, F1, AUC, sensitivity/specificity, latency | Pick metric aligned with risk | Using only accuracy with imbalance | Metric plan | Baseline threshold (e.g., F1 ≥ X) | Imbalance → PR-AUC + class weights |
| 4) Split strategy | Define train/val/test separation | Subject-wise split, session-wise, LOSO | Use subject-wise for generalization | Random window split (leakage) | Split protocol doc | No subject overlap across splits | Small N → LOSO + nested CV |
| 5) Reproducibility | Define run config + versioning | Config files, seed control, environment lock | Make every run reproducible | "Works on my laptop" setup | Repro bundle | Same seed → same results | Drift in libs → pin versions |
| 6) Benchmark plan | Decide comparisons | Classical ML, CNN on STFT/CWT, simple LSTM | Use strong but fair baselines | Comparing only to weak baselines | Baseline matrix | Baseline within 10–15% of SOTA | Different preproc → document |
| 7) Risk & ethics | Identify privacy, consent, safety, bias | De-ID, governance, audit logs | Build privacy from day 1 | Storing raw identifiers | Risk register | Compliance checklist complete | Missing consent → exclude data |
| 8) Definition of done | Finalize deliverables | Model card, data card, test report | Make phase gates explicit | Endless iteration without gates | Roadmap with milestones | All gates met before next phase | Scope creep → freeze requirements |

---

# PHASE 2: Data Acquisition + Dataset Design

| Step | What You Do | Best-Practice Options | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|----------------------|------|--------|--------|---------------|------------|
| 1) Data source selection | Pick 1 primary + 1 benchmark dataset | DEAP/SEED, PhysioNet, BCI Comp, Emotiv | Choose datasets matching label + task | Mixing 4-5 datasets at start | Data source log | Dataset supports label + enough subjects | Different montages → map to common set |
| 2) Ground truth + label rules | Define label creation rules | Event labels, stimulus labels, self-report, clinician labels | Write label rules like a contract | "Label = whatever file says" | Label rulebook v1 | Inter-rater agreement checks | Noisy self-report → binning + smoothing |
| 3) Subject & session metadata | Build metadata table | Subject ID, session ID, age, device, montage, sampling rate | Treat metadata as first-class data | Storing without unique IDs | metadata.csv | 100% records have subject+session IDs | Missing fields → create "unknown" |
| 4) Inclusion/exclusion criteria | Decide valid recordings | Min duration, min channels, artifact tolerance, SQI threshold | Define criteria before modeling | Cherry-picking after results | Data QC policy | % kept vs dropped reported | Too strict → soften thresholds |
| 5) Harmonize sampling rate | Make standard sampling rate | Resample to 128/256/512 Hz | Resample after anti-alias filtering | Resampling raw without filtering | Standardized signals | No aliasing (spectral check) | Different Fs → note loss at high freq |
| 6) Montage/channel mapping | Align channels across datasets | 10-20 mapping, common subset | Keep a "channel map" table | Training on different channels | Channel mapping spec | ≥X common channels | Missing channel → drop or interpolate |
| 7) Windowing strategy | Convert EEG into fixed-size examples | 1-4s (MI), 4-8s (emotion), 8-30s (sleep) | Window length must match phenomenon | Tiny windows for slow phenomena | Windowing config | #windows per class balanced | Label timing mismatch → buffer zones |
| 8) Leakage prevention | Ensure no information leaks | Subject-wise split, session-wise split | Split before heavy transforms | Random window split across subjects | Split manifest file | 0 subject overlap | Same subject multiple sessions → keep together |
| 9) Class balance planning | Quantify imbalance early | Stratified selection, class weights, focal loss | Report imbalance clearly | Oversampling test set | Class distribution report | Imbalance ratio documented | Extreme imbalance → PR-AUC |
| 10) Dataset versioning | Freeze versions and generate hashes | Data cards + checksum, DVC | Make dataset reproducible | "I edited files manually" | Dataset v1 package | Hashes match across machines | Updates → new version (v1.1) |
| 11) Baseline-ready format | Standardize file structure | .npz or .parquet with X, y, meta; HDF5 | Keep consistent schema | Ad-hoc naming per experiment | Dataset loader + schema | Loader passes unit tests | Multi-dataset → adapter layer |
| 12) Data documentation | Create Data Card | Source, consent, demographics, biases | Document limitations early | Hiding data issues | Data Card v1 | Completeness checklist | Missing demographics → note explicitly |

---

# PHASE 3: Filtering + Preprocessing

| Step | What You Do | Common Choices | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|----------------|------|--------|--------|---------------|------------|
| 1) Raw sanity checks | Verify signal looks like EEG | Plot 10-30s per channel, PSD overview, check clipping | Catch bad files early | Preprocess blindly | QC notebook + flags | %files passing QC | Flatline/clipped → drop segment |
| 2) Unit + scaling standard | Ensure consistent units | Convert to µV; float32 | Normalize units across datasets | Mixing µV/mV | Standardization note | Unit consistency check passes | Unknown units → infer from amplitude |
| 3) Re-referencing | Choose reference scheme | CAR, linked mastoids, Cz reference | Keep reference consistent | Changing reference per experiment | reference_config.yaml | Stable PSD + reduced noise | Few channels → CAR may hurt |
| 4) Notch filter (mains) | Remove powerline interference | 50 Hz or 60 Hz notch | Confirm mains frequency | Notch too wide | Notch-applied signals | Reduced 50/60 peak in PSD | Strong harmonics → add 2nd notch |
| 5) Bandpass filter | Keep EEG bands, remove drift/EMG | 0.5-45 Hz (general), 1-40 Hz (emotion) | Pick cutoffs based on task | Using 0-Nyquist without reason | Bandpass-applied signals | Drift reduced + bands preserved | Epilepsy spikes → extend upper cutoff |
| 6) Anti-alias before resample | Protect spectrum when downsampling | Lowpass at new Nyquist margin | Always filter before downsampling | Downsampling raw signals | Resampled clean data | No aliasing in PSD | Resample mismatch → resample after bandpass |
| 7) Artifact detection (segment-level) | Detect contaminated windows | Amplitude threshold, kurtosis, peak-to-peak, SQI | Reject/mark artifacts | Deleting data without reporting | Artifact mask + report | %windows rejected reported | Too many rejects → loosen thresholds |
| 8) Artifact removal (ICA/ASR) | Reduce eye-blink, EMG, motion | ICA (EOG removal), ASR, regression | Apply consistently and log | Over-cleaning | Cleaned signals + component logs | Improved SNR | No EOG channels → ICA heuristics |
| 9) Baseline correction | Adjust relative to baseline period | Subtract pre-stimulus mean, z-score vs baseline | Use only within-subject stats | Using test set baseline | Baseline-corrected epochs | Reduced between-session bias | No baseline → skip |
| 10) Bad channel handling | Identify and treat bad channels | Flatline detection, low correlation, interpolate | Track dropped/interpolated | Quietly filling everything | Bad-channel report | <X% channels repaired | Many bad → exclude recording |
| 11) Window extraction (post-clean) | Cut into fixed windows | Use phase-2 window config | Extract after artifact logic | Changing windowing while comparing | Final windows dataset | Window count stable | Event boundary → buffer |
| 12) Preprocessing reproducibility | Make pipeline deterministic | Config files, fixed filter order, seed for ICA | Store every parameter | "Manual tweaks" | Preproc config + pipeline | Same input → same output hash | Library differences → pin versions |

### Recommended Preprocessing Recipes

| Use Case | Minimal Safe Recipe |
|----------|---------------------|
| Emotion/Stress | Re-ref (CAR) → notch 60 → bandpass 1-40 → artifact mark |
| Motor Imagery | Re-ref → notch 60 → bandpass 8-30 (mu/beta) + 0.5-45 → epoch |
| Clinical Seizure | Re-ref → notch → bandpass 0.5-70 → careful artifact labeling |

---

# PHASE 4: Standardization + Normalization (Leakage-Safe)

| Step | What You Do | Options | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|---------|------|--------|--------|---------------|------------|
| 1) Decide what to normalize | Choose representation | Raw time-series, bandpower, STFT/CWT, embeddings | Normalize after splits defined | Compute global stats using all data | Normalization design note | Split leakage check = pass | Multi-dataset → normalize per-dataset |
| 2) Choose normalization scope | Define where stats come from | Train-only global, per-subject, per-session, per-window | Default: train-only global | Using test set mean/std | norm_stats_train.json | Reproducible stats hash | Session drift → per-session normalization |
| 3) Time-series scaling | Scale amplitude | Z-score, robust (median/IQR), min-max | Use robust scaling if artifacts | Min-max with outliers | Scaled tensors | Stable training loss | Heavy-tailed noise → robust scaler |
| 4) Channel-wise vs sample-wise | Pick stats scope | Channel-wise z-score (recommended), global | Channel-wise for multi-channel | Mixing channels into one scaler | Channel-wise scaler stats | Reduced channel bias | Different montages → per channel only |
| 5) Per-subject normalization | Reduce subject variability | Normalize within each subject | Use only for within-subject tasks | Using to hide generalization gap | Ablation experiment results | Report change in LOSO | Cross-subject → avoid relying on this |
| 6) Per-window normalization | Normalize each window independently | Subtract window mean; divide by std | Good for removing slow drift | Can erase amplitude biomarkers | Window-normalized version | Compare with/without | Amplitude predictive → keep raw scale |
| 7) Log transforms for power | Stabilize variance | log10(power + ε), dB scaling | Always use ε to avoid log(0) | Logging raw negative values | Feature transform spec | Reduced skewness/kurtosis | Zero/near-zero → increase ε |
| 8) Image normalization | Make spectrogram consistent | Per-image min-max, per-dataset z-score | For CNN: per-frequency-bin z-score | Per-image only when doing clinical | Normalized TFR images | Improved calibration | Different FFT params → recompute |
| 9) Dataset standardization | Align across devices/datasets | Resample Fs, channel subset, consistent bandpass | Keep a "compatibility layer" | Pretending datasets identical | Harmonized dataset v1 | Cross-dataset baseline exists | Small overlap → montage-agnostic features |
| 10) Leakage-safe computation | Compute mean/std training only | Fit scaler on train; apply to val/test | Lock scaler after training | Re-fitting on val/test | Saved scaler object | Re-run gives identical outputs | CV → fit scaler inside each fold |
| 11) Normalization QA | Prove normalization behaves | Check mean≈0 std≈1 on train | Track per-channel histograms | Only eyeballing plots | Normalization QA report | No abnormal distribution shift | Weird spikes → bad channel not removed |
| 12) Versioned data views | Keep multiple variants | raw, zscore_global, robust_global, per_window, logpower | Name variants clearly | Overwriting files | Dataset registry | Each view reproducible | Confusion → naming convention + manifest |

### Practical Defaults

| Representation | Default Normalization |
|----------------|----------------------|
| Raw time-series → deep model | Channel-wise z-score (train-only) |
| Bandpower features | log(power+ε) then z-score (train-only) |
| STFT/CWT images | Per-frequency-bin z-score (train-only) |

---

# PHASE 5: EDA + Feature Evaluation (Leakage-Safe)

| Step | What You Do | Techniques | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|------------|------|--------|--------|---------------|------------|
| 1) EDA scope & split lock | Freeze train/val/test before analysis | Subject-wise split locked | Do EDA on train only | Peeking at test labels | EDA protocol note | Leakage checklist = pass | Test inspected → discard and redo |
| 2) Signal quality overview | Quantify basic EEG health | SNR proxy, RMS amplitude, PSD slope, SQI | Report distributions | Ignoring noisy channels | SQI report | %windows passing SQI ≥ threshold | Persistent noise → revisit preprocessing |
| 3) Time-domain exploration | Understand waveform behavior | Mean, variance, skew/kurtosis, Hjorth | Compare class-wise on train | Overinterpreting single subject | Time-domain summary | Stable stats across folds | Heavy tails → robust stats |
| 4) Frequency-domain exploration | Check band relevance | Bandpower (δ θ α β γ), relative power, PSD peaks | Tie bands to neuroscience | Fishing for significance | Bandpower plots + tables | Expected band trends visible | No trend → re-check task alignment |
| 5) Time-frequency EDA | Validate TFR usefulness | STFT/CWT average maps per class | Average over windows & subjects | Showing cherry-picked images | Mean/variance TFR figures | Clear structural differences | Blurry maps → window/wavelet mismatch |
| 6) Spatial/channel EDA | See where information lives | Channel-wise bandpower maps, correlation matrices | Keep montage consistent | Mixing channel orders | Channel importance heatmaps | Consistent hotspots across folds | Device-specific → harmonize |
| 7) Class separability (univariate) | Measure feature discrimination | Effect size (Cohen's d), AUC per feature, KS-test | Prefer effect size over p-value | p-value hunting | Feature ranking table | Top features d ≥ 0.5 | Weak effects → consider interactions |
| 8) Class separability (multivariate) | Check combined feature power | LDA projection, PCA + class coloring, t-SNE/UMAP | Use for diagnostics, not claims | Claiming from t-SNE | Projection plots | Partial separation visible | No separation → nonlinear capacity |
| 9) Redundancy analysis | Identify correlated features | Pearson/Spearman corr, mutual info, VIF | Remove or group redundant | Blindly keeping all | Correlation matrix + clusters | Max corr below threshold | Strong collinearity → PCA |
| 10) Stability across subjects | Ensure features generalize | Feature mean/variance per subject, ICC | Favor stable features | Features driven by few subjects | Stability report | Low between-subject variance | Subject-specific → personalization track |
| 11) Leakage detection (EDA) | Detect leakage proactively | Check if simple classifier on ID-like features works | Intentionally test for leaks | Ignoring suspiciously high AUC | Leakage audit | Dummy models ≈ chance | High dummy AUC → revisit split |
| 12) Feature readiness decision | Decide which features move forward | Keep interpretable + stable + discriminative | Document why kept/dropped | "We kept everything" | Feature shortlist v1 | Shortlist size justified | Too many → prioritize by effect |

### EEG Feature Families

| Family | Examples | Why Evaluate |
|--------|----------|--------------|
| Time-domain | RMS, Hjorth, zero-crossings | Fast, interpretable |
| Frequency-domain | Bandpower, peak freq, ratios | Neuroscience-aligned |
| Time-frequency | Energy in TFR tiles | Captures nonstationarity |
| Connectivity | Coherence, PLV | Network-level info |
| Riemannian | Covariance geometry | Montage-robust, strong baseline |
| Image-derived | CNN embeddings from STFT/CWT | High-capacity representation |

---

# PHASE 6: Feature Selection & Dimensionality Reduction

| Step | What You Do | Methods | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|---------|------|--------|--------|---------------|------------|
| 1) Selection objective | Define why you reduce features | Improve generalization, reduce overfitting, speed, interpretability | Tie objective to metric | Reducing "just because" | Feature-selection goal doc | Objective aligned to metric | Conflicting goals → parallel tracks |
| 2) Selection scope | Decide eligible features | Handcrafted only, image-embeddings only, hybrid | Start with Phase-5 shortlist | Feeding raw everything | Candidate feature set | Candidate size justified | Too many → pre-filter |
| 3) Filter methods (fast) | Rank features independently | Variance threshold, ANOVA/F-test, mutual info, effect size | Use train-only stats | Using labels from val/test | Ranked feature list | Top-K improves baseline | Nonlinear → MI instead of ANOVA |
| 4) Correlation pruning | Remove redundancy | Pearson/Spearman threshold, clustering, VIF | Keep one per cluster | Keeping many correlated | Pruned feature set | Max corr ≤ threshold | Strong domain reason → keep + regularize |
| 5) Wrapper methods | Evaluate subsets using model | RFE (SVM/LR), sequential forward selection | Use nested CV | Wrapper on full dataset | Wrapper-selected subset | Stable subset across folds | Small N → wrappers unstable |
| 6) Embedded methods | Let model select during training | L1/Lasso, Elastic Net, tree importance | Prefer for linear/tree baselines | Overinterpreting single run | Embedded-selected features | Consistency across seeds | High variance → average over runs |
| 7) Stability selection | Check robustness | Bootstrapping + selection frequency | Keep high selection freq | One-shot selection | Stability table | Selection freq ≥ 70% | Unstable → relax K or increase data |
| 8) Dimensionality reduction (linear) | Compress preserving variance | PCA, ICA, CSP (MI tasks) | Fit on train only | Using all data | Component models | Explained variance ≥ target | PCA hurts interpretability → keep raw |
| 9) Dimensionality reduction (manifold) | Capture nonlinear structure | Autoencoders, kernel PCA | Use cautiously; report clearly | Claiming interpretability | Latent embeddings | Downstream metric improves | Overfitting → regularize |
| 10) Riemannian geometry | Leverage covariance structure | Covariance matrices → tangent space | Strong baseline for EEG | Mixing with incompatible features | Riemannian feature set | Competitive baseline achieved | Few channels → regularize |
| 11) Hybrid feature strategy | Combine complementary features | Bandpower + Riemannian, TFR-CNN + stats | Keep combinations small | Feature explosion | Hybrid feature schema | Hybrid > best single family | No gain → drop weakest |
| 12) Ablation studies | Prove contribution | Remove one family at a time | Mandatory for papers | Skipping ablations | Ablation table/plot | Performance drops as expected | No drop → feature redundant |
| 13) Leakage guardrails | Enforce safe fitting | Fit selectors inside CV folds | Lock pipeline order | Preselecting before split | Pipeline diagram | No optimistic bias | Suspected bias → nested CV |
| 14) Final feature freeze | Freeze features for modeling | Versioned feature list | Freeze before heavy tuning | Changing features mid-tuning | Feature set vFinal | Hash matches across runs | New idea → new experiment ID |

### Practical Defaults

| Scenario | Recommended Approach |
|----------|---------------------|
| Small N, handcrafted features | Filter (effect size/MI) → corr prune → L1 |
| Motor imagery EEG | CSP + bandpower → LDA/SVM |
| Cross-subject generalization | Riemannian tangent features |
| EEG → image → deep model | Use embeddings; avoid heavy manual selection |

---

# PHASE 7: Model Training

| Step | What You Do | Options | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|---------|------|--------|--------|---------------|------------|
| 1) Build baselines first | Train simple models before DL | LR, SVM, RF, XGBoost, LDA | Establish a "floor" performance | Jumping to deep model first | Baseline results table | Baseline stable across seeds | Baseline too low → check labels/preproc |
| 2) Define pipelines | Make one pipeline per family | preproc → norm → features → model | Put everything in one pipeline | Manual steps outside | Pipeline diagram + code | Reproducible run hash | Mismatch across runs → config locking |
| 3) Handle class imbalance | Choose imbalance strategy | Class weights, focal loss, balanced sampling | Use PR-AUC/F1 if imbalance | Reporting only accuracy | Imbalance strategy note | Minority recall meets target | Extreme → event-based metrics |
| 4) Choose representation | Decide model input type | Raw EEG (1D CNN), TFR images (2D CNN), features | Compare 2-3 representations max | Trying 10 at once | Representation benchmark | Best rep selected on val | No gain → revisit feature quality |
| 5) Model families | Select candidates | 1D CNN, EEGNet, TCN, BiLSTM, CNN+Attention, ViT | Pick models aligned to data size | Huge ViT with small dataset | Model shortlist | Params vs N justified | Small N → compact EEGNet/TCN |
| 6) Regularization strategy | Prevent overfitting | Dropout, weight decay, early stopping, augmentation | Use early stopping + weight decay | Training until loss → 0 | Training config | Train-val gap controlled | Overfit → increase reg |
| 7) Data augmentation | Improve robustness | Gaussian noise, time shift, channel dropout, mixup | Keep physiologically plausible | Augmenting that changes labels | Augmentation config | Val improves without instability | Aug hurts → ablation |
| 8) Hyperparameter search | Tune fairly | Random search, Bayesian, small grid | Use nested CV or held-out val | Tuning on test set | HPO log | Limited search budget | Too many trials → val overfit |
| 9) Training reproducibility | Make results repeatable | Fixed seeds, deterministic ops, pinned libs | Log everything | "It changed this time" | Experiment log | Std dev across 5 seeds acceptable | High variance → simplify model |
| 10) Calibration strategy | Make probabilities meaningful | Temperature scaling, Platt scaling, isotonic | Calibrate on validation only | Calibrating on test | Calibration report | ECE/Brier improves | Overconfident → calibration + reg |
| 11) Threshold selection | Choose decision threshold | Max F1, fixed sensitivity, Youden J | Select threshold on val | Picking after seeing test | Threshold rule | Meets clinical constraint | Drift → re-tune on new val |
| 12) Training efficiency | Control compute cost | Mixed precision, batch sizing, caching | Keep logs of compute | HPC "just because" | Resource log | Training time within budget | Memory errors → reduce batch |
| 13) Model selection rule | Decide what "best model" means | Primary metric + tie-breakers | Use consistent criteria | Picking best-looking run | Model selection spec | Best chosen without bias | Multiple winners → choose simplest |
| 14) Save artifacts | Package everything | Weights, configs, scaler, feature list | Save as "model bundle" | Saving only weights | Model bundle v1 | Loads & predicts deterministically | Bundle missing stats → checklist |

### Training Ladder

| Stage | What to Train | Why |
|-------|---------------|-----|
| A | Handcrafted features + LR/SVM | Fast sanity + interpretability |
| B | Riemannian tangent + LR/SVM | Strong EEG baseline |
| C | 1D CNN on raw EEG | Learns morphology |
| D | 2D CNN/ViT on CWT scalograms | Often top accuracy if data supports |

---

# PHASE 8: Model Validation

| Step | What You Do | Methods | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|---------|------|--------|--------|---------------|------------|
| 1) Lock validation protocol | Freeze how you validate | Subject-wise CV, LOSO, session-wise, nested CV | Write protocol in advance | Changing protocol to boost score | Validation protocol doc | Protocol unchanged | If changed → report as new experiment |
| 2) Choose CV scheme | Match CV to deployment | LOSO (strong), GroupKFold by subject | Use group-based folds | Random KFold on windows | Fold assignment file | No subject leakage | Few subjects → LOSO + CI |
| 3) Nested CV (for tuning) | Prevent val overfit during HPO | Inner loop tuning, outer loop scoring | Required for serious claims | Tuning on same fold you report | Nested CV results | Outer-fold performance stable | Too expensive → small random search |
| 4) Confidence intervals | Quantify uncertainty | Bootstrap CI, fold std, Bayesian CI | Report CI for key metrics | Reporting only best run | CI table | CI width acceptable | Wide CI → more data or simpler model |
| 5) Robustness checks | Validate under noise/variation | Add noise, missing channels, time shift, reduced Fs | Run stress tests systematically | Only testing "clean" data | Robustness report | Drop within tolerance | Large drop → add augmentation |
| 6) Stratified analysis | Evaluate per subgroup | Per-subject, per-session, per-class, per-device | Identify failure modes early | Hiding poor subgroup | Stratified metrics dashboard | Worst-case ≥ minimum threshold | Device bias → domain adaptation |
| 7) Error analysis | Study what model gets wrong | Confusion matrix, per-class errors, mislabel audit | Sample errors for review | Blaming data without checking | Error log + examples | Clear dominant error sources | Label noise → refine rules |
| 8) Calibration validation | Validate probability quality | Reliability curve, ECE, Brier score | Validate on held-out | Using training to calibrate | Calibration report | ECE improves | Overconfidence → temperature scaling |
| 9) Decision-threshold validation | Validate operating point | Sensitivity at fixed specificity, F1 max | Select threshold on val only | Choosing threshold after test | Operating point report | Constraint met | Different population → threshold may shift |
| 10) Leakage audit (again) | Detect "too good to be true" | Train on shuffled labels, ID-prediction tests | Run sanity baselines | Skipping sanity checks | Sanity-check appendix | Shuffled-label ≈ chance | If not → pipeline leakage exists |
| 11) Reproducibility validation | Confirm results hold across seeds | 5-10 seeds, rerun key experiments | Report mean±std | Reporting only best seed | Repro table | Std dev within tolerance | High variance → smaller model |
| 12) External validation | Test on different dataset/device | Cross-dataset test, leave-one-dataset-out | Strongest evidence | Claiming generalization without | External test report | Drop acceptable & explained | Large domain shift → domain adaptation |
| 13) Ablation validation | Prove which components matter | Remove preprocessing, feature family, swap model | Keep ablation list small | Too many ablations | Ablation table | Contributions consistent | No effect → simplify pipeline |
| 14) Validation sign-off | Decide if ready for final test | Checklist + thresholds | Use gate to stop infinite tuning | "One more tweak" cycle | Validation sign-off | All gates pass | If fail → return to Phase 6/7 |

### Best Validation Recipe

| Scenario | Recommended Validation |
|----------|------------------------|
| Cross-subject deployment | GroupKFold by subject or LOSO + bootstrap CI |
| Small dataset | LOSO + nested CV (small HPO budget) |
| Multi-session per subject | Group by subject (keep all sessions together) |
| Multi-device | External validation per device or leave-one-device-out |

---

# PHASE 9: Model Testing + Accuracy Reporting

| Step | What You Do | Methods | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|---------|------|--------|--------|---------------|------------|
| 1) Freeze everything | Lock code, data, preprocessing, normalization, threshold | Git tag, dataset hash, model bundle | Treat test like production | Changing pipeline after seeing test | Test freeze checklist | Hashes match | If bug → declare new test run |
| 2) One-time test execution | Run inference on test set once | Deterministic inference script | Keep logs + timestamps | Re-running many times | Test run log | Same output with same seed | Non-determinism → set flags |
| 3) Report primary metrics | Compute agreed metrics | Accuracy, F1, AUC, PR-AUC, sensitivity/specificity | Use metrics aligned to imbalance | Reporting only accuracy | Test metrics table | Primary metric meets target | Imbalance → focus PR-AUC |
| 4) Confusion matrix + per-class | Show what fails and how | Confusion matrix, per-class precision/recall | Always include per-class | Only macro averages | Error breakdown table | Worst-class recall above floor | One class collapses → threshold tuning |
| 5) Confidence intervals on test | Quantify uncertainty | Bootstrap CI, Wilson CI, DeLong CI | Report CI beside score | Single number only | CI report | CI width acceptable | CI too wide → needs more subjects |
| 6) Statistical comparison | Prove improvements not noise | McNemar, paired bootstrap, DeLong | Compare against best baseline | Comparing to weak baselines only | Significance table | Improvement statistically supported | Small N → paired bootstrap |
| 7) Benchmarking table | Compare fairly with prior work | Same dataset split, same metric definitions | Match protocols or disclose | Claiming SOTA without same split | Benchmark matrix | Apples-to-apples comparisons | Different preprocessing → note |
| 8) Robustness on test | Run predefined stress tests | Noise, missing channels, downsample | Predefine tests in Phase 8 | Inventing new tests after results | Test robustness appendix | Drop within tolerance | Big drop → document |
| 9) Calibration on test | Check probability quality | ECE, reliability curve, Brier | Report but don't recalibrate | Fitting calibration on test | Calibration figure/table | Calibration acceptable | Poor → note; recalibrate in next version |
| 10) Failure-mode audit | Inspect top FP/FN | Manual review, label audit sampling | Identify if labels or model wrong | Hand-wavy explanations | Failure audit notes | Clear dominant causes | Label errors → document |
| 11) Repro pack | Package everything | Scripts, configs, model card, data card | Make it runnable end-to-end | Missing seeds/configs | Repro bundle | Another machine reproduces | Dependency drift → container |
| 12) Go/No-Go decision | Decide deployment readiness | Criteria: performance + robustness + risk | Use decision matrix | Deploying because score is high | Release decision doc | All gates pass | If not → iterate Phase 3-8 |

### Accuracy Reporting

| Situation | What to Report |
|-----------|----------------|
| Balanced classes | Accuracy + macro F1 + CI |
| Imbalanced (common) | PR-AUC + macro F1 + sensitivity/specificity + CI |
| Clinical event detection | Sensitivity at fixed false alarm rate |
| Cross-subject generalization | Subject-wise results + mean±CI |

---

# PHASE 10: End-to-End Benchmarking + Reporting

| Step | What You Do | What to Include | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|-----------------|------|--------|--------|---------------|------------|
| 1) Build benchmark ladder | Define model set representing increasing sophistication | Baseline LR/SVM → Riemannian → 1D CNN → TFR-CNN/ViT | Keep ladder to 4-6 models | 20 models with no story | Benchmark plan | Ladder covers key families | Limited time → keep 3 strongest |
| 2) Standardize evaluation protocol | Ensure every model uses same split + metrics | Same folds, same grouping, same metrics | Make apples-to-apples | Changing split per model | Evaluation harness | Identical folds across runs | Different input → keep same split |
| 3) Single source of truth | Store results in one file | CSV/JSON with model, seed, fold, metrics | Track every run | Manual copy/paste | Results registry | No missing runs | Inconsistent → enforce schema |
| 4) Primary results table | Summarize key metrics | Mean±CI, macro F1, PR-AUC, sensitivity/specificity | Put best model bold; report CI | Showing only best seed | Main results table | CI included | CI too wide → explain data limits |
| 5) Baseline comparison table | Show improvement over baselines | Δ vs Riemannian + Δ vs best classical | Use paired stats if possible | Comparing only weak baseline | Baseline delta table | Improvements significant | No significance → phrase as "trend" |
| 6) Ablation table | Prove contribution | Remove: notch, ICA, per-freq norm, feature family | 5-8 ablations max | 30 ablations clutter | Ablation table | Expected drops occur | No drop → simplify |
| 7) Robustness table | Show behavior under stress | Noise, missing channels, resample, artifact-heavy | Use predefined tests | Designing tests after results | Robustness table | Drop within tolerance | Large drop → mitigation plan |
| 8) Generalization evidence | Show cross-subject and cross-dataset | LOSO / GroupKFold + external test | External test if available | Claiming generalization without | Generalization table | External results reported | Domain shift → add "future work" |
| 9) Error analysis pack | Make failure modes concrete | Confusion matrix, top FP/FN, label audit | Categorize errors | Hand-wavy "data is noisy" | Error analysis section | Top 3 failure modes | Label noise → revise rules |
| 10) Explainability pack | Provide interpretable evidence | Bandpower importance, SHAP, Grad-CAM | Use as support, not proof | Overclaiming causality | Explainability figures | Coherent patterns | Unstable maps → average across folds |
| 11) Efficiency + deployment metrics | Report compute/latency footprint | Inference time, model size, memory, throughput | Include for edge use | Ignoring runtime | Efficiency table | Meets latency target | Too slow → quantize |
| 12) Reproducibility & artifacts | Provide everything to reproduce | Data card, model card, configs, seeds | Make it runnable | Missing versions | Repro bundle | Re-run matches | Dependency drift → containerize |
| 13) Compliance/trust reporting | Add risk, privacy, monitoring | Risk register, bias checks, drift plan | Align to Responsible AI | Skipping governance | Governance appendix | Checklist complete | Sensitive data → de-ID |
| 14) Final narrative | Write logic: problem → pipeline → evidence → limits | Clear contributions | Tie every table/figure to claim | Random figures | Final report structure | Each claim supported | Too many claims → reduce to 3-5 |

---

# PHASE 11: Production/Pilot Deployment

| Step | What You Do | Options | Do's | Don'ts | Output | Quality Gates | Edge Cases |
|------|-------------|---------|------|--------|--------|---------------|------------|
| 1) Deployment context | Define where model runs | Edge, mobile, cloud, hybrid | Match latency & privacy needs | One-size-fits-all | Deployment decision doc | Latency & privacy targets met | Edge too slow → compression |
| 2) Inference pipeline | Freeze runtime pipeline | Same preproc + norm + features as training | Mirror training exactly | "Light" runtime shortcuts | Inference pipeline spec | Output parity vs offline | Drift due to missing step → enforce hash |
| 3) Input data validation | Validate EEG before inference | Signal range checks, missing channel check, SQI gate | Reject bad input early | Predicting on garbage | Input validator | %rejected within expected | Too many rejects → relax SQI |
| 4) Output post-processing | Make predictions usable | Thresholding, smoothing over time, hysteresis | Align with clinical objective | Raw frame-by-frame decisions | Decision logic spec | Stable decisions over time | Flicker → temporal smoothing |
| 5) Runtime monitoring (model) | Track model behavior | Prediction distribution, confidence, latency | Log silently & continuously | Only monitoring accuracy | Monitoring dashboard | No silent failures | Sudden confidence spike → investigate |
| 6) Runtime monitoring (data) | Detect input drift | PSD drift, bandpower stats, KL divergence | Compare to train distribution | Ignoring slow drift | Data drift alerts | Drift within tolerance | New device → re-baseline |
| 7) Performance feedback loop | Collect weak/strong labels | Human review, delayed outcomes, proxy labels | Separate training vs audit data | Training on noisy feedback blindly | Feedback dataset | Label quality tracked | Label noise → consensus |
| 8) Drift detection policy | Decide when model is outdated | Thresholds on data drift + confidence + error | Define triggers upfront | Ad-hoc retraining | Drift policy doc | Trigger rules tested | Frequent triggers → widen tolerance |
| 9) Retraining cadence | Plan how often to update | Time-based, event-based, hybrid | Retrain on versioned data | Continuous retrain without review | Retraining schedule | Retrain improves val | No improvement → stop & analyze |
| 10) Model update validation | Validate new model before release | Shadow deployment, A/B test, offline replay | Compare against current | Replacing without comparison | Model vNext validation | vNext ≥ vCurrent | Regression → rollback |
| 11) Rollback strategy | Ensure safe fallback | Keep last stable model live | Instant rollback capability | One-way upgrades | Rollback plan | Rollback tested | Corrupt update → auto-revert |
| 12) Explainability in production | Provide interpretable signals | Feature importance summaries, saliency | Use for audit & trust | Real-time heavy explainability | Explainability log | Stable patterns | Noisy explanations → aggregate |
| 13) Security & privacy | Protect EEG & predictions | Encryption, access control, anonymization | Least-privilege access | Storing raw IDs | Security checklist | Compliance met | Breach risk → tokenization |
| 14) Compliance & audit | Enable traceability | Model cards, data lineage, decision logs | Audit-ready by design | Retroactive documentation | Audit trail | Audit passes | Missing logs → block deployment |
| 15) KPI & ROI tracking | Measure real-world value | Accuracy proxy, recall@risk, latency, cost | Tie metrics to stakeholders | Only technical KPIs | KPI dashboard | ROI hypothesis tested | No value → revisit use-case |
| 16) Decommissioning plan | Decide when to retire | Performance decay, replaced by better model | Plan end-of-life | Zombie models | Decommission checklist | Clean shutdown | Forgotten model → governance failure |

### Production KPIs

| Category | Example KPIs |
|----------|--------------|
| Data health | %valid windows, SQI trend, drift score |
| Model behavior | Mean confidence, alert rate, latency |
| Outcome | Sensitivity@target, false alert rate |
| Trust | Explainability stability, audit success |
| Ops | Uptime, rollback success, retrain frequency |

---

# PERFORMANCE METRICS REFERENCE

## Classification Metrics

| No. | Metric | What Is Analyzed | Interpretation |
|-----|--------|------------------|----------------|
| 1 | Accuracy | Correct predictions / total | Overall correctness |
| 2 | Precision | TP / (TP + FP) | Prediction exactness |
| 3 | Recall (Sensitivity) | TP / (TP + FN) | Detection capability |
| 4 | F1-Score | 2 * (Precision * Recall) / (Precision + Recall) | Classification robustness |
| 5 | Specificity | TN / (TN + FP) | Negative detection reliability |
| 6 | Error Rate | Incorrect / total | Misclassification tendency |
| 7 | Confusion Matrix | TP, FP, TN, FN distribution | Error patterns |
| 8 | ROC Curve | TPR vs FPR | Discriminative power |
| 9 | AUC | Area under ROC | Model separability |
| 10 | PR-AUC | Area under Precision-Recall | Imbalanced data performance |

## System Performance Metrics

| No. | Metric | What Is Analyzed | Interpretation |
|-----|--------|------------------|----------------|
| 11 | Latency | Time per operation | System responsiveness |
| 12 | Throughput | Tasks per unit time | Processing capacity |
| 13 | Memory Utilization | Memory consumed | Resource efficiency |
| 14 | CPU/GPU Utilization | Processing unit workload | Hardware efficiency |
| 15 | Energy Consumption | Power usage | Energy efficiency |
| 16 | Model Size | Storage requirement | Deployment feasibility |

---

# MODEL ANALYSIS TYPES

| No. | Analysis Type | What Is Analyzed | Purpose |
|-----|---------------|------------------|---------|
| 1 | Architecture Analysis | Model structure and layers | Design effectiveness |
| 2 | Parameter Analysis | Number of trainable parameters | Model complexity |
| 3 | Convergence Analysis | Loss stabilization over epochs | Training stability |
| 4 | Overfitting Analysis | Train-test performance gap | Generalization quality |
| 5 | Bias-Variance Analysis | Error decomposition | Trade-off balance |
| 6 | Feature Dependency Analysis | Model reliance on features | Input importance |
| 7 | Ablation Analysis | Effect of removing components | Component contribution |
| 8 | Robustness Analysis | Behavior under noisy inputs | Model resilience |
| 9 | Generalization Analysis | Performance on unseen data | Real-world applicability |
| 10 | Interpretability Analysis | Decision transparency | Model explainability |
| 11 | Calibration Analysis | Probability correctness | Confidence reliability |
| 12 | Error Distribution Analysis | Error patterns across samples | Failure mode identification |

---

# CLINICAL VALIDATION MATRIX

| No. | Main Analysis | Sub-Analysis | What Is Validated | Metric |
|-----|---------------|--------------|-------------------|--------|
| 1 | Diagnostic Performance | Sensitivity Analysis | True condition detection | Sensitivity (%) |
| | | Specificity Analysis | Healthy exclusion accuracy | Specificity (%) |
| | | Accuracy Analysis | Overall correctness | Accuracy (%) |
| | | AUC Analysis | Diagnostic separability | AUC |
| 2 | Agreement Analysis | Model vs Clinician | Clinical concordance | Cohen's Kappa |
| | | Inter-Rater | Human labeling consistency | Kappa / ICC |
| 3 | Clinical Risk | False-Negative Risk | Missed clinical cases | FN Rate |
| | | False-Positive Risk | Over-diagnosis | FP Rate |
| | | Risk Stratification | Severity classification | Risk Score |
| 4 | Population Validation | Age-Group Analysis | Performance across ages | Mean F1 |
| | | Gender-Wise Analysis | Gender bias detection | Δ Accuracy |
| | | Comorbidity Analysis | Co-conditions handling | Subgroup Score |
| 5 | Subject-Wise Clinical | Patient-Wise Performance | Individual reliability | Patient Score |
| | | LOSO Clinical | Unseen patient generalization | Mean F1 / AUC |
| 6 | Clinical Robustness | Signal/Image Noise | Real-world data quality | Robustness Score |
| | | Artifact Resistance | Motion/physiological artifacts | Performance Drop (%) |

---

# RELIABILITY MATRIX

| No. | Main Analysis | Sub-Analysis | What Is Assessed | Metric |
|-----|---------------|--------------|------------------|--------|
| 1 | Test-Retest Reliability | Short-Term Retest | Consistency across repeated trials | ICC |
| | | Long-Term Retest | Stability over time | ICC |
| | | Session Gap Analysis | Temporal drift impact | Correlation (r) |
| 2 | Inter-Rater Agreement | Model vs Clinician | Agreement with expert | Cohen's Kappa |
| | | Clinician-Clinician | Human reliability baseline | Kappa / ICC |
| | | Multi-Rater Consensus | Consistency across raters | Fleiss' Kappa |
| 3 | Internal Consistency | Feature Consistency | Coherence of measures | Cronbach's Alpha |
| | | Channel/Sensor Consistency | Signal agreement | Alpha / Mean Corr |
| 4 | Cross-Session Stability | Session-Wise Performance | Stability across sessions | Δ F1 / Δ AUC |
| | | Day-Wise Consistency | Longitudinal robustness | Std. Deviation |
| 5 | Robustness Testing | Perturbation Robustness | Small input variations | Robustness Score |
| | | Stress-Case Testing | Extreme conditions | Performance Drop (%) |
| 6 | Noise Tolerance | Gaussian/Real-World Noise | Noise immunity | SNR-based Score |
| | | Low-Quality Signal Test | Degraded input handling | F1 Degradation |
| 7 | Artifact Resistance | Motion Artifacts | Movement noise resistance | Artifact Score |
| | | Physiological Artifacts | EMG/EOG robustness | Accuracy Drop |
| | | Pre vs Post Cleaning | Artifact removal benefit | Score Gain |
| 8 | Domain Shift Reliability | Lab → Real World | Environmental transferability | AUC Drop |
| | | Device/Sensor Shift | Hardware variability | Performance Gap |

### Reliability Thresholds

| Measure | Acceptable Standard |
|---------|---------------------|
| ICC | ≥ 0.75 (Good), ≥ 0.90 (Excellent) |
| Cohen's Kappa | ≥ 0.60 (Substantial) |
| Cronbach's Alpha | ≥ 0.70 |
| Cross-Session Δ F1 | ≤ 5% |
| Noise-Induced Drop | ≤ 10% |
| Artifact Impact | Minimal / recoverable |

---

# SUBJECT-WISE CROSS-VALIDATION

## Performance Table Example

| Subject ID | Accuracy (%) | Precision | Recall | F1-Score | AUC | Composite Score | Observation |
|------------|--------------|-----------|--------|----------|-----|-----------------|-------------|
| Subject-1 | 90.8 | 0.89 | 0.91 | 0.90 | 0.94 | 0.92 | Stable generalization |
| Subject-2 | 87.6 | 0.86 | 0.88 | 0.87 | 0.92 | 0.89 | Minor recall degradation |
| Subject-3 | 92.4 | 0.91 | 0.93 | 0.92 | 0.95 | 0.94 | Strong compatibility |
| Subject-4 | 84.9 | 0.83 | 0.85 | 0.84 | 0.90 | 0.87 | High subject variability |
| Subject-5 | 89.7 | 0.88 | 0.90 | 0.89 | 0.93 | 0.91 | Balanced performance |
| **Average** | **89.1** | **0.87** | **0.89** | **0.88** | **0.93** | **0.91** | **Robust overall** |

### Composite Score Formula

```
Composite Score = α · F1 + β · AUC
where α + β = 1 (typically α = 0.5, β = 0.5)
```

---

# JOB SCHEDULING FRAMEWORK

## Phase-by-Phase Job List

| Phase | Job ID | What It Does | Depends On | Job Count |
|-------|--------|--------------|------------|-----------|
| 2 | P2_ingest_manifest | Load raw files, build metadata | P1 | D×K |
| 2 | P2_split_make | Create leakage-safe splits | P2_ingest | D×K |
| 3 | P3_preprocess | Re-ref, notch, bandpass, artifact mask | P2_split | D×K |
| 3 | P3_qc_report | PSD before/after, SQI stats | P3_preprocess | D×K |
| 4 | P4_norm_fit | Fit train-only normalization stats | P3_preprocess | D×K×F |
| 4 | P4_apply_norm | Apply normalization to partitions | P4_norm_fit | D×K×F |
| 5 | P5_eda_trainonly | Train-only EDA: distributions, bandpower | P4_apply_norm | D×K |
| 5 | P5_feature_eval | Effect sizes, MI, separability, stability | P4_apply_norm | D×K×F |
| 6 | P6_feature_select | Filter/wrapper/embedded selection | P5_feature_eval | D×K×F |
| 6 | P6_feature_extract | Materialize features per representation | P4_apply_norm | D×K×R |
| 7 | P7_train | Model training (with HPO) | P6_feature_extract | D×K×R×M×F×S |
| 8 | P8_validate | Validation metrics, CI, calibration | P7_train | D×K×R×M×F×S |
| 9 | P9_test_final | One-time holdout test | P8_validate | D×K |
| 10 | P10_benchmark_pack | Build final tables/figures | P8 + P9 | D |
| 11 | P11_deploy_pack | Package inference pipeline | P9 + P10 | D |

### Suggested Settings

**For thesis-grade plan:**
- K = 2 datasets per disease
- R = 2 reps (Riemannian + CWT-image)
- M = 3 models per rep
- F = 5 folds
- S = 3 seeds

**Training jobs:** P7_train jobs = 5×2×2×3×5×3 = **900 jobs**

---

# FEATURE ENGINEERING

## Time-Domain Features

| Group | Features | Output |
|-------|----------|--------|
| Temporal statistics | mean, variance, std, RMS, skewness, kurtosis | feature vector |
| Signal dynamics | zero-crossing rate, slope sign changes, Hjorth (activity, mobility, complexity) | feature vector |
| Complexity | entropy (sample/approx/permutation), fractal dimension | feature vector |

## Spatial Features

| Group | Features | Output |
|-------|----------|--------|
| Channel topology | electrode neighborhood aggregation | spatial embedding |
| Connectivity | correlation, coherence, PLV, mutual information | adjacency / graph |
| Region-wise pooling | frontal/parietal/temporal band pooling | region features |

## Frequency Bands

| Band | Range | Significance |
|------|-------|--------------|
| Delta (δ) | 0.5-4 Hz | Deep sleep, pathology |
| Theta (θ) | 4-8 Hz | Drowsiness, memory |
| Alpha (α) | 8-13 Hz | Relaxation, attention |
| Beta (β) | 13-30 Hz | Active cognition |
| Gamma (γ) | 30-50 Hz | Higher processing |

---

# MANDATORY RESULT CHARTS

## Required Visualizations

| Chart Type | Use Case |
|------------|----------|
| **Pie Chart** | Class distribution, subject distribution, artifact types proportion |
| **Bar Chart** | Model vs baselines (F1, AUC), ablation scores, per-subject performance |
| **Heatmap** | Confusion matrix, feature importance (channels × bands), subject-wise scores |
| **Line Chart** | Training/validation loss curves, robustness degradation curves |
| **ROC Curve** | Binary ROC + AUC, Multi-class One-vs-Rest ROC |
| **Box Plot** | Cross-fold performance distribution, subject-wise variability |

---

# THESIS-READY STATEMENTS

## Validation Statement
*"To ensure subject-independent evaluation, subject-wise cross-validation was employed. Data from each subject were exclusively used either for training or testing, thereby eliminating subject leakage and enabling realistic generalization assessment."*

## Reliability Statement
*"A comprehensive reliability assessment was conducted encompassing test-retest reliability, inter-rater agreement, internal consistency, robustness to noise and artifacts, and cross-session stability. The consolidated analysis confirms consistent and reliable system behavior under real-world conditions."*

## Clinical Validation Statement
*"Clinical validation and real-world performance assessment were conducted through diagnostic accuracy, agreement analysis, robustness testing, subject-wise evaluation, and deployment-level analysis. The results demonstrate reliable, safe, and transferable system behavior under realistic clinical conditions."*

---

*Document Version: 1.0*
*Framework for: AgenticFinder EEG Classification System*
*Last Updated: January 4, 2026*
