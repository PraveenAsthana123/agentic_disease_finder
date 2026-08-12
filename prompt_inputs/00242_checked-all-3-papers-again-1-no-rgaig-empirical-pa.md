# Request #242

- When: 2026-06-30 00:49:12 MDT
- Status: open

## Input
Checked all 3 papers again.
1. No-RGAIG Empirical Paper
File: [q1_noRGAIG_2col.tex](/media/praveen/Asthana4/rajveer/agenticfinder/epilepsypaper-ready/code/paper/q1_noRGAIG_2col.tex)
Best paper right now. This is the safest submission candidate.
Current scope:
Patient-independent EEG seizure detection, CHB-MIT epoch-level vs CHB-MIT LOSO, Bonn/UCI only as secondary easy-dataset reference.
Novelty:
Validation-honesty benchmark. Shows that high epoch-level accuracy collapses under subject-wise testing.
Future scope:
Nested LOSO, external adult dataset, EEGNet/CNN/TCN/Transformer, false alarms/hour, latency, calibration, patient adaptation.
Remaining problem:
There is still RGAIG contamination. Line 115 references Sec.~\ref{sec:rgaig} and MCP even though this is the no-RGAIG paper. Also line 39 says “governed-deployment layer the literature omits.” Remove or soften that.
Decision:
Submit this one first after cleaning leftover RGAIG/MCP text.
2. RGAIG Empirical Paper
File: [q1_full_2col.tex](/media/praveen/Asthana4/rajveer/agenticfinder/epilepsypaper-ready/code/paper/q1_full_2col.tex)
Interesting, but risky.
Current scope:
Same seizure-detection benchmark plus RGAIG deployment framework: governance, RAG explanation, MCP tools, AIOps, audit, drift, fairness gate.
Novelty:
Combines honest LOSO seizure benchmark with governed deployment architecture.
Future scope:
Actual RGAIG system evaluation: latency, audit workflow, RAG faithfulness, clinician review, drift-monitoring validation, human-in-the-loop trial.
Remaining problem:
RGAIG is specified but not really evaluated. The paper says this clearly, which helps, but reviewers may still say the title overclaims. If RGAIG is in the title, they expect RGAIG results.
Decision:
Good as a second paper or systems/deployment paper. Not the safest Q1 empirical submission unless you add real RGAIG evaluation.
3. Review Paper
File: [review_full_2col.tex](/media/praveen/Asthana4/rajveer/agenticfinder/epilepsypaper-ready/code/paper/review_full_2col.tex)
Strong idea, but highest rejection risk unless evidence tables are perfect.
Current scope:
Systematic review of 50 EEG epilepsy AI works from 2021-2026, focused on evaluation leakage, architecture trends, datasets, governance, RAI/XAI, and deployment.
Novelty:
Evaluation-honesty lens plus governed deployment taxonomy. This is a good review angle.
Future scope:
Community LOSO benchmark, cross-dataset transfer, calibration/fairness reporting, consumer EEG validation, foundation models at clinical sensitivity, governed human-in-the-loop deployment.
Remaining problem:
PRISMA is still too approximate: “~220 records” looks weak. A review paper needs exact search strings, database dates, inclusion/exclusion table, risk-of-bias table, and a clean included-study matrix. Also the review source depends on included files like _scopus, _clinical, _reviz, _bibmatrix; make sure it builds from this folder.
Decision:
Do not submit the review until the PRISMA and evidence matrix are rigorous.
Brutal Ranking
q1_noRGAIG_2col.tex = best and safest.
q1_full_2col.tex = good concept, but RGAIG must be evaluated or clearly secondary.
review_full_2col.tex = potentially strong, but only if PRISMA/evidence table is cleaned hard.
Main fix now: clean the no-RGAIG paper of leftover RGAIG/MCP references, then make that the primary submission.
