# Model card - yolo26n-cls tyre casing triage

Weights `4d124daeb50c8acc` · task `classify` · preprocess `v1` · policy `v1` · gate status **FAILED**

## Intended use
Image-level triage of tyre photographs into retread-candidate / expert-review / defect-flagged queues at a casing intake. A human inspector gives final clearance; the model never certifies a tyre as safe.

## Data
Mendeley D1 (doi:10.17632/bn7ch8tvyp.1, CC BY 4.0): 1828 unique public photographs after removing 28 byte-identical duplicates, grouped into reviewed same-tyre clusters and split 70/15/15 by group. No Michelin production data is used; operational metadata in replays is simulated.
Fitted on 1084 images (dev_train); checkpoint selected on dev_val.

## Evidence (actual measurements)
- Gate cohort (development evidence, outer validation): recall 0.967 (145/150, 95% CI 0.924-0.986); specificity 0.896 (112/125, 95% CI 0.830-0.938)
- Locked test at threshold 0.205788: recall 0.940 (141/150, 95% CI 0.890-0.968); specificity 0.906 (115/127, 95% CI 0.842-0.945)
- Test AP 0.979385630198583, Brier 0.0661486179700978
- Stress (paired test parents): camera_blur: recall 0.8866666666666667, specificity 0.9212598425196851; low_light: recall 0.7466666666666667, specificity 0.8740157480314961
- CPU model-only latency p95 15.0 ms (batch 1, warm, Apple M5)

## Limitations
- Public photographs of mixed passenger/commercial tyres; not shearography or plant imagery.
- Same-tyre clusters were inferred from visual similarity and capture order; residual leakage is possible.
- Image-level labels only: the attribution heatmap is not a defect localisation.
- Sustainability figures shown in the app are estimates from Michelin's published truck/bus factors.
