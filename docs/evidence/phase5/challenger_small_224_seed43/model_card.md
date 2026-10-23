# Model card - yolo26s-cls tyre casing triage

Weights `0aa305a7f9d08b10` · task `classify` · preprocess `v1` · policy `v1` · gate status **PASSED**

## Intended use
Image-level triage of tyre photographs into retread-candidate / expert-review / defect-flagged queues at a casing intake. A human inspector gives final clearance; the model never certifies a tyre as safe.

## Data
Mendeley D1 (doi:10.17632/bn7ch8tvyp.1, CC BY 4.0): 1828 unique public photographs after removing 28 byte-identical duplicates, grouped into reviewed same-tyre clusters and split 70/15/15 by group. No Michelin production data is used; operational metadata in replays is simulated.
Fitted on 1084 images (dev_train); checkpoint selected on dev_val.

## Evidence (actual measurements)
- Gate cohort (development evidence, outer validation): recall 0.953 (143/150, 95% CI 0.907-0.977); specificity 0.976 (122/125, 95% CI 0.932-0.992)
- Locked test at threshold 0.843939: recall 0.887 (133/150, 95% CI 0.826-0.928); specificity 0.969 (123/127, 95% CI 0.922-0.988)
- Test AP 0.9888506753281553, Brier 0.047875178865820475
- Stress (paired test parents): camera_blur: recall 0.8466666666666667, specificity 0.9763779527559056; low_light: recall 0.68, specificity 0.9763779527559056
- CPU model-only latency p95 10.0 ms (batch 1, warm, Apple M5)

## Limitations
- Public photographs of mixed passenger/commercial tyres; not shearography or plant imagery.
- Same-tyre clusters were inferred from visual similarity and capture order; residual leakage is possible.
- Image-level labels only: the attribution heatmap is not a defect localisation.
- Sustainability figures shown in the app are estimates from Michelin's published truck/bus factors.
