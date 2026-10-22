# Triage policy and human clearance

## What the three queues mean
TreadOps turns the calibrated probability that a photo shows a defective tyre, p(defective), into one of three queues. Below the low threshold the casing is a retread candidate (good_candidate); at or above the high threshold it is defect flagged (defect_flagged); everything between goes to expert review (manual_review). The thresholds are versioned in the policy file and selected on a held-out policy cohort, never on the locked test set.

## Low-quality photos are never auto-cleared
If the image-quality screen reports underexposure, overexposure or blur, a casing that would otherwise be a retread candidate is escalated to expert review. A dark or blurred photo hides cracks, so the right action is to recapture the image.

## Human clearance is always required
The model is a triage aid trained on image-level labels from public photographs. It does not certify a tyre as safe, roadworthy or fit for retreading. A qualified inspector makes the final decision, and retreaders perform their own full casing inspection (for example shearography) before any casing is retreaded.

## Separate evaluation threshold
A binary evaluation threshold is used for recall and specificity reporting. It is chosen to meet the recall target on the policy cohort and is kept separate from the three-way triage thresholds.
