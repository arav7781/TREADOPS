# Model and data

## Model
TreadOps serves an Ultralytics YOLO26 classification model (nano by default) fine-tuned for two classes, good and defective, on full-image letterboxed inputs. Probabilities are temperature-calibrated on a dedicated calibration cohort.

## Data
Training uses the public Mendeley dataset "Digital images of defective and good condition tyres" (doi:10.17632/bn7ch8tvyp.1, CC BY 4.0). Photos of the same tyre were grouped so they never appear on both sides of a train/test split. No Michelin production data is used.

## Evidence and gates
Each model bundle carries a gate report: hard gates (no leakage, complete lineage, finite outputs, correct class mapping, ONNX parity) and quality targets (defect recall at least 0.95 and specificity at least 0.90 on the development gate cohort). A bundle that fails or lacks evidence cannot be promoted.

## Limitations
Labels are image-level, so the attribution heatmap is an explanation aid, not a defect location. The dataset mixes passenger and commercial tyres photographed with a phone, which differs from plant imaging such as shearography.
