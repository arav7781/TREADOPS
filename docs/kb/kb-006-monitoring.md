# Monitoring and drift

## Three views
System health covers queue age, retries, failures and latency. Input and output shift compares image descriptors (brightness, contrast, blur, saturation, aspect ratio), triage rates and fixed-encoder embeddings with the reference cohort. Labelled quality measures recall and specificity on inspections that have verified labels.

## When an alert opens
A window of the latest 500 completed inspections is evaluated every 200 completions. It needs at least 200 distinct parent images, otherwise the status is insufficient data. A drift alert opens only after two consecutive windows exceed the threshold (PSI above 0.20) for the same feature.

## What to do
A drift alert is an investigation signal, not proof that the model got worse. Check which cameras or scenarios are affected, look at sample images, and collect verified labels to measure quality. Retraining and promotion are separate gated operations; an alert never changes the serving model.
