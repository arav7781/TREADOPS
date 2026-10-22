# Data acquisition (D1)

1. `make fetch-data` downloads the publisher's "Download All" archive through
   `https://data.mendeley.com/public-api/zip/bn7ch8tvyp/download/1` (it redirects to a short-lived signed URL)
   using resumable parallel range requests, then records `docs/evidence/data/download_receipt.json` with the
   archive SHA-256 and the publisher's per-file SHA-256 listing.
2. Manual alternative: open https://data.mendeley.com/datasets/bn7ch8tvyp/1, click **Download All**, and run
   `make prepare-data ARCHIVE=/path/to/bn7ch8tvyp-1.zip`.
3. `prepare` reads members in memory (no extraction to disk), cross-checks publisher hashes, quarantines rejects
   with a reason, and writes canonical derivatives plus manifests. The publisher's file listing returns at most
   1000 entries per folder, so 28 archive members are "unlisted" (they are the byte-identical duplicates).
