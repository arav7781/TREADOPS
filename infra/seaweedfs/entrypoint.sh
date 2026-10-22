#!/bin/sh
# Single-node SeaweedFS with an S3 gateway; the identity file is rendered from env, never committed.
set -eu
mkdir -p /etc/seaweedfs
cat > /etc/seaweedfs/s3.json <<JSON
{"identities":[{"name":"treadops","credentials":[{"accessKey":"${S3_ACCESS_KEY_ID}","secretKey":"${S3_SECRET_ACCESS_KEY}"}],
"actions":["Admin","Read","List","Tagging","Write"]}]}
JSON
exec weed server -dir=/data -ip=127.0.0.1 -ip.bind=0.0.0.0 -master.volumeSizeLimitMB=512 -volume.max=0 \
  -s3 -s3.port=8333 -s3.config=/etc/seaweedfs/s3.json
