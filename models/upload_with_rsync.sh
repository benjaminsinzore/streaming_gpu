#!/bin/bash

LOCAL_FILE="/Users/benjaminsinzore/csm_streaming_cpu/models/basqui-1b-instruct.gguf"
REMOTE_USER="ecs-assist-user"
REMOTE_HOST="139.129.39.218"
REMOTE_PATH="/home/ecs-assist-user/csm_streaming_cpu/models/"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "Starting resumable upload of $(basename $LOCAL_FILE)..."

# Use rsync with partial progress and compression
rsync -avz --progress --partial \
      -e "ssh -i $SSH_KEY" \
      "$LOCAL_FILE" \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Upload completed successfully!"
else
    echo "⚠️  Upload interrupted. You can resume by running this script again."
fi
