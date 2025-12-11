#!/bin/bash

LOCAL_FILE="/Users/benjaminsinzore/csm_streaming_cpu/models/basqui-1b-instruct.gguf"
REMOTE_USER="ecs-assist-user"
REMOTE_HOST="139.129.39.218"
REMOTE_PATH="/home/ecs-assist-user/csm_streaming_cpu/models/basqui-1b-instruct.gguf"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🔍 Checking upload status..."

LOCAL_SIZE=$(stat -f%z "$LOCAL_FILE" 2>/dev/null)
REMOTE_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
             "stat -c%s '$REMOTE_PATH' 2>/dev/null || echo 0")

echo "Progress: $((REMOTE_SIZE*100/LOCAL_SIZE))% complete"

if [ "$REMOTE_SIZE" -eq "$LOCAL_SIZE" ]; then
    echo "✅ File already fully uploaded!"
    exit 0
fi

# Always use scp - it shows progress and will overwrite the partial file
echo "📤 Uploading with progress..."
scp -i "$SSH_KEY" "$LOCAL_FILE" "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PATH}.tmp"
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "mv '${REMOTE_PATH}.tmp' '$REMOTE_PATH'"

echo "✅ Upload completed!"
