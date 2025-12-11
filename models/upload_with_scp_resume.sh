#!/bin/bash

LOCAL_FILE="/Users/benjaminsinzore/csm_streaming_cpu/models/basqui-1b-instruct.gguf"
REMOTE_USER="ecs-assist-user"
REMOTE_HOST="139.129.39.218"
REMOTE_PATH="/home/ecs-assist-user/csm_streaming_cpu/models/basqui-1b-instruct.gguf"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🔍 Checking upload status..."

# Get file sizes
LOCAL_SIZE=$(stat -f%z "$LOCAL_FILE" 2>/dev/null)
REMOTE_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
             "stat -c%s '$REMOTE_PATH' 2>/dev/null || echo 0")

echo "Local file: $LOCAL_SIZE bytes"
echo "Remote file: $REMOTE_SIZE bytes"

if [ "$REMOTE_SIZE" -eq "$LOCAL_SIZE" ]; then
    echo "✅ File already fully uploaded!"
    exit 0
elif [ "$REMOTE_SIZE" -gt "$LOCAL_SIZE" ]; then
    echo "⚠️ Remote file is larger than local. Removing and starting fresh..."
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "rm -f '$REMOTE_PATH'"
    REMOTE_SIZE=0
fi

if [ "$REMOTE_SIZE" -gt 0 ]; then
    echo "🔄 Resuming upload from byte $REMOTE_SIZE..."
    # Use tail to skip already uploaded bytes and pipe to ssh
    tail -c +$((REMOTE_SIZE + 1)) "$LOCAL_FILE" | \
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
        "cat >> '$REMOTE_PATH'"
else
    echo "📤 Starting new upload..."
    scp -i "$SSH_KEY" "$LOCAL_FILE" "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PATH}.tmp"
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "mv '${REMOTE_PATH}.tmp' '$REMOTE_PATH'"
fi

# Verify
FINAL_REMOTE_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                   "stat -c%s '$REMOTE_PATH' 2>/dev/null || echo 0")

if [ "$FINAL_REMOTE_SIZE" -eq "$LOCAL_SIZE" ]; then
    echo "✅ Upload completed successfully!"
else
    echo "⚠️ Upload incomplete. Run script again to resume."
    echo "   Progress: $FINAL_REMOTE_SIZE of $LOCAL_SIZE bytes"
fi
