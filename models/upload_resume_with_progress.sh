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

LOCAL_MB=$((LOCAL_SIZE/1024/1024))
REMOTE_MB=$((REMOTE_SIZE/1024/1024))
REMAINING_MB=$(( (LOCAL_SIZE-REMOTE_SIZE)/1024/1024 ))
PERCENT=$((REMOTE_SIZE*100/LOCAL_SIZE))

echo "📊 Current status: $PERCENT% complete"
echo "📁 $REMOTE_MB MB of $LOCAL_MB MB uploaded"
echo "⏳ $REMAINING_MB MB remaining"

if [ "$REMOTE_SIZE" -eq "$LOCAL_SIZE" ]; then
    echo "✅ File already fully uploaded!"
    exit 0
elif [ "$REMOTE_SIZE" -gt "$LOCAL_SIZE" ]; then
    echo "⚠️ Remote file is larger than local. Removing and starting fresh..."
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "rm -f '$REMOTE_PATH'"
    REMOTE_SIZE=0
    REMOTE_MB=0
    REMAINING_MB=$LOCAL_MB
fi

if [ "$REMOTE_SIZE" -gt 0 ]; then
    echo "🔄 Resuming upload from $REMOTE_MB MB..."
    echo "📈 Starting progress monitor..."
    
    # Start background progress monitoring
    (
        while true; do
            sleep 5  # Update every 5 seconds
            CURRENT_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                          "stat -c%s '$REMOTE_PATH' 2>/dev/null || echo 0")
            CURRENT_MB=$((CURRENT_SIZE/1024/1024))
            CURRENT_PERCENT=$((CURRENT_SIZE*100/LOCAL_SIZE))
            echo "Progress: $CURRENT_PERCENT% ($CURRENT_MB/$LOCAL_MB MB)"
            
            # Stop when upload is complete
            if [ "$CURRENT_SIZE" -ge "$LOCAL_SIZE" ]; then
                echo "✅ Upload complete!"
                break
            fi
        done
    ) &
    PROGRESS_PID=$!
    
    # Do the actual resume upload
    echo "⏩ Uploading remaining $REMAINING_MB MB..."
    tail -c +$((REMOTE_SIZE + 1)) "$LOCAL_FILE" | \
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "cat >> '$REMOTE_PATH'"
    
    # Stop progress monitor
    kill $PROGRESS_PID 2>/dev/null
    wait $PROGRESS_PID 2>/dev/null
    
else
    echo "📤 Starting new upload..."
    # For new uploads, use scp which shows progress
    scp -i "$SSH_KEY" "$LOCAL_FILE" "$REMOTE_USER@$REMOTE_HOST:${REMOTE_PATH}.tmp"
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "mv '${REMOTE_PATH}.tmp' '$REMOTE_PATH'"
fi

# Final verification
FINAL_REMOTE_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                   "stat -c%s '$REMOTE_PATH' 2>/dev/null || echo 0")

if [ "$FINAL_REMOTE_SIZE" -eq "$LOCAL_SIZE" ]; then
    echo "🎉 Upload completed successfully!"
else
    echo "⚠️ Upload incomplete. Run script again to resume."
    echo "   Final: $((FINAL_REMOTE_SIZE/1024/1024)) MB of $LOCAL_MB MB"
fi
