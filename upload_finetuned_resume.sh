#!/bin/bash

LOCAL_DIR="/Users/benjaminsinzore/csm_streaming_cpu/finetuned_model"
REMOTE_USER="ecs-assist-user"
REMOTE_HOST="139.129.39.218"
REMOTE_DIR="/home/ecs-assist-user/csm_streaming_cpu/finetuned_model"
REMOTE_TEMP="${REMOTE_DIR}.tmp"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🔍 Checking upload status..."

# Check if local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "❌ Local directory not found: $LOCAL_DIR"
    exit 1
fi

# Get local directory info
echo "📊 Calculating local directory size..."
LOCAL_SIZE=$(find "$LOCAL_DIR" -type f -exec stat -f%z {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')
LOCAL_MB=$((LOCAL_SIZE/1024/1024))
TOTAL_FILES=$(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ')

echo "📁 Uploading: finetuned_model"
echo "💾 Size: $LOCAL_MB MB"
echo "📄 Files: $TOTAL_FILES"

# Check if final directory already exists and is complete
echo "📡 Checking existing remote directory..."
REMOTE_EXISTS=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                "if [ -d '$REMOTE_DIR' ]; then du -sb '$REMOTE_DIR' | cut -f1; else echo 0; fi")

if [ "$REMOTE_EXISTS" -eq "$LOCAL_SIZE" ]; then
    echo "✅ Remote directory already exists and is complete!"
    exit 0
elif [ "$REMOTE_EXISTS" -gt 0 ]; then
    echo "⚠️ Remote directory exists but size doesn't match: $((REMOTE_EXISTS/1024/1024)) MB vs $LOCAL_MB MB"
    echo "   We'll upload to .tmp and then replace if complete"
fi

# Check temp directory progress
echo "📡 Checking temp directory progress..."
TEMP_EXISTS=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
              "if [ -d '$REMOTE_TEMP' ]; then du -sb '$REMOTE_TEMP' | cut -f1; else echo 0; fi")

TEMP_MB=$((TEMP_EXISTS/1024/1024))
PERCENT=$((TEMP_EXISTS*100/LOCAL_SIZE))

if [ "$TEMP_EXISTS" -gt 0 ]; then
    echo "🔄 Found existing temp directory: $PERCENT% complete ($TEMP_MB/$LOCAL_MB MB)"
    echo "📤 Resuming upload..."
else
    echo "📤 Starting new upload..."
fi

echo "🚀 Starting resumable upload with .tmp safety..."

# Start progress monitoring
(
    while true; do
        sleep 10
        CURRENT_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                      "if [ -d '$REMOTE_TEMP' ]; then du -sb '$REMOTE_TEMP' | cut -f1; else echo 0; fi" 2>/dev/null || echo 0)
        CURRENT_MB=$((CURRENT_SIZE/1024/1024))
        CURRENT_PERCENT=$((CURRENT_SIZE*100/LOCAL_SIZE))
        echo "Progress: $CURRENT_PERCENT% ($CURRENT_MB/$LOCAL_MB MB)"
        
        if [ "$CURRENT_SIZE" -ge "$LOCAL_SIZE" ]; then
            echo "✅ Upload complete!"
            break
        fi
    done
) &
PROGRESS_PID=$!

# Always use tar method (more reliable on macOS)
echo "📤 Using tar + ssh to .tmp..."
cd "/Users/benjaminsinzore/csm_streaming_cpu"
tar czf - "finetuned_model" 2>/dev/null | \
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
    "cd /home/ecs-assist-user/csm_streaming_cpu && mkdir -p '$REMOTE_TEMP' && cd '$REMOTE_TEMP' && tar xzf -"

# Stop progress monitor
kill $PROGRESS_PID 2>/dev/null
wait $PROGRESS_PID 2>/dev/null
echo

# Verify temp directory is complete
echo "🔍 Verifying upload..."
FINAL_TEMP_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                  "if [ -d '$REMOTE_TEMP' ]; then du -sb '$REMOTE_TEMP' | cut -f1; else echo 0; fi")
FINAL_TEMP_MB=$((FINAL_TEMP_SIZE/1024/1024))

if [ "$FINAL_TEMP_SIZE" -eq "$LOCAL_SIZE" ]; then
    echo "✅ Temp directory verified! Moving to final location..."
    
    # Atomic move from .tmp to final
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
        "rm -rf '$REMOTE_DIR' && mv '$REMOTE_TEMP' '$REMOTE_DIR'"
    
    echo "🎉 Upload completed successfully! Final directory is ready."
    
    # Final verification
    FINAL_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                "du -sb '$REMOTE_DIR' | cut -f1")
    FINAL_MB=$((FINAL_SIZE/1024/1024))
    
    echo "📊 Final verification: $FINAL_MB MB of $LOCAL_MB MB"
else
    echo "⚠️ Upload incomplete. Temp directory: $FINAL_TEMP_MB MB of $LOCAL_MB MB"
    echo "💡 Run this script again to resume the upload"
fi
