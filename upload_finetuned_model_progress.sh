#!/bin/bash

LOCAL_DIR="/Users/benjaminsinzore/csm_streaming_cpu/finetuned_model"
REMOTE_USER="ecs-assist-user"
REMOTE_HOST="139.129.39.218"
REMOTE_DIR="/home/ecs-assist-user/csm_streaming_cpu/finetuned_model"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🔍 Checking upload status..."

# Check if local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "❌ Local directory not found: $LOCAL_DIR"
    exit 1
fi

# Get local directory size and file count
echo "📊 Calculating local directory size..."
LOCAL_SIZE=$(find "$LOCAL_DIR" -type f -exec stat -f%z {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')
LOCAL_MB=$((LOCAL_SIZE/1024/1024))
TOTAL_FILES=$(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ')

echo "📁 Uploading: finetuned_model"
echo "💾 Size: $LOCAL_MB MB"
echo "📄 Files: $TOTAL_FILES"

# Check and clean remote directory
echo "📡 Checking remote directory..."
REMOTE_EXISTS=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                "if [ -d '$REMOTE_DIR' ]; then echo 'exists'; else echo 'not_exists'; fi")

if [ "$REMOTE_EXISTS" = "exists" ]; then
    echo "⚠️ Removing existing remote directory..."
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "rm -rf '$REMOTE_DIR'"
fi

echo "🚀 Starting upload with progress..."

# Method 1: Use pv for progress bar (if installed)
if command -v pv >/dev/null 2>&1; then
    echo "📊 Using pv for progress display..."
    cd "/Users/benjaminsinzore/csm_streaming_cpu"
    tar czf - "finetuned_model" 2>/dev/null | \
    pv -s ${LOCAL_SIZE} | \
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
        "cd /home/ecs-assist-user/csm_streaming_cpu && tar xzf -"

# Method 2: Use scp with progress (fallback)
else
    echo "📤 Using scp with progress (installing pv is recommended for better progress)..."
    echo "💡 Install pv with: brew install pv"
    
    # Start background progress monitoring
    (
        echo "⏳ Upload in progress... (scp will show basic progress)"
        while true; do
            sleep 10
            # Check remote directory size during upload
            REMOTE_CURRENT_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                                "find '$REMOTE_DIR' -type f -exec stat -c%s {} \; 2>/dev/null | awk '{sum+=\$1} END {print sum+0}'" 2>/dev/null || echo 0)
            REMOTE_CURRENT_MB=$((REMOTE_CURRENT_SIZE/1024/1024))
            PERCENT=$((REMOTE_CURRENT_SIZE*100/LOCAL_SIZE))
            echo "Progress: $PERCENT% ($REMOTE_CURRENT_MB/$LOCAL_MB MB)"
            
            if [ "$REMOTE_CURRENT_SIZE" -ge "$LOCAL_SIZE" ]; then
                break
            fi
        done
    ) &
    PROGRESS_PID=$!
    
    # Use scp for upload (shows progress)
    scp -r -i "$SSH_KEY" "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$(dirname "$REMOTE_DIR")/"
    
    # Stop progress monitor
    kill $PROGRESS_PID 2>/dev/null
    echo
fi

if [ $? -eq 0 ]; then
    echo "✅ Upload completed successfully!"
    
    # Verification
    REMOTE_FILES=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                   "find '$REMOTE_DIR' -type f | wc -l" 2>/dev/null | tr -d ' ')
    REMOTE_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                 "find '$REMOTE_DIR' -type f -exec stat -c%s {} \; 2>/dev/null | awk '{sum+=\$1} END {print sum}'")
    REMOTE_MB=$((REMOTE_SIZE/1024/1024))
    
    echo "📊 Verification:"
    echo "   Files: $TOTAL_FILES local → $REMOTE_FILES remote"
    echo "   Size: $LOCAL_MB MB local → $REMOTE_MB MB remote"
    
    if [ "$TOTAL_FILES" -eq "$REMOTE_FILES" ] && [ "$LOCAL_SIZE" -eq "$REMOTE_SIZE" ]; then
        echo "🎉 Upload verified! All files transferred successfully."
    else
        echo "⚠️ Verification mismatch. Files or sizes don't match."
    fi
else
    echo "❌ Upload failed."
fi
