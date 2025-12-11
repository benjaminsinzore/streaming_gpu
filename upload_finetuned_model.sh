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
    echo "Available directories in /Users/benjaminsinzore/csm_streaming_cpu/:"
    ls -la "/Users/benjaminsinzore/csm_streaming_cpu/" | head -10
    exit 1
fi

# Get local directory size (macOS compatible)
echo "📊 Calculating local directory size..."
LOCAL_SIZE=$(find "$LOCAL_DIR" -type f -exec stat -f%z {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')
LOCAL_MB=$((LOCAL_SIZE/1024/1024))

echo "📁 Uploading: finetuned_model"
echo "💾 Local size: $LOCAL_MB MB"

# Count files
TOTAL_FILES=$(find "$LOCAL_DIR" -type f | wc -l | tr -d ' ')
echo "📄 Total files: $TOTAL_FILES"

# Check remote directory - just check if it exists, don't create
echo "📡 Checking remote directory..."
REMOTE_EXISTS=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                "if [ -d '$REMOTE_DIR' ]; then echo 'exists'; else echo 'not_exists'; fi")

if [ "$REMOTE_EXISTS" = "exists" ]; then
    echo "⚠️ Remote directory already exists. Removing to start fresh..."
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "rm -rf '$REMOTE_DIR'"
else
    echo "📁 Remote directory will be created by tar extraction"
fi

echo "📤 Starting upload with tar + ssh..."
echo "⏳ Uploading $LOCAL_MB MB across $TOTAL_FILES files..."

# Upload with tar - shows file names as they're processed
cd "/Users/benjaminsinzore/csm_streaming_cpu"
tar czf - "finetuned_model" 2>/dev/null | \
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
    "cd /home/ecs-assist-user/csm_streaming_cpu && tar xzf -"

if [ $? -eq 0 ]; then
    echo "✅ Upload completed successfully!"
    
    # Verify by counting remote files
    REMOTE_FILES=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                   "find '$REMOTE_DIR' -type f | wc -l" 2>/dev/null | tr -d ' ')
    
    echo "📊 Verification:"
    echo "   Local files: $TOTAL_FILES"
    echo "   Remote files: $REMOTE_FILES"
    
    if [ "$TOTAL_FILES" -eq "$REMOTE_FILES" ]; then
        echo "🎉 Upload verified! All files transferred successfully."
    else
        echo "⚠️ File count mismatch. Local: $TOTAL_FILES, Remote: $REMOTE_FILES"
    fi
    
    # Show remote directory structure
    echo "📁 Remote directory structure:"
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "ls -la '$REMOTE_DIR/'"
else
    echo "❌ Upload failed."
fi
