#!/bin/bash

LOCAL_BASE="/Users/benjaminsinzore/.cache/huggingface/hub"
REMOTE_BASE="/home/ecs-assist-user/.cache/huggingface/hub"
MODEL_DIR="models--sentence-transformers--all-MiniLM-L6-v2"

LOCAL_PATH="$LOCAL_BASE/$MODEL_DIR"
REMOTE_PATH="$REMOTE_BASE/$MODEL_DIR"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_USER="ecs-assist-user"
REMOTE_HOST="139.129.39.218"

echo "🔍 Checking if model directory exists locally..."
if [ ! -d "$LOCAL_PATH" ]; then
    echo "❌ Local model directory not found: $LOCAL_PATH"
    exit 1
fi

echo "📁 Model: $MODEL_DIR"
echo "📍 Local: $LOCAL_PATH"
echo "📍 Remote: $REMOTE_PATH"

# Check local size
LOCAL_SIZE=$(du -sh "$LOCAL_PATH" | cut -f1)
echo "💾 Local size: $LOCAL_SIZE"

# Create remote directory if it doesn't exist
echo "📋 Creating remote directory..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_BASE'"

# Upload using rsync with progress
echo "🚀 Starting upload..."
rsync -avz --progress --partial \
      -e "ssh -i $SSH_KEY" \
      "$LOCAL_PATH/" \
      "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

if [ $? -eq 0 ]; then
    echo "✅ Upload completed successfully!"
    
    # Verify remote size
    echo "🔍 Verifying remote copy..."
    REMOTE_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "du -sh '$REMOTE_PATH' | cut -f1")
    echo "💾 Remote size: $REMOTE_SIZE"
    
    if [ "$LOCAL_SIZE" = "$REMOTE_SIZE" ]; then
        echo "✅ Sizes match! Upload verified."
    else
        echo "⚠️ Size mismatch: Local $LOCAL_SIZE vs Remote $REMOTE_SIZE"
    fi
else
    echo "⚠️ Upload interrupted. Run script again to resume."
fi
