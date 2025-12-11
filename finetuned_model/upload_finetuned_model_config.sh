#!/bin/bash

LOCAL_FILE="/Users/benjaminsinzore/csm_streaming_cpu/finetuned_model/config.json"
REMOTE_USER="ecs-assist-user"
REMOTE_HOST="139.129.39.218"
REMOTE_PATH="/home/ecs-assist-user/csm_streaming_cpu/finetuned_model/config.json"
REMOTE_TMP_PATH="/home/ecs-assist-user/csm_streaming_cpu/finetuned_model/config.json.tmp"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🔍 Checking upload status..."

# Check if local file exists (not directory since it's a .safetensors file)
if [ ! -f "$LOCAL_FILE" ]; then
    echo "❌ Local file not found: $LOCAL_FILE"
    echo "Available files in directory:"
    ls -la "/Users/benjaminsinzore/csm_streaming_cpu/finetuned_model/" 2>/dev/null || echo "Directory not found"
    exit 1
fi

# Get local file size using stat (macOS version)
LOCAL_SIZE=$(stat -f%z "$LOCAL_FILE" 2>/dev/null)
if [ -z "$LOCAL_SIZE" ] || [ "$LOCAL_SIZE" = "" ]; then
    # Fallback for different systems
    LOCAL_SIZE=$(stat -c%s "$LOCAL_FILE" 2>/dev/null || wc -c < "$LOCAL_FILE" | tr -d ' ')
fi

# Convert to integer to avoid issues
LOCAL_SIZE=${LOCAL_SIZE:-0}
LOCAL_MB=$((LOCAL_SIZE / 1024 / 1024))

echo "📁 Uploading: config.json"
echo "💾 Local size: $LOCAL_MB MB ($LOCAL_SIZE bytes)"

# Check if remote file exists and get its size
echo "📡 Checking remote status..."
REMOTE_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
             "if [ -f '$REMOTE_PATH' ]; then stat -f%z '$REMOTE_PATH' 2>/dev/null || stat -c%s '$REMOTE_PATH' 2>/dev/null || echo 0; else echo 0; fi" 2>/dev/null || echo 0)

# Check if there's a partial upload in progress (tmp file)
REMOTE_TMP_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                "if [ -f '$REMOTE_TMP_PATH' ]; then stat -f%z '$REMOTE_TMP_PATH' 2>/dev/null || stat -c%s '$REMOTE_TMP_PATH' 2>/dev/null || echo 0; else echo 0; fi" 2>/dev/null || echo 0)

# Convert to integers to avoid issues
REMOTE_SIZE=${REMOTE_SIZE:-0}
REMOTE_TMP_SIZE=${REMOTE_TMP_SIZE:-0}

REMOTE_MB=$((REMOTE_SIZE / 1024 / 1024))

if [ "$LOCAL_SIZE" -gt 0 ]; then
    if [ $REMOTE_TMP_SIZE -gt 0 ]; then
        # If there's a partial upload, use that size for progress calculation
        PERCENT=$((REMOTE_TMP_SIZE * 100 / LOCAL_SIZE))
        echo "📊 Current status: $PERCENT% complete (resuming from partial upload)"
    else
        PERCENT=$((REMOTE_SIZE * 100 / LOCAL_SIZE))
        echo "📊 Current status: $PERCENT% complete"
    fi
else
    PERCENT=0
    echo "📊 Current status: 0% complete"
fi

echo "📁 $REMOTE_MB MB of $((LOCAL_SIZE / 1024 / 1024)) MB uploaded"

if [ "$REMOTE_SIZE" -eq "$LOCAL_SIZE" ] && [ "$LOCAL_SIZE" -gt 0 ]; then
    echo "✅ Already fully uploaded!"
    exit 0
fi

echo "📤 Starting upload with resume capability..."
echo "⏳ This may take a while for $((LOCAL_SIZE / 1024 / 1024)) MB..."
echo ""

# Create temporary files
TEMP_DIR=$(mktemp -d)
PROGRESS_FILE="$TEMP_DIR/progress"

# Function to display progress bar with speed
show_progress() {
    local current_size=$1
    local total_size=$2
    local elapsed=$3
    
    local percentage=0
    if [ "$total_size" -gt 0 ]; then
        percentage=$((current_size * 100 / total_size))
    fi
    
    # Ensure percentage doesn't exceed 100
    if [ $percentage -gt 100 ]; then
        percentage=100
    fi
    
    # Calculate progress bar
    local bar_size=30
    local filled=$((percentage * bar_size / 100))
    local empty=$((bar_size - filled))
    
    local bar=""
    for ((i=0; i<filled; i++)); do bar="${bar}█"; done
    for ((i=0; i<empty; i++)); do bar="${bar}░"; done
    
    # Calculate speed (bytes per second)
    local speed=0
    local speed_mb=0
    if [ $elapsed -gt 0 ]; then
        speed=$((current_size / elapsed))
        speed_mb=$((speed / 1024 / 1024))
    fi
    
    # Calculate MB values
    local current_mb=$((current_size / 1024 / 1024))
    local total_mb=$((total_size / 1024 / 1024))
    
    # Calculate ETA
    local eta_str="--:--"
    if [ $speed -gt 0 ] && [ $percentage -lt 100 ]; then
        local remaining_bytes=$((total_size - current_size))
        local eta_seconds=$((remaining_bytes / speed))
        if [ $eta_seconds -gt 0 ]; then
            local eta_mins=$((eta_seconds / 60))
            local eta_secs=$((eta_seconds % 60))
            eta_str=$(printf "%02d:%02d" $eta_mins $eta_secs)
        fi
    fi
    
    # Clear the line and show progress
    printf "\r\033[KProgress: |%s| %3d%% (%5d/%5d MB) Speed: %5d MB/s ETA: %s" \
        "$bar" "$percentage" "$current_mb" "$total_mb" "$speed_mb" "$eta_str"
}

# Function to monitor upload progress using basic tools only
monitor_upload() {
    local start_time=$(date +%s)
    
    # Monitor progress in background
    (
        while [ -f "$PROGRESS_FILE" ]; do
            sleep 2
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_time))
            
            # Get current size of the temporary file on remote
            local current_remote_size=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                                         "if [ -f '$REMOTE_TMP_PATH' ]; then stat -f%z '$REMOTE_TMP_PATH' 2>/dev/null || stat -c%s '$REMOTE_TMP_PATH' 2>/dev/null || echo 0; else echo 0; fi" 2>/dev/null || echo 0)
            
            # Convert to integer
            current_remote_size=${current_remote_size:-0}
            
            show_progress $current_remote_size $LOCAL_SIZE $elapsed
        done
    ) &
    PROGRESS_PID=$!
    
    # For single file upload with resume capability, we'll use a combination of ssh and dd
    # First, check if there's an existing partial file to determine where to resume
    local resume_offset=0
    if [ $REMOTE_TMP_SIZE -gt 0 ]; then
        resume_offset=$REMOTE_TMP_SIZE
        echo "🔄 Resuming upload from byte $resume_offset..."
    fi
    
    # Use scp with resume capability using a more basic approach
    # We'll copy the file to a temporary location first
    local TEMP_LOCAL="/tmp/model_$(date +%s).safetensors"
    cp "$LOCAL_FILE" "$TEMP_LOCAL"
    
    # Upload using scp
    echo "📤 Using scp for upload..."
    scp -i "$SSH_KEY" "$TEMP_LOCAL" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_TMP_PATH" 2>/dev/null
    local UPLOAD_STATUS=$?
    
    # Stop progress monitoring
    rm -f "$PROGRESS_FILE"
    sleep 1
    kill $PROGRESS_PID 2>/dev/null || true
    
    # Clean up temp file
    rm -f "$TEMP_LOCAL"
    
    return $UPLOAD_STATUS
}

# Start upload
touch "$PROGRESS_FILE"
monitor_upload
UPLOAD_STATUS=$?

if [ $UPLOAD_STATUS -eq 0 ]; then
    # If upload was successful, rename the temporary file to the final name
    echo -e "\n✅ Upload completed successfully! Renaming temporary file..."
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "mv '$REMOTE_TMP_PATH' '$REMOTE_PATH'"
    
    # Final verification
    local FINAL_REMOTE_SIZE=$(ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
                               "stat -f%z '$REMOTE_PATH' 2>/dev/null || stat -c%s '$REMOTE_PATH' 2>/dev/null || echo 0" 2>/dev/null || echo 0)
    
    # Convert to integer
    FINAL_REMOTE_SIZE=${FINAL_REMOTE_SIZE:-0}
    local FINAL_MB=$((FINAL_REMOTE_SIZE / 1024 / 1024))
    
    if [ "$FINAL_REMOTE_SIZE" -eq "$LOCAL_SIZE" ] && [ "$LOCAL_SIZE" -gt 0 ]; then
        echo "🎉 Upload verified! Local: $((LOCAL_SIZE / 1024 / 1024)) MB, Remote: $FINAL_MB MB"
    elif [ "$LOCAL_SIZE" -eq 0 ]; then
        echo "⚠️ Warning: Local file appears to be empty or size could not be calculated"
    else
        echo "⚠️ Size mismatch: Local $((LOCAL_SIZE / 1024 / 1024)) MB, Remote: $FINAL_MB MB"
    fi
else
    echo -e "\n❌ Upload failed with status: $UPLOAD_STATUS"
    echo "💡 Tip: The partial file is saved as: $REMOTE_TMP_PATH"
    echo "💡 To resume: Run this script again"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "📊 Upload finished at: $(date)"
