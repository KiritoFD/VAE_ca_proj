#!/bin/bash

# LGT Training Helper Script
# Usage: source train_helper.sh

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function: Start fresh training
lgt_train() {
    echo -e "${GREEN}Starting LGT training...${NC}"
    python train.py --config config.json
}

# Function: Resume from latest checkpoint
lgt_resume() {
    if [ -f "checkpoints/latest.pt" ]; then
        echo -e "${GREEN}Resuming from latest checkpoint...${NC}"
        python train.py --resume checkpoints/latest.pt
    else
        echo -e "${RED}No latest checkpoint found!${NC}"
        return 1
    fi
}

# Function: Resume from specific epoch
lgt_resume_epoch() {
    if [ -z "$1" ]; then
        echo -e "${RED}Usage: lgt_resume_epoch <epoch_number>${NC}"
        return 1
    fi
    
    epoch_pad=$(printf "%04d" $1)
    checkpoint="checkpoints/epoch_${epoch_pad}.pt"
    
    if [ -f "$checkpoint" ]; then
        echo -e "${GREEN}Resuming from epoch $1...${NC}"
        python train.py --resume "$checkpoint"
    else
        echo -e "${RED}Checkpoint not found: $checkpoint${NC}"
        return 1
    fi
}

# Function: Quick debug training
lgt_debug() {
    echo -e "${YELLOW}Starting quick debug training...${NC}"
    python train_launch.py --quick-debug
}

# Function: High quality training
lgt_quality() {
    echo -e "${YELLOW}Starting high quality training...${NC}"
    python train_launch.py --high-quality
}

# Function: List all checkpoints
lgt_list_ckpts() {
    echo -e "${GREEN}Available checkpoints:${NC}"
    ls -lh checkpoints/epoch_*.pt 2>/dev/null | awk '{print $9, "("$5")"}'
    
    if [ -f "checkpoints/latest.pt" ]; then
        echo -e "${GREEN}Latest:${NC}"
        ls -lh checkpoints/latest.pt | awk '{print $9, "("$5")"}'
    fi
}

# Function: View inference results
lgt_view_inference() {
    if [ -z "$1" ]; then
        echo -e "${GREEN}Latest inference results:${NC}"
        latest_epoch=$(ls -d checkpoints/inference/epoch_* 2>/dev/null | tail -1)
        if [ -n "$latest_epoch" ]; then
            echo "Directory: $latest_epoch"
            ls -1 "$latest_epoch"/*.jpg 2>/dev/null | head -10
        else
            echo -e "${RED}No inference results found${NC}"
        fi
    else
        epoch_pad=$(printf "%04d" $1)
        result_dir="checkpoints/inference/epoch_${epoch_pad}"
        if [ -d "$result_dir" ]; then
            echo -e "${GREEN}Inference results for epoch $1:${NC}"
            ls -lh "$result_dir"/*.jpg 2>/dev/null
        else
            echo -e "${RED}No results for epoch $1${NC}"
        fi
    fi
}

# Function: View training log
lgt_log() {
    echo -e "${GREEN}Latest training log:${NC}"
    tail -20 checkpoints/logs/training_*.csv 2>/dev/null
}

# Function: Monitor training in real-time
lgt_monitor() {
    watch -n 2 'tail -15 checkpoints/logs/training_*.csv'
}

# Function: Clean up old checkpoints (keep last N)
lgt_cleanup_ckpts() {
    if [ -z "$1" ]; then
        keep=5
    else
        keep=$1
    fi
    
    echo -e "${YELLOW}Keeping last $keep checkpoints, removing older ones...${NC}"
    
    # Count total checkpoints
    total=$(ls -1 checkpoints/epoch_*.pt 2>/dev/null | wc -l)
    to_remove=$((total - keep))
    
    if [ $to_remove -gt 0 ]; then
        ls -1 checkpoints/epoch_*.pt | head -$to_remove | while read ckpt; do
            echo "Removing: $ckpt"
            rm -f "$ckpt"
        done
        echo -e "${GREEN}Cleanup completed!${NC}"
    else
        echo "Nothing to remove."
    fi
}

# Function: Backup checkpoints
lgt_backup() {
    if [ -z "$1" ]; then
        backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    else
        backup_name=$1
    fi
    
    mkdir -p "backups/$backup_name"
    
    echo -e "${YELLOW}Backing up to: backups/$backup_name${NC}"
    cp -v checkpoints/latest.pt "backups/$backup_name/latest.pt" 2>/dev/null
    cp -v checkpoints/epoch_*.pt "backups/$backup_name/" 2>/dev/null
    
    echo -e "${GREEN}Backup completed!${NC}"
}

# Function: Show help
lgt_help() {
    cat << EOF
${GREEN}LGT Training Helper Functions${NC}

Usage (add to .bashrc or source this file):
  source train_helper.sh

Available commands:
  ${GREEN}lgt_train${NC}                - Start fresh training
  ${GREEN}lgt_resume${NC}               - Resume from latest checkpoint
  ${GREEN}lgt_resume_epoch <n>${NC}     - Resume from specific epoch
  ${GREEN}lgt_debug${NC}                - Quick debug training (5 epochs)
  ${GREEN}lgt_quality${NC}              - High quality training (200 epochs)
  
  ${GREEN}lgt_list_ckpts${NC}           - List all checkpoints
  ${GREEN}lgt_view_inference [n]${NC}   - View inference results
  ${GREEN}lgt_log${NC}                  - Show latest training log
  ${GREEN}lgt_monitor${NC}              - Real-time monitor (requires watch)
  
  ${GREEN}lgt_cleanup_ckpts [n]${NC}    - Keep only last N checkpoints (default: 5)
  ${GREEN}lgt_backup [name]${NC}        - Backup all checkpoints
  
  ${GREEN}lgt_help${NC}                 - Show this help

${YELLOW}Examples:${NC}
  # Start fresh training
  lgt_train
  
  # Resume training
  lgt_resume
  
  # Resume from epoch 50
  lgt_resume_epoch 50
  
  # Quick debugging
  lgt_debug
  
  # Monitor progress
  lgt_monitor
  
  # Backup before major changes
  lgt_backup v1.0_before_weight_change
  lgt_cleanup_ckpts 3
EOF
}

echo -e "${GREEN}✓ LGT Training helper loaded!${NC}"
echo "Type 'lgt_help' to see available commands"
