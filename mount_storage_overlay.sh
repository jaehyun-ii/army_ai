#!/bin/bash
set -e

# ===============================================
# Army AI Platform - OverlayFS Mount Script
# ===============================================
# Mounts storage_base (RO) + storage_data (RW) → storage (merged)
# Usage:
#   ./mount_storage_overlay.sh mount   - Mount OverlayFS
#   ./mount_storage_overlay.sh umount  - Unmount OverlayFS
#   ./mount_storage_overlay.sh status  - Check mount status
# ===============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Paths
LOWER_DIR="$SCRIPT_DIR/storage_base"
UPPER_DIR="$SCRIPT_DIR/storage_data"
WORK_DIR="$SCRIPT_DIR/storage_work"
MERGED_DIR="$SCRIPT_DIR/storage"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function: Check if already mounted
is_mounted() {
    mountpoint -q "$MERGED_DIR" 2>/dev/null
}

# Function: Mount OverlayFS
mount_overlay() {
    echo "========================================="
    echo "Army AI Platform - OverlayFS Setup"
    echo "========================================="

    # Check if already mounted
    if is_mounted; then
        echo -e "${YELLOW}⚠ OverlayFS already mounted at: $MERGED_DIR${NC}"
        mount | grep "$MERGED_DIR"
        return 0
    fi

    # Create directories if they don't exist
    echo "Creating directories..."
    mkdir -p "$LOWER_DIR"
    mkdir -p "$UPPER_DIR"
    mkdir -p "$WORK_DIR"
    mkdir -p "$MERGED_DIR"

    # Verify directories exist
    if [ ! -d "$LOWER_DIR" ]; then
        echo -e "${RED}✗ Error: Lower directory not found: $LOWER_DIR${NC}"
        exit 1
    fi

    if [ ! -d "$UPPER_DIR" ]; then
        echo -e "${RED}✗ Error: Upper directory not found: $UPPER_DIR${NC}"
        exit 1
    fi

    if [ ! -d "$WORK_DIR" ]; then
        echo -e "${RED}✗ Error: Work directory not found: $WORK_DIR${NC}"
        exit 1
    fi

    # Mount OverlayFS
    echo "Mounting OverlayFS..."
    sudo mount -t overlay overlay \
        -o lowerdir="$LOWER_DIR",upperdir="$UPPER_DIR",workdir="$WORK_DIR" \
        "$MERGED_DIR"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ OverlayFS mounted successfully!${NC}"
        echo ""
        echo "Configuration:"
        echo "  Lower (RO):  $LOWER_DIR"
        echo "  Upper (RW):  $UPPER_DIR"
        echo "  Work:        $WORK_DIR"
        echo "  Merged:      $MERGED_DIR"
        echo ""
        mount | grep "$MERGED_DIR"
        echo "========================================="
    else
        echo -e "${RED}✗ Error: Failed to mount OverlayFS${NC}"
        exit 1
    fi
}

# Function: Unmount OverlayFS
umount_overlay() {
    echo "========================================="
    echo "Army AI Platform - OverlayFS Unmount"
    echo "========================================="

    if ! is_mounted; then
        echo -e "${YELLOW}⚠ OverlayFS not mounted at: $MERGED_DIR${NC}"
        return 0
    fi

    echo "Unmounting OverlayFS..."
    sudo umount "$MERGED_DIR"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ OverlayFS unmounted successfully!${NC}"
        echo "========================================="
    else
        echo -e "${RED}✗ Error: Failed to unmount OverlayFS${NC}"
        echo "Try: sudo umount -l $MERGED_DIR (lazy unmount)"
        exit 1
    fi
}

# Function: Show status
show_status() {
    echo "========================================="
    echo "Army AI Platform - OverlayFS Status"
    echo "========================================="

    if is_mounted; then
        echo -e "${GREEN}✅ OverlayFS is MOUNTED${NC}"
        echo ""
        mount | grep "$MERGED_DIR"
        echo ""
        echo "Directories:"
        echo "  Lower (RO):  $LOWER_DIR"
        echo "  Upper (RW):  $UPPER_DIR"
        echo "  Work:        $WORK_DIR"
        echo "  Merged:      $MERGED_DIR"
    else
        echo -e "${YELLOW}⚠ OverlayFS is NOT MOUNTED${NC}"
        echo ""
        echo "To mount, run:"
        echo "  $0 mount"
    fi
    echo "========================================="
}

# Main
case "${1:-}" in
    mount)
        mount_overlay
        ;;
    umount|unmount)
        umount_overlay
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {mount|umount|status}"
        echo ""
        echo "Commands:"
        echo "  mount   - Mount OverlayFS (storage_base + storage_data → storage)"
        echo "  umount  - Unmount OverlayFS"
        echo "  status  - Check mount status"
        exit 1
        ;;
esac
