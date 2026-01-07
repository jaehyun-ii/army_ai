#!/bin/bash
set -e

# ===============================================
# Army AI Platform - Storage Permission Fixer
# ===============================================
# Fixes permission issues when Docker containers
# create files as root in storage_data
#
# Usage:
#   ./fix_storage_permissions.sh        - Fix all storage directories
#   ./fix_storage_permissions.sh acl    - Use ACL for fine-grained control
# ===============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Paths
STORAGE_DATA="$SCRIPT_DIR/storage_data"
STORAGE_WORK="$SCRIPT_DIR/storage_work"
STORAGE="$SCRIPT_DIR/storage"

# Get current user
CURRENT_USER="${SUDO_USER:-$(whoami)}"
CURRENT_GROUP=$(id -gn "$CURRENT_USER")

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================="
echo "Army AI - Storage Permission Fixer"
echo "========================================="
echo "User: $CURRENT_USER"
echo "Group: $CURRENT_GROUP"
echo ""

# Function: Basic permission fix (setgid + group write)
fix_basic() {
    echo "Applying basic permission fix..."
    echo "  - Owner: $CURRENT_USER:$CURRENT_GROUP"
    echo "  - Mode: 2775 (setgid + group write)"
    echo ""

    # Fix ownership
    if [ -d "$STORAGE_DATA" ]; then
        sudo chown -R "$CURRENT_USER:$CURRENT_GROUP" "$STORAGE_DATA"
        sudo chmod -R 2775 "$STORAGE_DATA"
        echo -e "${GREEN}✅ Fixed: $STORAGE_DATA${NC}"
    fi

    if [ -d "$STORAGE_WORK" ]; then
        sudo chown -R "$CURRENT_USER:$CURRENT_GROUP" "$STORAGE_WORK"
        sudo chmod -R 2775 "$STORAGE_WORK"
        echo -e "${GREEN}✅ Fixed: $STORAGE_WORK${NC}"
    fi

    if [ -d "$STORAGE" ] && mountpoint -q "$STORAGE" 2>/dev/null; then
        # Don't recursively chown merged dir (expensive), just ensure it's accessible
        sudo chmod 755 "$STORAGE"
        echo -e "${GREEN}✅ Fixed: $STORAGE (merged - lightweight)${NC}"
    fi

    echo ""
    echo -e "${GREEN}✅ Basic permission fix complete!${NC}"
    echo ""
    echo "Note: Files created by Docker containers will inherit group ownership."
}

# Function: ACL-based permission fix (more flexible)
fix_acl() {
    echo "Applying ACL-based permission fix..."
    echo "  - User ACL: $CURRENT_USER (rwx)"
    echo "  - Default ACL: Inherit for new files"
    echo ""

    # Check if setfacl is available
    if ! command -v setfacl &> /dev/null; then
        echo -e "${RED}✗ Error: setfacl not found. Install acl package:${NC}"
        echo "  sudo apt-get install acl"
        exit 1
    fi

    # Apply ACL to storage_data
    if [ -d "$STORAGE_DATA" ]; then
        echo "Setting ACL for $STORAGE_DATA..."

        # Set ACL for current user
        sudo setfacl -R -m u:${CURRENT_USER}:rwx "$STORAGE_DATA"
        sudo setfacl -R -m u:root:rwx "$STORAGE_DATA"

        # Set default ACL for new files
        sudo setfacl -R -d -m u:${CURRENT_USER}:rwx "$STORAGE_DATA"
        sudo setfacl -R -d -m u:root:rwx "$STORAGE_DATA"
        sudo setfacl -R -d -m g:${CURRENT_GROUP}:rwx "$STORAGE_DATA"

        echo -e "${GREEN}✅ ACL set: $STORAGE_DATA${NC}"

        # Show ACL
        echo "Sample ACL (first directory):"
        getfacl "$(find "$STORAGE_DATA" -type d -print -quit 2>/dev/null || echo "$STORAGE_DATA")" | head -n 15
        echo ""
    fi

    # Apply ACL to storage_work
    if [ -d "$STORAGE_WORK" ]; then
        echo "Setting ACL for $STORAGE_WORK..."
        sudo setfacl -R -m u:${CURRENT_USER}:rwx "$STORAGE_WORK"
        sudo setfacl -R -m u:root:rwx "$STORAGE_WORK"
        sudo setfacl -R -d -m u:${CURRENT_USER}:rwx "$STORAGE_WORK"
        sudo setfacl -R -d -m u:root:rwx "$STORAGE_WORK"
        echo -e "${GREEN}✅ ACL set: $STORAGE_WORK${NC}"
    fi

    echo ""
    echo -e "${GREEN}✅ ACL permission fix complete!${NC}"
    echo ""
    echo "Note: Both $CURRENT_USER and root can now read/write all files."
}

# Function: Show current permissions
show_status() {
    echo "Current permissions:"
    echo ""

    if [ -d "$STORAGE_DATA" ]; then
        echo "storage_data:"
        ls -ld "$STORAGE_DATA"
        echo "  Sample files:"
        find "$STORAGE_DATA" -type f -o -type d | head -n 5 | xargs ls -ld 2>/dev/null || true
        echo ""
    fi

    if [ -d "$STORAGE_WORK" ]; then
        echo "storage_work:"
        ls -ld "$STORAGE_WORK"
        echo ""
    fi

    if [ -d "$STORAGE" ]; then
        echo "storage (merged):"
        ls -ld "$STORAGE"
        mountpoint -q "$STORAGE" && echo "  Status: OverlayFS mounted" || echo "  Status: Not mounted"
        echo ""
    fi
}

# Main
case "${1:-basic}" in
    basic)
        fix_basic
        ;;
    acl)
        fix_acl
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {basic|acl|status}"
        echo ""
        echo "Commands:"
        echo "  basic  - Fix using chown + setgid (default, simple)"
        echo "  acl    - Fix using ACL (flexible, allows both user and root)"
        echo "  status - Show current permissions"
        echo ""
        echo "Recommendation:"
        echo "  - Use 'basic' for most cases (simple, fast)"
        echo "  - Use 'acl' if you need both host user and container root to coexist"
        exit 1
        ;;
esac

echo "========================================="
