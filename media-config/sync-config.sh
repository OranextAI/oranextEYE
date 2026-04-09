#!/bin/bash
# Sync mediamtx.yml from project to ~/mediamtx-config/
# Run this after editing oranextEYE/media-config/mediamtx.yml
# then restart the container: sudo docker restart mediamtx

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/mediamtx.yml" "$HOME/mediamtx-config/mediamtx.yml"
echo "✅ Config synced to ~/mediamtx-config/mediamtx.yml"
echo "🔄 Restarting mediamtx container..."
sudo docker restart mediamtx
echo "✅ Done"
