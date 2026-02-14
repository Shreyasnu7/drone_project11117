#!/bin/bash
# Install GStreamer for sustained camera streaming
set -e

echo "📦 Installing GStreamer..."
sudo apt-get update
sudo apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good v4l-utils

echo "✅ GStreamer installed"
echo "Test with: gst-launch-1.0 v4l2src device=/dev/video0 ! fakesink"
