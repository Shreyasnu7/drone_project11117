#!/bin/bash
# Complete camera setup for Pi Camera V2.1 (IMX219)
set -e

echo "📷 CONFIGURING CAMERA V2.1 (IMX219) FOR STREAMING..."

# 1. Reset
sudo media-ctl -r -d /dev/media0

# 2. Configure pipeline
SENSOR="'m00_b_imx219 2-0010'"
DPHY="'rockchip-csi2-dphy0'"
CSI="'rkisp-csi-subdev'"
ISP="'rkisp-isp-subdev'"
MAIN="'rkisp_mainpath'"

echo "🔗 Linking pipeline..."
sudo media-ctl -d /dev/media0 -l "$SENSOR:0->$DPHY:0 [1]"
sudo media-ctl -d /dev/media0 -l "$DPHY:1->$CSI:0 [1]"
sudo media-ctl -d /dev/media0 -l "$CSI:1->$ISP:0 [1]"
sudo media-ctl -d /dev/media0 -l "$ISP:2->$MAIN:0 [1]"

echo "📐 Setting formats..."
# IMX219 native: SRGGB10_1X10, max 3280x2464 (8MP)
# Using 1920x1080 for drone streaming performance
sudo media-ctl -d /dev/media0 -V "$SENSOR:0 [fmt:SRGGB10_1X10/1920x1080]"
sudo media-ctl -d /dev/media0 -V "$ISP:2 [fmt:YUYV8_2X8/1920x1080]"

echo "🎬 Starting video stream on /dev/video0..."
sudo v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=NV12

# Test capture
echo "🧪 Testing capture..."
sudo timeout 3 v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=5 --stream-skip=2 2>&1 | head -10

echo "✅ Camera V2.1 (IMX219) configured and streaming on /dev/video0"
echo "   Format: NV12 1920x1080"
