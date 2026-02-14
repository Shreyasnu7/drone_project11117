#!/bin/bash
echo "🕵️ PROBING CAMERA FORMATS..."

# 1. Identify Subdevs
# Sensor is usually /dev/v4l-subdev3 (from previous logs)
# DPHY is usually /dev/v4l-subdev2

echo "--- SENSOR CAPABILITIES (/dev/v4l-subdev3) ---"
v4l2-ctl -d /dev/v4l-subdev3 --list-subdev-mbus-codes
v4l2-ctl -d /dev/v4l-subdev3 --list-subdev-frame-sizes pad=0

echo -e "\n--- DPHY CAPABILITIES (/dev/v4l-subdev2) ---"
v4l2-ctl -d /dev/v4l-subdev2 --list-subdev-mbus-codes

echo -e "\n--- ISP CAPABILITIES (/dev/v4l-subdev0) ---"
v4l2-ctl -d /dev/v4l-subdev0 --list-subdev-mbus-codes
