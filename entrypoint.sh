#!/bin/sh
set -e

prometheus-node-exporter &

sleep 0.3

exec python app.py
