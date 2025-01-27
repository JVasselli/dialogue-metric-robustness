#!/bin/bash
# Generate adversarial benchmark for TopicalChat
python benchmark/generate_benchmark.py -i data/topicalchat_subset.json -o data/topicalchat_attacks.json

# Generate adversarial benchmark for DailyDialog
python benchmark/generate_benchmark.py -i data/dailydialog_subset.json -o data/dailydialog_attacks.json
