#!/bin/bash
# Fetch SIH26038 datasets. All four are access-gated: this script automates
# what's automatable and tells you exactly what needs a browser click.
# Layout produced:
#   data/aptos2019/train_images/*.png + train.csv
#   data/idrid/grading/  data/idrid/segmentation/  data/idrid/localization/
#   data/drive/training/ data/drive/test/
#   data/messidor2/
set -u
cd "$(dirname "$0")/.."
mkdir -p data

echo "=== 1/4 APTOS 2019 (Kaggle, needs API token) ==="
if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "ACTION NEEDED (2 min, one time):"
  echo "  1. log in at kaggle.com -> Account -> API -> Create New Token (saves kaggle.json)"
  echo "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
  echo "  3. open https://www.kaggle.com/c/aptos2019-blindness-detection and click 'Join Competition'"
  echo "  4. re-run this script"
else
  pip install -q --break-system-packages kaggle 2>/dev/null || pip install -q kaggle
  kaggle competitions download -c aptos2019-blindness-detection -p data/aptos2019
  unzip -oq data/aptos2019/aptos2019-blindness-detection.zip -d data/aptos2019
  echo "APTOS: $(ls data/aptos2019/train_images 2>/dev/null | wc -l) train images"
fi

echo ""
echo "=== 2/4 IDRiD (IEEE DataPort, free registration, manual download) ==="
echo "ACTION NEEDED:"
echo "  1. register at https://ieee-dataport.org then open"
echo "     https://ieeedataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"
echo "  2. download + unpack so you get:"
echo "     data/idrid/grading/ data/idrid/segmentation/ data/idrid/localization/"

echo ""
echo "=== 3/4 DRIVE (grand-challenge.org, free registration) ==="
echo "ACTION NEEDED:"
echo "  1. register at https://drive.grand-challenge.org/ -> Data page -> download DRIVE.zip"
echo "  2. unpack so you get: data/drive/training/ data/drive/test/"

echo ""
echo "=== 4/4 Messidor-2 (adcis.net form, approval takes ~1-2 days) ==="
echo "ACTION NEEDED:"
echo "  1. request access at https://www.adcis.net/en/third-party/messidor2/"
echo "  2. unpack into: data/messidor2/"
echo ""
echo "Start steps 2-4 in a browser NOW (approvals take time); step 1 works as soon as kaggle.json exists."
