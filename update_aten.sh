#!/bin/bash
PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
TARGET_DIR="aten-src-ATen"
rm -rf $TARGET_DIR
git clone --depth 1 $PYTORCH_REPO temp-pytorch
mkdir -p $TARGET_DIR
cp -r temp-pytorch/aten/src/ATen/* $TARGET_DIR/
cp temp-pytorch/LICENSE $TARGET_DIR/
rm -rf temp-pytorch
echo "Update complete."