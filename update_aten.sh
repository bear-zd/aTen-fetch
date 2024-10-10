#!/bin/bash
PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
TARGET_DIR_ATEN="aten-src-ATen"
TARGET_DIR_C10="aten-src-c10"
rm -rf $TARGET_DIR_ATEN
rm -rf $TARGET_DIR_C10
git clone --depth 1 $PYTORCH_REPO temp-pytorch
mkdir -p $TARGET_DIR
cp -r temp-pytorch/aten/src/ATen/* $TARGET_DIR_ATEN/
cp -r temp-pytorch/aten/src/c10/* $TARGET_DIR_C10/
cp temp-pytorch/LICENSE .
rm -rf temp-pytorch
echo "Update complete."
