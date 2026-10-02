#!/usr/bin/env bash
# 下载渲染所需的字体与离线中文语音模型到 ./assets（不纳入版本库）
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets/fonts
GH=https://raw.githubusercontent.com/notofonts/noto-cjk/main
for f in Sans/SubsetOTF/SC/NotoSansSC-Regular.otf Sans/SubsetOTF/SC/NotoSansSC-Bold.otf \
         Serif/SubsetOTF/SC/NotoSerifSC-Bold.otf; do
  [ -f "assets/fonts/$(basename $f)" ] || curl -fsSL -o "assets/fonts/$(basename $f)" "$GH/$f"
done
REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
[ -d assets/matcha-icefall-zh-baker ] || curl -fsSL "$REL/tts-models/matcha-icefall-zh-baker.tar.bz2" | tar xj -C assets
[ -f assets/vocos-22khz-univ.onnx ] || curl -fsSL -o assets/vocos-22khz-univ.onnx "$REL/vocoder-models/vocos-22khz-univ.onnx"
echo "assets ready"
