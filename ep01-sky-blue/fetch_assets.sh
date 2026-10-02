#!/usr/bin/env bash
# 下载字体（及可选的离线语音模型）到 ./assets（不纳入版本库）
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets/fonts
NOTO=https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/SC
for f in NotoSansSC-Regular.otf NotoSansSC-Bold.otf; do
  [ -f "assets/fonts/$f" ] || curl -fsSL -o "assets/fonts/$f" "$NOTO/$f"
done
[ -f assets/fonts/ZCOOLKuaiLe-Regular.ttf ] || curl -fsSL -o assets/fonts/ZCOOLKuaiLe-Regular.ttf \
  https://raw.githubusercontent.com/google/fonts/main/ofl/zcoolkuaile/ZCOOLKuaiLe-Regular.ttf
if [ -z "${MINIMAX_API_KEY:-}" ]; then   # 未配置 MiniMax 时才需要离线语音模型
  REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
  [ -d assets/matcha-icefall-zh-baker ] || curl -fsSL "$REL/tts-models/matcha-icefall-zh-baker.tar.bz2" | tar xj -C assets
  [ -f assets/vocos-22khz-univ.onnx ] || curl -fsSL -o assets/vocos-22khz-univ.onnx "$REL/vocoder-models/vocos-22khz-univ.onnx"
fi
echo "assets ready"
