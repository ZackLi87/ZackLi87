# MiniMax 语音合成使用说明

本仓库的视频配音使用 MiniMax 语音合成（T2A v2）。本文说明密钥的保存方式、单独合成语音的方法、
在视频脚本中的用法以及常见问题。

## 一、保存密钥

密钥只保存在本地文件中，**不得提交到 GitHub**（本仓库为公开仓库时，任何人都能看到提交内容）。

1. 在 MiniMax 开放平台（国内站 platform.minimaxi.com）的“账户管理 → 接口密钥”中创建密钥。
2. 在仓库根目录新建文件 `.minimax_key`，内容只有一行，即密钥本身：

   ```bash
   cd ZackLi87
   echo "你的密钥" > .minimax_key
   chmod 600 .minimax_key        # 仅本人可读（macOS / Linux）
   ```

   Windows 可用记事本新建该文件，注意文件名为 `.minimax_key`，不要带 `.txt` 后缀。

3. `.minimax_key` 已列入 `.gitignore`，执行 `git status` 时不会出现，也不会被提交。

程序按以下顺序读取密钥：

| 顺序 | 来源 | 适用情形 |
| --- | --- | --- |
| 1 | 环境变量 `MINIMAX_API_KEY` | 临时使用，或在云端环境中配置 |
| 2 | 环境变量 `MINIMAX_KEY_FILE` 指定的文件 | 密钥文件放在仓库以外的位置 |
| 3 | 仓库根目录的 `.minimax_key` | 日常使用（推荐） |

三者都没有时，视频脚本自动改用离线语音模型（音质较差），并在终端提示。

> 密钥曾在对话中出现过的，建议在平台上删除旧密钥、重新生成一个，再写入 `.minimax_key`。

## 二、单独合成一段语音

```bash
pip install numpy pillow                       # 另需安装 ffmpeg

# 用默认音色（温柔学姐）合成
python3 tts_say.py "小朋友，抬头看看天空。" -o hello.mp3

# 指定音色、语速、情绪
python3 tts_say.py "千江有水千江月。" -o zen.mp3 \
    --voice "Chinese (Mandarin)_Radio_Host" --speed 0.82

python3 tts_say.py "嘭！夜空中绽放出烟花。" -o fire.mp3 --emotion happy

# 插入停顿：<#秒数#>
python3 tts_say.py "不是风动，<#0.3#>不是幡动，<#0.3#>仁者心动。" -o koan.mp3

# 列出可用的普通话系统音色
python3 tts_say.py --list-voices
```

### 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--voice` | `Chinese (Mandarin)_Gentle_Senior` | 音色 ID，可用 `--list-voices` 查询 |
| `--model` | `speech-2.6-hd` | 语音模型；`speech-02-hd` 亦可用 |
| `--speed` | `1.0` | 语速，范围 0.5–2.0 |
| `--emotion` | 无 | `happy`、`sad`、`angry`、`fearful`、`disgusted`、`surprised`、`neutral` |
| `--host` | `api.minimaxi.com` | 国内站地址；国际站密钥改用 `api.minimax.io` |
| `-o` | `tts.mp3` | 输出文件 |

### 本项目用过的音色

| 音色 ID | 名称 | 用于 |
| --- | --- | --- |
| `Chinese (Mandarin)_Gentle_Senior` | 温柔学姐 | 好奇心实验室第 1–5 期（`--emotion happy`，语速 1.0） |
| `Chinese (Mandarin)_Radio_Host` | 电台男主播 | 禅意视频《照见》（语速 0.82） |
| `Chinese (Mandarin)_Gentleman` | 温润男声 | 备选 |
| `Chinese (Mandarin)_Lyrical_Voice` | 抒情男声 | 备选 |

## 三、在视频脚本中使用

保存好 `.minimax_key` 后，直接运行各期脚本即可，无需额外设置：

```bash
python3 ep03-rainbow/make_video.py --cover cover
python3 ep03-rainbow/make_video.py --voice "Chinese (Mandarin)_Warm_Bestie" --speed 1.05
```

- 每句旁白合成后缓存在 `assets/tts_cache/`。缓存以“文字 + 音色 + 模型 + 语速 + 情绪”为键，
  修改其中任一项会重新合成，未修改的句子不会重复计费。
- 文案中需要停顿时，可在 `SCENES` 的 `say` 字段中使用 `<#秒数#>`（字幕显示 `show` 字段，不受影响）。

在代码中调用：

```python
import vidkit as vk

tts = vk.MiniMaxTTS(vk.minimax_key(), "Chinese (Mandarin)_Gentle_Senior", speed=1.0, emotion="happy")
tts("你好，欢迎来到好奇心实验室。", "hello.mp3")
```

## 四、常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `invalid api key`（状态码 2049） | 密钥与站点不匹配：国内站密钥只能用 `api.minimaxi.com`，国际站密钥用 `api.minimax.io`；或密钥已删除，重新生成 |
| 状态码 1004 | 鉴权失败，检查 `.minimax_key` 内容是否完整、是否有多余字符 |
| 状态码 1008 | 账户余额不足，到平台充值 |
| 状态码 1002 | 请求过于频繁，稍后重试 |
| 状态码 2013 | 参数错误，如音色 ID 拼写有误、语速超出范围 |
| 连接被拒绝、超时 | 网络无法访问 MiniMax；云端环境需在网络设置中放行 `api.minimaxi.com` |
| 终端提示“使用离线语音模型” | 没有读取到密钥，检查 `.minimax_key` 是否位于仓库根目录 |
