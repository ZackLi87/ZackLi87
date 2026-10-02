# MiniMax 语音合成调用方法

`minimax_tts.py` 是一个独立脚本，只依赖 Python 标准库（3.8 及以上），可复制到任何项目中使用。

## 一、准备密钥

1. 登录 MiniMax 开放平台（国内站 platform.minimaxi.com），在“账户管理 → 接口密钥”中创建密钥。
2. 在本目录新建 `minimax_key.txt`，把密钥粘贴进去，文件中只保留这一行。
3. `minimax_key.txt` 已列入仓库的 `.gitignore`，不会被提交。复制到其他项目时，也请把该文件名加入
   那个项目的 `.gitignore`。

脚本按以下顺序读取密钥：

1. 命令行参数 `--key-file 路径`
2. 环境变量 `MINIMAX_API_KEY`
3. 与脚本同目录的 `minimax_key.txt`

## 二、命令行用法

```bash
# 合成一句话（默认输出 output.mp3）
python3 minimax_tts.py "你好，欢迎收听。" -o hello.mp3

# 从文本文件读取文稿
python3 minimax_tts.py -f 文稿.txt -o 文稿.mp3

# 指定音色、语速、情绪
python3 minimax_tts.py "今天天气真好！" -o a.mp3 --voice "Chinese (Mandarin)_Warm_Bestie" --speed 1.1 --emotion happy

# 插入停顿：<#秒数#>
python3 minimax_tts.py "第一句。<#0.8#>第二句。" -o pause.mp3

# 输出 wav
python3 minimax_tts.py "测试" -o test.wav --format wav

# 列出系统音色（可加关键词筛选）
python3 minimax_tts.py --list-voices
python3 minimax_tts.py --list-voices 男声
```

### 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `text` / `-f` | — | 要朗读的文字，或 UTF-8 文本文件 |
| `-o` | `output.mp3` | 输出文件 |
| `--voice` | `Chinese (Mandarin)_Gentle_Senior` | 音色 ID，用 `--list-voices` 查询 |
| `--model` | `speech-2.6-hd` | 模型；也可用 `speech-02-hd` 等 |
| `--speed` | `1.0` | 语速，0.5–2.0 |
| `--vol` | `1.0` | 音量，0–10 |
| `--pitch` | `0` | 音调，−12–12 |
| `--emotion` | 无 | `happy`、`sad`、`angry`、`fearful`、`disgusted`、`surprised`、`neutral` |
| `--format` | `mp3` | `mp3`、`wav`、`flac`、`pcm` |
| `--host` | `api.minimaxi.com` | 国内站；国际站密钥改用 `api.minimax.io` |
| `--key-file` | — | 指定密钥文件 |

### 常用普通话音色

| 音色 ID | 名称 |
| --- | --- |
| `Chinese (Mandarin)_Gentle_Senior` | 温柔学姐 |
| `Chinese (Mandarin)_Warm_Bestie` | 温暖闺蜜 |
| `Chinese (Mandarin)_News_Anchor` | 新闻女声 |
| `Chinese (Mandarin)_Wise_Women` | 阅历姐姐 |
| `Chinese (Mandarin)_Radio_Host` | 电台男主播 |
| `Chinese (Mandarin)_Gentleman` | 温润男声 |
| `Chinese (Mandarin)_Lyrical_Voice` | 抒情男声 |
| `Chinese (Mandarin)_Male_Announcer` | 播报男声 |
| `Chinese (Mandarin)_Reliable_Executive` | 沉稳高管 |

完整列表以 `--list-voices` 的输出为准。

## 三、在 Python 中调用

```python
from minimax_tts import load_key, synthesize

key = load_key()                      # 读取 minimax_key.txt 或环境变量
seconds, chars = synthesize("你好，欢迎收听。", "hello.mp3", key,
                            voice="Chinese (Mandarin)_Radio_Host", speed=0.9)
print(seconds, chars)
```

## 四、不用脚本，直接调用接口

接口：`POST https://api.minimaxi.com/v1/t2a_v2`，请求头 `Authorization: Bearer <密钥>`。
返回 JSON 中 `data.audio` 为十六进制编码的音频数据。

```bash
KEY=$(cat minimax_key.txt)
curl -s https://api.minimaxi.com/v1/t2a_v2 \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
        "model": "speech-2.6-hd",
        "text": "你好，欢迎收听。",
        "stream": false,
        "voice_setting": {"voice_id": "Chinese (Mandarin)_Gentle_Senior", "speed": 1.0, "vol": 1.0, "pitch": 0},
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1}
      }' \
  | python3 -c "import sys, json; open('hello.mp3', 'wb').write(bytes.fromhex(json.load(sys.stdin)['data']['audio']))"
```

## 五、常见问题

| 现象 | 处理 |
| --- | --- |
| 状态码 2049，`invalid api key` | 密钥与站点不匹配：国内站密钥用 `api.minimaxi.com`，国际站密钥用 `api.minimax.io`；或密钥已失效 |
| 状态码 1004 | 鉴权失败，检查 `minimax_key.txt` 是否只有密钥一行、没有多余字符 |
| 状态码 1008 | 余额不足，到平台充值 |
| 状态码 1002 | 请求过于频繁，稍后重试 |
| 状态码 2013 | 参数错误，如音色 ID 拼写有误、语速超出范围 |
| 无法连接 | 当前网络无法访问 MiniMax 接口 |

## 六、安全提示

- 密钥只保存在本地文件或环境变量中，不要写进代码、文档或提交记录。
- 密钥一旦在聊天、截图或公开仓库中出现过，应到平台删除并重新生成。
- 按计费字符收费，脚本每次合成后会打印本次的计费字符数。
