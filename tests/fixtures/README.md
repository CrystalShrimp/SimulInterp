# 手工推理验证样本

这些音频仅供手工模型验证，单元测试不会自动读取或下载模型。

- `english-smoke.wav`：本机 Windows SAPI / Microsoft Zira 合成语音，9.868 秒。文本为本项目编写：
  “Good morning. Today we are talking about language learning and technology. You can read Chinese subtitles while listening to this presentation.”
- `german-guten-morgen.ogg`：Jeuwre 录制的德语发音 **guten Morgen**，原始文件未修改，约 2.2 秒。
  来源：[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:De-guten_Morgen.ogg)。
  本音频采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)；作者 [Jeuwre](https://commons.wikimedia.org/wiki/User:Jeuwre)。
  原始 SHA-1：`2cb2e778bf662434fc81b4d62e8d0eb74ddb7dae`。

```powershell
.\.venv\Scripts\python.exe -m simulinterp.transcribe --audio tests\fixtures\english-smoke.wav --language en --output exports\english-smoke.srt
.\.venv\Scripts\python.exe -m simulinterp.transcribe --audio tests\fixtures\german-guten-morgen.ogg --language de --output exports\german-smoke.srt
```
