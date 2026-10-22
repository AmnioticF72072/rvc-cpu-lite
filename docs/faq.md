# 报错对照表

下面这些是我自己跑出来过的，按报错信息排序。

## `Reshape ... input_shape_size == size was false`

```
Invalid input shape: {1,2,8008201}, requested shape: {1,2,2002,4001}
```

生成器的定长对不上。三种可能：

- 模型的 `--T` 和 `rvc_cpu/infer.py` 里的 `GEN_T` 不一致 → 改成一样。
- 你改过分块参数，块长超过 `GEN_T` → 看 README 里那条约束。
- 模型本来就导成了动态 shape → 那导错了，重新定长导。

报错里的形状数字不用看，`{1,2,2002,4001}` 这种是被 `int(length)-1` 那行算出来的假象，
它并不对应你输入的任何维度，照着它查方向会跑偏。

## `Non-zero status code returned while running Conv node ... Invalid input shape: {1}`

```
Status Message: Invalid input shape: {1}
```

ContentVec 收到了一段太短的音频。常见于切点算到文件末尾，剩下几十毫秒甚至一个空片段。
本仓库在 `convert_file` 里做了过滤（每段至少 1.5 秒、离结尾不到 1.5 秒的切点丢掉），
如果你自己改了 `X_CENTER` 之类的参数，把段长又切短了，这个错会回来。

## 输出比输入短

不是 bug 是参数问题。相邻切点跨度加两侧上下文超过 `GEN_T` 帧，尾部被截掉了。
我最早跑 20 秒素材出 18.86 秒，就是差这一条。默认参数下应该是等长的，
允许有几十毫秒的块边界误差（每块多几帧），不会差到秒级。

## 输出比输入长

通常是重复做了上下文裁剪，或者分块之后又整体裁了一次。
官方 pipeline 是**每块**裁 `[t_pad_tgt:-t_pad_tgt]`，不要在外面再裁一遍。
早期版本这两处都做了，20 秒素材出 23.11 秒。

## 声音发闷、发糊，像隔着一层

- `--key` 给错了。男声材料降到 -12 会闷，那是正常的物理结果，不是模型坏了。
- 素材本身有伴奏或者环境噪声。RVC 只该喂干声，先分离或者找干净录音。
- 生成器采样率和原素材差太多也有一点影响，但这个影响远小于前两条。

## `Couldn't open shared file mapping` / `HOST_ALLOCATION_FAILED`

主机内存不够，不是显存。这台测试机 16G，同时开游戏和浏览器时，导出大模型会撞上。
导出时把别的程序关掉，或者把 `--T` 降到 768 试试。

## 杀毒软件报毒 / 便携包被删

便携版里有 python.exe，某些杀软对"文件夹里带解释器"的组合很敏感，直接当木马。
加白名单，或者改用自己装 Python 的方式跑源码。

## `ImportError: DLL load failed while importing onnxruntime`

VC++ 运行库缺失，或者 Python 位数不对（32 位解释器加载不了 64 位 wheel）。
装一下 Microsoft Visual C++ Redistributable 2015-2022。
