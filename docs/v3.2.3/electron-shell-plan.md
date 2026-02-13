# AnchorFlux Electron 壳方案

> 状态: 待实施 | 预估工作量: 4-5 天 | 创建日期: 2026-02-12

## 1. 目标

将 AnchorFlux 前端包裹在 Electron 窗口中，使其从"浏览器标签页"升级为"桌面应用"。
同时利用 Chromium 原生 HEVC 硬解能力，消除 H265 视频的强制转码环节。

**核心原则：Electron 只做壳，后端完全不动。**

---

## 2. 架构对比

### 2.1 当前架构

```
用户双击 AnchorFlux.exe (PyInstaller 2MB 壳)
    |
bootloader.py
    |-- 定位 tools/python/python.exe
    |-- 检查依赖哈希 -> 按需安装
    |-- 启动 uvicorn (port 8000, 托管 frontend/dist/)
    '-- 打开浏览器标签页 http://localhost:8000
```

### 2.2 目标架构

```
用户双击 AnchorFlux.exe
    |
bootloader.py (不变)
    |-- 启动 uvicorn (port 8000)
    |
同时启动 Electron 窗口
    |-- 显示 loading.html ("后端启动中...")
    |-- 轮询 http://localhost:8000/api/heartbeat
    '-- 后端就绪 -> loadURL('http://localhost:8000')
```

**关键变化：bootloader 的"打开浏览器"改为"启动 Electron 窗口"。其余一切不变。**

---

## 3. 为什么可以这么简单

经调查确认，项目前端天然适配 Electron 壳方案：

| 维度 | 现状 | 影响 |
|------|------|------|
| API 通信 | `client.js:32` 硬编码 `http://localhost:8000` | 不依赖相对路径代理，Electron 直接可用 |
| SSE 连接 | `sseChannelManager.js:34` 硬编码 `http://localhost:8000` | 绝对地址，不受同源限制 |
| 视频播放 | 原生 `<video>` 标签 + HTTP URL | 无 Blob URL / MSE，Electron 完全兼容 |
| WaveSurfer | 独立 audio 元素，通过 PlaybackManager 同步 | 不依赖 video 元素，替换播放器不影响波形 |
| 前端框架 | Vue 3 + Axios + EventSource（纯 Web 技术栈） | 无任何 Node.js / Electron 特有 API 调用 |

**前端代码零改动。后端代码零改动。**

---

## 4. HEVC 免转码方案

### 4.1 技术原理

Chrome/Electron 在 Windows 上使用 `D3D11VideoDecoder` 直接调用 `D3D11VA` 接口与 GPU 驱动对话，
**绕过了 Windows Media Foundation (MFT) 路径**，因此不需要用户安装付费的 HEVC Video Extensions 插件。

| 解码路径 | 使用者 | 是否需要 HEVC 插件 |
|---------|--------|-------------------|
| MFT (Media Foundation) | Edge、Firefox、系统播放器 | 需要（付费 7 元） |
| D3D11VideoDecoder -> D3D11VA | Chrome / Electron | **不需要** |

参考: [enable-chromium-hevc-hardware-decoding](https://github.com/StaZhu/enable-chromium-hevc-hardware-decoding)

### 4.2 硬解覆盖率分析

HEVC 硬解的 GPU 起点严格早于 CUDA 12.8 的 GPU 起点：

| GPU 厂商 | HEVC 硬解最低要求 | 年份 | AnchorFlux CUDA 最低要求 |
|---------|------------------|------|------------------------|
| Intel 核显 | 第 6 代 Skylake (HD 520) | 2015 | N/A（非 CUDA 设备） |
| NVIDIA 独显 | GTX 650 (Kepler) | 2012 | GTX 750Ti (Maxwell, CUDA 12.8) |
| AMD 独显 | RX 460 (Polaris) | 2016 | N/A（非 CUDA 设备） |
| AMD APU | Ryzen 2200G (Vega) | 2018 | N/A |

**结论：能跑 AnchorFlux 的 GPU，100% 支持 HEVC 硬解。不需要软解回退补丁。**

唯一的理论边界：2015 年前的纯 Intel 核显老机器（第 4/5 代），但这些机器跑 ONNX 推理性能极差，
本身不是目标用户。对这 0.1% 的边界场景，保留现有 FFmpeg 转码流程作为兜底即可。

### 4.3 播放条件（仅两个）

1. Electron >= v22.0.0（对应 Chromium 107+，HEVC 硬解已内置）
2. 用户 GPU 支持 HEVC 硬解（2015 年后的显卡基本都支持）

不需要：
- HEVC Video Extensions 付费插件
- 自编译 Chromium / Electron
- 替换 ffmpeg.dll
- 任何系统级编解码器安装

### 4.4 容器兼容性

Chromium `<video>` 标签对部分容器格式不支持，需通过 remux（重封装）处理：

| 容器 | Chromium 支持 | 处理方式 | 耗时 |
|------|-------------|---------|------|
| MP4 (.mp4/.m4v) | 原生支持 | 直接播放 | 0 |
| MOV (.mov) | 原生支持 | 直接播放 | 0 |
| MKV (.mkv) | 不支持 | FFmpeg remux -> MP4 | 3-10 秒（零转码） |
| AVI (.avi) | 不支持 | FFmpeg remux -> MP4 | 3-10 秒 |
| WebM (.webm) | 原生支持 | 直接播放 | 0 |

remux 和 transcode 的开销天壤之别：
```
remux:     ffmpeg -i input.mkv -c copy output.mp4     -> 3-10 秒（零转码，只改容器）
transcode: ffmpeg -i input.mkv -c:v libx264 output.mp4 -> 5-15 分钟（完整重编码）
```

### 4.5 修改后的转码决策流程

```
用户导入视频 -> FFprobe 分析编码和容器（已有逻辑）
    |
    |-- H264 + MP4     -> DIRECT_PLAY（不变）
    |-- H264 + MKV     -> REMUX_ONLY（不变，几秒）
    |-- H265 + MP4     -> 检测硬解能力
    |   |-- 支持       -> DIRECT_PLAY（新增，零转码）
    |   '-- 不支持     -> TRANSCODE_FULL（现有逻辑，回退）
    |-- H265 + MKV     -> 检测硬解能力
    |   |-- 支持       -> REMUX_ONLY（新增，几秒 remux 到 MP4）
    |   '-- 不支持     -> TRANSCODE_FULL（现有逻辑，回退）
    '-- VP9/AV1/其他   -> 按需处理
```

前端检测硬解能力：
```javascript
const result = await navigator.mediaCapabilities.decodingInfo({
  type: 'file',
  video: { contentType: 'video/mp4; codecs="hvc1"', width: 1920, height: 1080, framerate: 30, bitrate: 10000000 }
});
// result.supported && result.powerEfficient -> 硬解可用
```

---

## 5. Electron 壳文件结构

```
electron/
|-- main.js          # ~80 行，窗口创建 + 后端等待 + 生命周期
|-- preload.js       # ~10 行，基本空（不需要 IPC）
|-- loading.html     # ~30 行，启动等待页
|-- package.json     # electron + electron-builder 配置
'-- icons/           # 应用图标（.ico / .png）
```

### 5.1 main.js 核心逻辑

```
1. 创建 BrowserWindow（无菜单栏、自定义标题、最小尺寸）
2. 加载 loading.html（"后端启动中..."）
3. 轮询 http://localhost:8000/api/heartbeat（间隔 1 秒，超时 60 秒）
4. 后端就绪 -> win.loadURL('http://localhost:8000')
5. 窗口关闭 -> app.quit()（后端由 bootloader 独立管理生命周期）
```

### 5.2 启动流程

```
用户双击 AnchorFlux.exe
    |
bootloader.py 初始化（不变）
    |-- 定位 Python 运行时
    |-- 检查依赖
    |-- 启动 uvicorn (port 8000)
    |
    |-- 启动 Electron 窗口（替代原来的"打开浏览器"）
    |   |-- 显示 loading.html
    |   |-- 轮询 localhost:8000
    |   '-- 就绪后切换到主界面
    |
    '-- 守护循环（检测后端退出、更新信号等，不变）
```

### 5.3 bootloader.py 改动

仅需修改一处：将"打开浏览器"替换为"启动 Electron"。

改动前：
```python
subprocess.Popen(['cmd', '/c', 'start', '', target_url])
```

改动后：
```python
electron_exe = os.path.join(BASE_DIR, 'electron', 'AnchorFlux.exe')
subprocess.Popen([electron_exe])
```

---

## 6. HTMLVideoElement API 依赖面

已审计前端对 `<video>` 标签的完整依赖，确认 Electron 原生 `<video>` 100% 覆盖：

### 高频（核心同步链路）

| API | 用途 | 文件位置 |
|-----|------|---------|
| currentTime（读写） | seek 操作、时间同步 | VideoStage/index.vue, PlaybackManager.js |
| duration（读） | 边界计算 | VideoStage/index.vue |
| paused（读） | 状态判断 | VideoStage/index.vue |
| play() / pause() | 播放控制 | VideoStage/index.vue |
| timeupdate 事件 | PlaybackManager 核心同步 | VideoStage/index.vue, PlaybackManager.js |
| loadedmetadata 事件 | 初始化时长和速率 | VideoStage/index.vue |

### 中频

| API | 用途 |
|-----|------|
| playbackRate（读写） | 倍速播放 |
| volume（读写） | 音量控制 |
| load() | 重载视频源 |
| play/pause/ended 事件 | 状态同步 |

### 低频

| API | 用途 |
|-----|------|
| error / error.code | 错误处理 |
| seeking/seeked 事件 | seek 状态 |
| waiting/canplay 事件 | 缓冲提示 |

**关键发现：WaveSurfer 使用独立的 audio 元素（非 video），二者通过 PlaybackManager 同步时间。
Electron 替换不影响波形图。**

---

## 7. 被否决的方案

### 7.1 mpv.js + electron-plugin-structure

- mpv.js 已停更 6 年（2018-2019）
- 核心技术 PPAPI 已被 Chromium 彻底移除（2022.6，Electron 20+）
- 结论：技术路线已死，不可用

### 7.2 libVLC / GStreamer + HTMLMediaElement shim

- libVLC 的 Electron 绑定 (WebChimera.js) 明确标记 ABANDONED
- GStreamer 的 Node.js 绑定 (gstreamer-superficial) 最后更新 2016 年
- HTMLMediaElement shim 全世界没有先例，需从零实现完整媒体播放器状态机
- 结论：绑定层全部死亡，工程量月级别，不推荐

### 7.3 castlabs/electron-releases

- 定位是 DRM/Widevine 支持，不是 HEVC 解码
- README 完全没有提及 H265/HEVC
- 结论：方向不对，解决的是不同问题

### 7.4 StaZhu Release 直接使用

- Release 提供的是完整 Chromium 浏览器安装包（chromium_xxx_win32_x64.exe），不是 Electron
- 不能把 Chromium 浏览器当 Electron 用
- 结论：产物类型不匹配

### 7.5 自编译 Electron / 替换 ffmpeg.dll（为软解回退）

- Chromium 的 D3D11VA 硬解路径不依赖 HEVC 插件，标准 Electron 22+ 开箱即用
- AnchorFlux 用户 GPU（CUDA 12.8 起步）100% 覆盖 HEVC 硬解
- 0.1% 边界场景（古老纯核显）由现有 FFmpeg 转码流程兜底
- 结论：不需要为软解回退去研究补丁或自编译

---

## 8. 工作清单

| # | 工作项 | 工作量 | 改动范围 |
|---|--------|--------|---------|
| 1 | Electron 项目初始化 | 2 小时 | `electron/` 新目录 |
| 2 | main.js + preload.js + loading.html | 半天 | `electron/` |
| 3 | electron-builder 打包配置 | 半天 | `electron/package.json` |
| 4 | bootloader.py 启动方式修改 | 1 小时 | `bootloader.py` 一处 |
| 5 | 窗口行为调优（托盘、最小化、标题） | 半天 | `electron/main.js` |
| 6 | 前端 HEVC 硬解检测（MediaCapabilities） | 半天 | 前端新增 ~20 行 |
| 7 | 后端转码决策修改（H265 走 DIRECT_PLAY） | 半天 | `media_prep_service.py` ~10 行 |
| 8 | MKV 容器 remux 路径验证 | 半天 | 验证现有 REMUX_ONLY 逻辑 |
| 9 | H264/H265 x MP4/MKV 矩阵测试 | 1 天 | 手动测试 |
| **合计** | | **4-5 天** | |

---

## 9. 体积影响

| 组件 | 增量 |
|------|------|
| Electron 运行时 | +80MB |
| Python 运行时 + 依赖 | 不变（~2.7GB） |
| 前端静态文件 | 不变（~5MB） |
| FFmpeg | 不变（~100MB） |
| 模型文件 | 不变（~1-3GB） |

Electron 的 80MB 相对于 2.7GB 的 Python 运行时可忽略（+3%）。

---

## 10. 用户体验收益

### 桌面应用体验

| 维度 | 之前（浏览器标签页） | 之后（Electron 窗口） |
|------|-------------------|---------------------|
| 窗口 | 浏览器标签页，混在其他标签中 | 独立桌面窗口，任务栏独立图标 |
| 关闭行为 | 关闭标签页后端仍在跑 | 关闭窗口 = 退出应用 |
| 最小化 | 最小化整个浏览器 | 可最小化到托盘 |
| 标题栏 | 浏览器标题栏 + 地址栏 | 自定义标题栏，无地址栏 |

### H265 免转码

| 场景 | 之前 | 之后 |
|------|------|------|
| H265 + MP4 | 5-15 分钟全量转码 | 即时播放（零等待） |
| H265 + MKV | 5-15 分钟全量转码 | 3-10 秒 remux |
| 转码期间 | 需卸载推理模型腾显存 | 不需要，推理不受影响 |
| 720p 调度 | 转录完成后才能开始 720p 转码 | H265 场景不再需要 720p 转码 |

---

## 11. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| Electron 版本升级导致 API 变化 | 低 | main.js 仅用基础 API（BrowserWindow, loadURL），稳定性极高 |
| CORS 问题 | 低 | Electron 默认不受同源策略限制；后端已有 CORS 中间件 |
| 端口冲突 (8000) | 低 | 与现有方案相同风险，可通过端口检测解决 |
| 极少数用户无 HEVC 硬解 | 极低 | 现有 FFmpeg 转码流程自动兜底，无需额外处理 |
| Chromium 内存占用 | 中 | 额外 100-200MB，对于 GPU 推理场景可接受 |
