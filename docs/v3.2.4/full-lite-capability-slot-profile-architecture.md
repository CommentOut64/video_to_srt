# Full/Lite 能力系统架构

> Type: Architecture | Status: Active
> Version: V3.2.4+dev.20260304.11
> Last Updated: 2026-03-04

## 1. Summary

* **Goal**: 定义 Full/Lite 双版本的能力边界、依赖分离策略、前端能力快照机制，并修复已知缺陷。
* **设计原则**:
  - 能力由运行环境（flavor）决定，不由项目数据决定。项目快照不得提升运行时能力（base 为上限）。
  - 同一代码库，条件装配（Conditional Assembly）— 不做代码分叉。
  - Lite 版追求最小依赖（~200-250MB），Full 版包含完整 GPU 推理栈（~4.5GB）。
* **范围**: 前端能力快照归一化、IS_LITE 泄漏修复、Lite 依赖边界定义、Lite 功能范围定义、Proxy 简化方案（Electron）、后端模块耦合治理。

## 2. Diagram

### 2.1 前端能力流

```
flavor.js (编译时常量)
    |
    v
capabilitySelector.js (DEFAULT_CAPABILITY_SNAPSHOT)
    |                          ^
    v                          | 回退
mergeSnapshot(snapshot)  <-- normalizeCapabilitySnapshot(后端/缓存数据)
    |                          |
    |                     List[str] → null (拒绝)
    |                     Object{capabilities} → 通过(受 base 上限约束)
    v
selectCapabilities() / selectRouteVisibility() / selectFlavor()
    |
    v
TaskListView / EditorHeader / EditorView (v-if 消费)
```

后端能力边界独立于前端：`backend/app/main.py` 在启动时通过 `IS_LITE` 条件注册路由，是最终安全边界。

### 2.2 依赖分层

```
[共享层 — Lite 基线]                      [Full 增量层]
  fastapi / uvicorn / pydantic              torch==2.8.0 / torchaudio==2.8.0
  ffmpeg-python                             faster-whisper==1.2.1
  pypinyin / sudachipy / cmudict            demucs==4.0.1
  wave / struct (stdlib)                    pyannote.audio / speechbrain
  customtkinter                             numpy / av / wetext
                                            librosa / scipy
                                            punctuators / sentencepiece
                                            silero-vad / prometheus-client / tqdm
```

### 2.3 后端模块耦合图

```
[Lite 安全模块]                           [Full-only 模块（IS_LITE 门控）]
  ProjectService                            TranscriptionService
  SubtitleEditStore                         AsyncDualPipeline
  SubtitleOutputService                     FastWorker / SlowWorker
  HomophoneService (*)                      SenseVoiceONNXService
  FileService                              WhisperService / VADService
  UserConfigService                        DemucsService / BrouhahaService
  MediaPrepService (**)                    DNSMOSService / SpeechAnalysisService
  media_routes (波形生成)                   PunctuationService / TimelineService
                                           model_routes / model_runtime_routes

(*) HomophoneService 本身无 GPU 依赖，但路由被困在 transcription_routes.py 内（见缺陷 5）
(**) MediaPrepService 有 torch.cuda.is_available() 调用需解耦（见 8.2）
```

## 3. Key Components

* `frontend/src/config/flavor.js`: 编译/运行时 flavor 与 `CAPABILITIES`(8 字段) / `ROUTE_VISIBILITY`(4 字段) 静态常量入口。
* `frontend/src/state/capabilities/capabilitySelector.js`: 能力读取唯一入口（`DEFAULT_CAPABILITY_SNAPSHOT` + `mergeSnapshot` + `select*` 函数族）。
* `frontend/src/stores/projectStore.js`: `normalizeCapabilitySnapshot()` 定义处，所有写入 `meta.capabilitySnapshot` 的路径均经此归一化。
* `frontend/src/views/EditorView.vue`: `resolveCapabilitySnapshot()` 定义处，项目初始化时从后端/缓存读取快照并写入 store。
* `frontend/src/stores/editorSessionStore.js`: 会话域，`capabilitySnapshot` 计算属性委托到 `projectStore.meta`。
* `frontend/src/composables/useProxyVideo.js`: Proxy 视频管理，已改为经 `capabilitySelector` 读取能力，不再直读 `IS_LITE`。
* `frontend/src/views/TaskListView.vue` / `TaskListHeader.vue` / `TaskCardGrid.vue`: Task 端 `selectCapabilities()` 消费点。
* `frontend/src/components/editor/EditorHeader.vue`: Editor 端 `selectRouteVisibility()` 消费点（响应式 computed）。
* `backend/app/main.py`: Lite/Full 路由装配边界（9 处 `IS_LITE` 分支，全部集中于启动装配/初始化/接口拦截）。
* `backend/app/models/project_models.py:185`: `capability_snapshot: List[str]` — 后端项目元数据中的能力标签（当前仅 `["subtitle.import_export"]`）。
* `backend/app/api/routes/homophone_routes.py`: 同音检索路由（纯 `project_id` 语义，Full/Lite 共用）。
* `backend/app/services/homophone/`: 同音检索服务（service.py / tokenizers.py / db.py / runtime.py），纯 CPU，无 GPU 依赖。
* `pyproject.toml`: 依赖管理入口，已完成 Full/Lite 依赖分组（见 8.5）。

## 4. Lite 版功能范围

### 4.1 Lite 必须包含的功能

| 功能 | 后端模块 | 前端入口 | 依赖情况 |
|------|---------|---------|---------|
| 字幕编辑（完整） | SubtitleEditStore | EditorView | 纯 CPU |
| 同音/近音匹配 | HomophoneService | EditorView | pypinyin/sudachipy/cmudict（~70MB） |
| 项目创建（编辑项目） | ProjectService | TaskListView | 纯 CPU |
| 字幕导入 | FileService | ImportPage | 纯 CPU |
| 媒体导入 | MediaPrepService | ImportPage | FFmpeg（需解耦 torch） |
| 任务管理系统 | ProjectService | TaskListView | 纯 CPU |
| 自动更新 | 启动器（customtkinter） | — | customtkinter（~1MB） |
| 字幕导出（完整） | SubtitleOutputService | EditorView | 纯 CPU |
| 波形显示 | media_routes | EditorView | FFmpeg + stdlib（wave/struct） |
| 媒体预览 | media_routes | EditorView | FFmpeg |

### 4.2 Lite 排除的功能

| 功能 | 排除原因 | 节省空间 |
|------|---------|---------|
| 语音转录 | GPU 推理（torch/faster-whisper/ctranslate2） | ~3.5GB |
| 人声分离 | GPU 推理（demucs） | ~200MB |
| 频谱分诊 | GPU 推理（librosa/scipy） | ~200MB |
| 模型管理 | 无推理引擎则无需管理 | — |
| VAD / 声纹识别 | GPU 推理（pyannote/speechbrain） | ~300MB |
| 标点恢复 | 依赖 GPU 推理管线（punctuators/sentencepiece） | ~50MB |

### 4.3 关键发现：波形无重依赖

波形生成（`media_routes.py`）使用的技术栈：

- `_generate_peaks_with_ffmpeg()`: FFmpeg 子进程 + `struct.unpack`（stdlib）
- `_generate_peaks_with_wave()`: `wave.open()` + `struct.unpack`（均为 Python stdlib）
- `_ensure_waveform_audio()`: FFmpeg `-vn -acodec pcm_s16le -ar 16000 -ac 1`

**不依赖 librosa / scipy / numpy 做波形**。这使得 Lite 可以安全排除这三个库，依赖体积从 ~400MB 进一步降至 ~200-250MB。

### 4.4 Lite 高级设置 UI 收敛

Lite 模式下高级设置仅保留三个 Tab：

- `常规`
- `快捷键`
- `关于`

`Audio / ASR / LLM / System` 仅在 Full 模式显示。该规则在 `AdvancedSettings.vue` 内按 flavor 统一门控，避免不同入口（Task/Editor）出现不一致。

## 5. 依赖分离策略

### 5.1 方案：pyproject.toml 可选依赖组

```toml
# 核心依赖（Lite 基线，~200-250MB）
[project.dependencies]
  fastapi, uvicorn, pydantic,
  ffmpeg-python, pypinyin, sudachipy, cmudict,
  customtkinter, loguru, psutil, filelock, ...

# Full 增量依赖（~4GB）
[project.optional-dependencies]
full = [
  "numpy==1.26.4",
  "torch==2.8.0",
  "torchaudio==2.8.0",
  "av>=10.0.0",
  "faster-whisper==1.2.1",
  "ctranslate2>=4.5.0",
  "onnxruntime-gpu==1.23.2",
  "demucs==4.0.1",
  "librosa>=0.10.0",
  "scipy>=1.11.0,<2.0.0",
  "pyannote.audio==4.0.0",
  "speechbrain==1.0.3",
  "silero-vad>=5.0.0",
  "punctuators>=0.0.7",
  "sentencepiece>=0.2.0",
  "wetext>=0.1.2",
  "prometheus-client>=0.19.0",
  "tqdm",
  "pydub>=0.25.0",
  "soundfile>=0.12.0",
]
```

**安装命令**:
- Lite: `uv sync`（仅核心依赖）
- Full: `uv sync --extra full`（核心 + GPU 推理栈）

### 5.2 后端条件导入模式

已有模式（`main.py`）— `if not IS_LITE: from ... import ...`。全部 GPU 相关模块均在 IS_LITE 门控内延迟导入，Lite 模式下不触发 `import torch`。

### 5.3 依赖预算

| 层 | 包含内容 | 磁盘占用 |
|----|---------|---------|
| Lite 核心 | FastAPI 栈 + FFmpeg 绑定 + 同音检索 + 启动器 | ~200-250MB |
| Full 增量 | PyTorch + 推理引擎 + 音频分析 + VAD/声纹 | ~4GB |
| **Full 总计** | | **~4.5GB** |

## 6. 现状审计

### 6.1 已具备条件

1. 分域 store 已稳定（taskRuntimeStore/editorSessionStore/subtitleDocumentStore/playbackStore），版本分支逻辑不再由 store 承担。
2. `capabilitySnapshot + selector` 已打通主页面读取入口：TaskListView/TaskListHeader/TaskCardGrid/EditorHeader 均通过 `selectCapabilities()` 或 `selectRouteVisibility()` 读取。
3. Phase 0.5 已完成 95% 的 IS_LITE 收口。业务层 IS_LITE 直读泄漏仅剩 `useProxyVideo.js` 一个文件。

### 6.2 已知缺陷

#### 缺陷 1（已修复）：normalizeCapabilitySnapshot 对数组格式放行（数据污染）

当前行为：`normalizeCapabilitySnapshot()` 已显式拒绝 `List[str]` 与非标准对象格式，统一回退默认能力快照，不再生成数组索引污染键。

#### 缺陷 2（已修复）：useProxyVideo.js IS_LITE 直读

当前行为：`useProxyVideo.js` 已移除 `IS_LITE` 直读，改为通过 `selectCapabilities().canTranscribe` 读取能力语义。

#### 缺陷 3：快照结构歧义

`DEFAULT_CAPABILITY_SNAPSHOT` 中 `canShowEditorProgress`、`canUseTaskMonitor`、`taskSwitcherMode` 三个字段放在顶层而非 `capabilities` 子对象内。导致 `selectCapabilities()` 返回的 8 字段对象不包含这三个字段。消费侧如果读取这三个字段需要走 `selectCapabilitySnapshot()` 而非 `selectCapabilities()`。

**当前处理**: 标记为技术债，不在本期修复（移动会影响所有消费点）。

#### 缺陷 4：前后端 flavor 可能错配

前端 flavor 是构建时注入（`vite.config.js:10` 的 `__APP_FLAVOR__`），后端 flavor 是运行时环境解析（`config.py:474` 的 `ANCHORFLUX_FLAVOR`）。生产部署为本地捆绑，错配概率极低；但开发态 `npm run dev` + `uvicorn` 分别启动时可能不一致。

**当前处理**: 后端启动时已打印 `AnchorFlux flavor: lite (IS_LITE=True)` 日志（`config.py:490`），开发者可通过日志排查。不引入 HTTP 协商端点。

#### 缺陷 5（已修复）：同音路由与转录路由强耦合

**修复前问题**: 同音端点位于 `transcription_routes.py` 内并受 `IS_LITE` 门控，Lite 模式不可访问。

**当前状态**:
- 同音路由已独立为 `backend/app/api/routes/homophone_routes.py`，并在 `main.py` 无条件注册。
- 接口已收敛为纯 project 语义（不再提供 `legacy/tasks/*/homophone/*` 旧格式）。
- 当前有效端点为：
  - `POST /api/projects/{project_id}/homophone/find`
  - `GET /api/projects/{project_id}/homophone/index-status`
  - `POST /api/projects/{project_id}/homophone/batch-replace`
  - `GET /api/settings/homophone/global-terms`
  - `PUT /api/settings/homophone/global-terms`

**HomophoneService 依赖边界**:
- 依赖：pypinyin / sudachipy / sudachidict-core / cmudict / phonemizer（全部 CPU-only，~70MB）
- 存储：SQLite
- 无 torch / 无 numpy 依赖

#### 缺陷 6：taskCreate 路由可见性语义混淆

`ROUTE_VISIBILITY.taskCreate: !IS_LITE` 将"创建转录任务"和"创建编辑项目"混为一谈。Lite 需要"创建编辑项目"（导入字幕开始编辑）但不需要"创建转录任务"（提交音视频做 ASR）。

## 7. 能力字段基线（当前活跃代码）

以 `flavor.js:16-27` 的 `CAPABILITIES` 为唯一基线，禁止重定义不兼容的类型：

```js
// flavor.js -- 8 个能力字段（当前基线，只增不改）
CAPABILITIES = {
  canTranscribe:     !IS_LITE,  // 转录
  canSeparateVocal:  !IS_LITE,  // 人声分离
  canSpectralTriage: !IS_LITE,  // 频谱分诊
  canManageModels:   !IS_LITE,  // 模型管理
  canImportSubtitle: true,      // 字幕导入
  canExportSubtitle: true,      // 字幕导出
  canEditSubtitle:   true,      // 字幕编辑
  canPreviewMedia:   true,      // 媒体预览
}

// flavor.js -- 4 个路由可见性字段
ROUTE_VISIBILITY = {
  taskList:   true,       // 任务列表（Lite 也使用）
  taskCreate: !IS_LITE,   // 任务创建（见缺陷 6：语义待拆分）
  importPage: true,       // 导入页
  editor:     true,       // 编辑器
}
```

**扩展规则**: 新增能力字段必须加入 `CAPABILITIES` 对象内（不放顶层），并同时在 `DEFAULT_CAPABILITY_SNAPSHOT.capabilities` 中补默认值。

**快照顶层额外字段**（capabilitySelector.js，技术债留位）:
- `canShowEditorProgress: true` — 应归入 capabilities，暂留顶层
- `canUseTaskMonitor: true` — 同上
- `taskSwitcherMode: 'legacy'` — 同上，注意实际值是 `'legacy'` 而非文档曾定义的 `'monitor'|'quick_switcher'|'none'`

**待拆分字段**（缺陷 6 修复时执行）:

```js
// taskCreate 拆分为：
ROUTE_VISIBILITY = {
  // ...
  transcribeCreate: !IS_LITE,  // 转录任务创建（Full-only）
  projectCreate:    true,       // 编辑项目创建（Full/Lite 共用）
}
```

## 8. 修复与实施方案

### 8.1 [P0] 重写 normalizeCapabilitySnapshot

**位置**: `frontend/src/stores/projectStore.js:223-227`

**当前代码**（空壳）:
```js
function normalizeCapabilitySnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object') { return null }
  return snapshot  // 数组也放行
}
```

**修复后**:
```js
// 治理规则：项目/会话快照不得提升运行环境能力（base 为上限）。
// 当前：后端 List[str] 被拒绝，回退运行时默认值。
// 将来接入结构化后端快照时，须以 flavor.js CAPABILITIES 为上限裁剪。
function normalizeCapabilitySnapshot(snapshot) {
  if (!snapshot) return null
  if (Array.isArray(snapshot)) {
    // 后端 List[str] 格式（如 ["subtitle.import_export"]），降级为诊断元数据，不参与能力合并
    console.debug('[normalizeCapabilitySnapshot] 忽略后端 tag 列表格式，使用运行时默认能力')
    return null
  }
  if (typeof snapshot === 'object' && snapshot.capabilities) {
    return snapshot
  }
  console.warn('[normalizeCapabilitySnapshot] 非标准快照格式，已忽略:', typeof snapshot)
  return null
}
```

**行为变化**: 无。当前数组格式也未真正参与能力判断（展开后被默认值覆盖），改为显式拒绝后结果等价。

### 8.2 [P0] 修复 useProxyVideo.js IS_LITE 泄漏

**位置**: `frontend/src/composables/useProxyVideo.js:13,73`

**当前代码**:
```js
import { IS_LITE } from '@/config/flavor'
// ...
const isReady = computed(() => {
  if (IS_LITE && !identityId.value) { return true }
  // ...
})
```

**修复**: 移除 `IS_LITE` 导入，改为从 selector 读取:
```js
import { selectCapabilities } from '@/state/capabilities/capabilitySelector'
// ...
const isReady = computed(() => {
  if (!selectCapabilities().canTranscribe && !identityId.value) { return true }
  // ...
})
```

**语义说明**: 这里的逻辑是"无转录能力（Lite）模式下，没有视频身份时跳过就绪等待"。用 `!canTranscribe` 比用 `IS_LITE` 语义更精确 — Lite 的核心特征就是无转录。

### 8.3 [P1] mergeSnapshot 白名单合并

**位置**: `frontend/src/state/capabilities/capabilitySelector.js:16-29`

**修复**: 限制顶层展开的键，防止未知属性混入:

```js
const ALLOWED_TOP_KEYS = new Set([
  'profile', 'flavor', 'isLite', 'version',
  'capabilities', 'routeVisibility',
  // 技术债：以下三个字段应归入 capabilities 子对象，暂留顶层保持兼容
  'canShowEditorProgress', 'canUseTaskMonitor', 'taskSwitcherMode',
])

// 治理规则：base 为上限。capabilities 中 base 为 false 的字段，不可被外部快照提升为 true。
// 当前实现：外部快照经 normalizeCapabilitySnapshot 拒绝 List[str] 后回退默认值，上限自动满足。
// 将来如果接入结构化后端快照，须在此处增加上限裁剪：
//   base false -> 结果必须 false（单向降级，不可提权）
function mergeSnapshot(snapshot = {}) {
  const filtered = {}
  for (const key of Object.keys(snapshot)) {
    if (ALLOWED_TOP_KEYS.has(key)) {
      filtered[key] = snapshot[key]
    }
  }
  return {
    ...DEFAULT_CAPABILITY_SNAPSHOT,
    ...filtered,
    capabilities: {
      ...DEFAULT_CAPABILITY_SNAPSHOT.capabilities,
      ...(filtered.capabilities || {}),
    },
    routeVisibility: {
      ...DEFAULT_CAPABILITY_SNAPSHOT.routeVisibility,
      ...(filtered.routeVisibility || {}),
    },
  }
}
```

**注**: P0 的 normalizeCapabilitySnapshot 已拦截数组，白名单的主要作用是防止未来新增字段时意外展开未知属性（防御性编程）。

### 8.4 [P1] 提取同音路由

**目标**: 同音检索与转录路由解耦，并统一为纯 `project_id` 语义。

**修复**: 将同音路由提取到独立文件 `backend/app/api/routes/homophone_routes.py`，在 `main.py` 中无条件注册（不受 IS_LITE 门控），并移除 `transcription_routes.py` 中同音旧入口。

**提取范围**:
- 数据模型：`HomophoneFindRequest`、`BatchReplaceRequest`、`GlobalTermSyncRequest`
- 路由端点：`project_homophone_find`、`project_homophone_index_status`、`project_homophone_batch_replace`、`get_homophone_global_terms`、`put_homophone_global_terms`
- 辅助函数：`_ensure_homophone_index_ready`、`_collect_segments_for_project`

**注册变更**（`main.py`）:
```python
# 同音路由 — Full/Lite 共用，不受 IS_LITE 门控
from app.api.routes.homophone_routes import create_homophone_router
app.include_router(create_homophone_router(...))
```

**依赖安全性**: HomophoneService 仅依赖 pypinyin / sudachipy / cmudict（CPU-only），不引入任何 GPU 依赖。

### 8.5 [P1] pyproject.toml 依赖分组（已完成）

**当前状态**: GPU/推理依赖已从 `[project.dependencies]` 迁移到 `[project.optional-dependencies].full`，默认 `uv sync` 仅安装 Lite 基线依赖。

**修复结果**:
- Lite: `uv sync` / `uv sync --extra test` 不再安装 torch/onnxruntime-gpu/demucs 等 Full 依赖
- Full: `uv sync --extra full` 可恢复完整 GPU 推理栈

**需移入 `full` 组的依赖**:
```
torch==2.8.0, torchaudio==2.8.0,
faster-whisper==1.2.1, ctranslate2>=4.5.0,
onnxruntime-gpu==1.23.2, demucs==4.0.1,
librosa>=0.10.0, scipy>=1.11.0, soundfile>=0.12.0, pydub>=0.25.0,
pyannote.audio==4.0.0, speechbrain==1.0.3, silero-vad>=5.0.0,
punctuators>=0.0.7, sentencepiece>=0.2.0, tokenizers>=0.20.0,
huggingface-hub>=0.24.0, intel-openmp, sympy>=1.12,
```

**保留在核心依赖中**:
```
numpy==1.26.4, fastapi, uvicorn, pydantic,
ffmpeg-python, av, pypinyin, sudachipy, sudachidict-core, cmudict, phonemizer,
customtkinter, loguru, psutil, filelock, click, colorama, tqdm,
python-dateutil, python-multipart, sse-starlette, setuptools<78,
wetext>=0.1.2, prometheus-client,
```

### 8.6 [P1] MediaPrepService torch 解耦（已完成）

**问题**: `MediaPrepService` 中 720p 转码路径直接 `import torch`，Lite 环境无 torch 时会中断任务流程。

**修复结果**:
- 新增 `_is_cuda_available_for_proxy()`：`ImportError` 时返回 `False` 并回退 CPU 转码
- 新增 `_clear_cuda_cache_if_available()`：Lite 环境无 torch 时无操作
- `_execute_proxy_task()` 改为调用安全检测函数，不再直接导入 torch
- `_unload_all_models_before_720p()` 改为条件清理 CUDA 缓存，避免 Lite 下误报异常

### 8.7 [P2] CI Full/Lite 双矩阵构建门禁

**当前可做范围**（无前端 E2E 测试框架）:

**层 1 — 构建验证**:
1. `VITE_APP_FLAVOR=full npm run build` 成功
2. `VITE_APP_FLAVOR=lite npm run build` 成功

**层 2 — 后端路由验证**:
3. `ANCHORFLUX_FLAVOR=lite` 启动后断言转录路由不存在
4. `ANCHORFLUX_FLAVOR=full` 启动后断言转录路由存在
5. `ANCHORFLUX_FLAVOR=lite` 启动后断言同音路由存在（提取后）

**层 3 — 前端 UI 入口验证**:
暂缓，需引入 Playwright/Cypress 后作为增量补充。

**新增配置**:
- `frontend/package.json` 新增 `"build:lite": "VITE_APP_FLAVOR=lite vite build"`
- CI 配置新增 Lite 构建 job

## 9. Proxy 视频简化方案（Electron 模式）

### 9.1 背景

当前 Proxy 策略基于浏览器播放限制设计：H265 无法播放需完整转码，MKV 容器不支持需 remux。引入 Electron 后，Chromium 可配置原生 H265 解码支持，大幅简化 Proxy 需求。

### 9.2 当前 vs Electron 策略对比

| 场景 | 当前策略 | Electron 策略 |
|------|---------|-------------|
| H265/MP4 | TRANSCODE_FULL（360p→720p 渐进） | DIRECT_PLAY |
| H264/MP4 | DIRECT_PLAY | DIRECT_PLAY |
| H264/MKV | REMUX → MP4 | REMUX → MP4 |
| H265/MKV | TRANSCODE_FULL | REMUX → MP4 |
| 其他编码 | TRANSCODE_FULL | 简单 CPU 转码 fallback |

### 9.3 可移除组件（Electron 落地后）

- 360p/720p 渐进代理管线
- `proxy_720_scheduler`（GPU 转码调度）
- GPU 转码相关代码路径
- 保留仅：FFmpeg remux + 简单格式检测 + 简单 CPU 转码 fallback

### 9.4 时序约束

**Proxy 简化必须等 Electron 落地后执行**，原因：
1. 需要实测 Electron 的实际编解码支持矩阵
2. 不同平台（Windows/macOS/Linux）的硬件解码能力不同
3. 当前浏览器模式仍需完整 Proxy 管线

## 10. 实施路线图

### Phase 1：立即可做（不依赖 Electron）

| 编号 | 任务 | 优先级 | 工作量 |
|------|------|--------|--------|
| 8.1 | 重写 normalizeCapabilitySnapshot | P0 | 小 |
| 8.2 | 修复 useProxyVideo.js IS_LITE 泄漏 | P0 | 小 |
| 8.3 | mergeSnapshot 白名单合并 | P1 | 小 |
| 8.4 | 提取同音路由到独立文件 | P1 | 中 |
| 8.5 | pyproject.toml 依赖分组（已完成） | P1 | 中 |
| 8.6 | MediaPrepService torch 解耦（已完成） | P1 | 小 |

### Phase 2：Electron 落地后

| 编号 | 任务 | 前置条件 |
|------|------|---------|
| 9.3 | Proxy 管线简化 | Electron 编解码实测完成 |
| — | taskCreate 路由拆分（缺陷 6） | 前端路由重构 |

### Phase 3：打包与 CI

| 编号 | 任务 | 前置条件 |
|------|------|---------|
| 8.7 | CI 双矩阵构建门禁 | Phase 1 完成 |
| — | Lite 独立打包脚本 | pyproject.toml 分组完成 |
| — | Lite 安装包体积验证 | 打包脚本完成 |

## 11. 治理规则

1. **IS_LITE 禁区**: 组件/composable 禁止新增 `IS_LITE` 直读。允许读取 `IS_LITE` 的文件仅限 `flavor.js`（源头定义）和 `capabilitySelector.js`（构建默认快照）。
2. **能力字段扩展**: 新增能力字段必须放入 `CAPABILITIES` 对象内，同时补 `DEFAULT_CAPABILITY_SNAPSHOT.capabilities` 默认值、至少一个消费点、一条验证断言。
3. **快照归一化**: 所有写入 `projectStore.meta.capabilitySnapshot` 的路径必须经 `normalizeCapabilitySnapshot()` 过滤。
4. **base 为上限**: 项目/会话快照不得提升运行环境能力。`CAPABILITIES` 中值为 `false` 的字段在合并后仍须为 `false`。
5. **UI 差异化**: 当前使用 `v-if="capabilities.canXxx"` 实现，差异点约 6 个，不引入 SlotBinding 体系（差异点不足 50 个，slot registry 收益不抵认知成本）。
6. **依赖隔离**: GPU/推理依赖必须在 `[project.optional-dependencies] full` 组内。核心依赖（`[project.dependencies]`）禁止引入 torch / CUDA 相关包。
7. **条件导入**: 后端 Full-only 模块必须在 `if not IS_LITE:` 块内延迟导入。顶层 `import torch` 等语句禁止出现在 Lite 可达路径中。

## 12. 暂缓事项与理由

### SlotBinding 体系

**暂缓理由**: 当前 Full/Lite UI 差异点仅约 6 个，`v-if` 完全胜任。引入 slot registry 后调试链从 1 步（看模板条件）变为 4 步（模板 → slotId → 注册表 → profile → 能力字段），认知成本不成比例。当差异点超过 50 个或需要支持第三方插件注入时再考虑。

### ServerNegotiation / HTTP 协商

**暂缓理由**: 本项目是本地桌面应用（PyInstaller + uvicorn + 浏览器/Electron），前后端捆绑部署，生产环境 flavor 错配概率极低。开发态错配可通过后端启动日志 `AnchorFlux flavor: ...` 定位。引入 HTTP 协商轮询增加启动延迟、后端代码量和失败处理复杂度，解决的是一个当前不存在的问题。

### CapabilityRuntime 三层合并

**暂缓理由**: 原方案设计了 `baseProfile > sessionSnapshot > negotiatedSnapshot` 三层动态合并。但对桌面应用，flavor 在进程启动时确定，整个生命周期不变，只有 base 一层有意义。`session > base` 的优先级存在安全隐患（Lite 打开 Full 项目时，项目快照的 `canTranscribe: true` 会错误提权）。当前 selector 的单层 `DEFAULT_CAPABILITY_SNAPSHOT` + `mergeSnapshot` 回退机制已满足需求。

## 13. 验收口径

### Phase 1 验收

1. **数据污染修复**: `normalizeCapabilitySnapshot` 拒绝 `List[str]` 格式，快照对象不出现数组索引垃圾键。
2. **IS_LITE 收口**: 全局搜索 `import.*IS_LITE` 仅出现在 `flavor.js` 和 `capabilitySelector.js` 两个允许位置。
3. **行为等价**: Full/Lite 模式下所有页面行为与修复前完全一致（无功能变化）。
4. **同音可用**: Lite 模式下同音检索 API 正常响应。
5. **Lite 安装**: `uv sync`（不含 `--extra full`）成功，且不拉取 torch / CUDA 相关包。
6. **Full 安装**: `uv sync --extra full` 成功，功能与当前完全一致。

### Phase 2 验收

7. **Proxy 简化**: Electron 模式下 H265 直接播放，无转码延迟。
8. **构建通过**: `VITE_APP_FLAVOR=full` 和 `VITE_APP_FLAVOR=lite` 两种模式均构建成功。

### Phase 3 验收

9. **CI 门禁**: 双矩阵构建均绿灯。
10. **后端路由隔离**: Lite 模式启动后转录路由不可达、同音路由可达；Full 模式启动后全部路由正常注册。
11. **Lite 体积**: 打包后安装包 < 300MB（不含 FFmpeg）。
