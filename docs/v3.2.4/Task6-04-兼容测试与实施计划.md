# Task 6-04: 兼容、测试与实施计划

> Type: Architecture | Status: Active
> Version: V3.2.4+dev.20260222.01
> 上游文档: `Task6-00-总纲`, `Task6-01 ~ Task6-03`

## 1. 旧任务兼容方案

### 1.1 兼容目标

1. 历史任务零重跑转录
2. 完整支持编辑、预览、导出
3. 保留旧入口 `/editor/:jobId` 和旧链接可用
4. 旧任务迁移后与新项目共用 `SubtitleDoc` 数据形态

### 1.2 懒迁移触发点

| 触发场景 | 触发方式 | 用户感知 |
|----------|---------|---------|
| 打开旧编辑路由 `/editor/:jobId` | EditorView resolve → `legacyApi.resolveTask(jobId)` | 路由重定向到 `/editor/project/:projectId` |
| 调用旧字幕编辑 API `/jobs/{job_id}/subtitles` | `transcription_routes.py` 桥接检查 | 透明，API 响应格式不变 |
| Lite 打开旧项目 | 同上 | 允许编辑/预览/导出，禁止转录 |

### 1.3 迁移流程

```
1. resolve(job_id) 被调用
2. 检查 _legacy_map.json 中是否已有映射
   ├── 有 → 返回已映射的 project_id
   └── 无 → 继续
3. 检查 {JOBS_DIR}/{job_id}/ 目录是否存在
   ├── 不存在 → 404
   └── 存在 → 继续
4. project_id = generate_legacy_project_id(job_id)  // 确定性派生
5. 恢复字幕（按优先级）
   ├── subtitle_edits.json + checkpoint → 最完整
   ├── checkpoint/transcription_text → 次优
   └── *.srt → 兜底
6. 为每个字幕段生成 segment_id
7. 扫描媒体资产
8. 创建 project_meta.json（写入原 job_dir，与 job_meta.json 并存）
9. 记录映射到 _legacy_map.json
10. 返回 (project_id, is_newly_migrated=true)
```

### 1.4 字幕恢复优先级详解

**策略 1: subtitle_edits.json + checkpoint（最完整）**

```python
def _restore_from_edits_and_checkpoint(job_dir):
    # 1. 加载 checkpoint 中的原始转录结果
    checkpoint = load_checkpoint(job_dir)
    base_segments = checkpoint.get("transcription_text", [])

    # 2. 加载编辑记录
    edits = load_edits(job_dir)
    deleted = load_deleted_indices(job_dir)

    # 3. 合并
    result = []
    for idx, seg in enumerate(base_segments):
        if idx in deleted:
            continue
        if str(idx) in edits:
            edit = edits[str(idx)]
            seg["text"] = edit.get("text", seg["text"])
            seg["start"] = edit.get("start", seg["start"])
            seg["end"] = edit.get("end", seg["end"])
            seg["is_modified"] = True
            seg["original_text"] = seg.get("original_text", seg["text"])
        result.append(seg)

    # 4. 添加手动新增的条目（负索引）
    for idx_str, edit in edits.items():
        if int(idx_str) < 0:
            result.append({
                "text": edit["text"],
                "start": edit["start"],
                "end": edit["end"],
                "is_modified": True,
            })

    return sorted(result, key=lambda s: s["start"])
```

**策略 2: 仅 checkpoint**

```python
def _restore_from_checkpoint(job_dir):
    checkpoint = load_checkpoint(job_dir)
    segments = checkpoint.get("transcription_text", [])
    return segments if segments else None
```

**策略 3: SRT 兜底**

```python
def _restore_from_srt(job_dir):
    for srt_path in job_dir.glob("*.srt"):
        segments = SubtitleDocService.parse_srt(srt_path.read_text(encoding="utf-8"))
        if segments:
            return segments
    return None
```

### 1.5 媒体兼容策略

1. **直接复用旧目录**: legacy 项目的 `project_dir` 就是原 `job_dir`，所有媒体文件路径不变
2. **缺 audio.wav 或 peaks**: 仅触发 `media.prepare_preview` 运行补齐（不触发转录）
3. **缺视频**: 编辑器进入"纯字幕模式"，保留字幕时间轴和导出能力
4. **Lite 模式下不生成 proxy**: 避免触发重媒体链路

### 1.6 旧 API 桥接行为

| 旧端点 | 桥接行为 |
|--------|---------|
| `GET /transcription-text/{job_id}` | resolve → 从 SubtitleDocService 加载 |
| `PUT /jobs/{job_id}/subtitles` | resolve → SubtitleDocService.update_segment() |
| `POST /jobs/{job_id}/subtitles` | resolve → SubtitleDocService.create_segment() |
| `DELETE /jobs/{job_id}/subtitles/{index}` | resolve → SubtitleDocService.delete_segment() |
| `GET /download/{job_id}` | resolve → SubtitleDocService.export_srt() |
| `GET /{job_id}/video` | _resolve_media_dir(job_id) → 文件返回 |
| `GET /{job_id}/audio` | 同上 |
| `GET /{job_id}/peaks` | 同上 |

### 1.7 错误码收敛

| 错误码 | 含义 | 触发条件 |
|--------|------|---------|
| `404 legacy_mapping_not_found` | 旧任务不存在 | job_dir 不存在 |
| `409 legacy_transcription_conflict` | 对 legacy 任务重新转录 | Full 模式下尝试对 legacy 项目启动转录 |
| `422 capability_not_allowed` | Lite 调用禁用能力 | Lite 模式下调用转录/分离相关 API |

### 1.8 回滚策略

- **回滚方式**: 删除 `project_meta.json` + `_legacy_map.json` 中的映射条目
- **非破坏保障**: 旧 `job_meta.json`、`subtitle_edits.json`、所有媒体文件完全不变
- **回滚开关**: 配置项 `LEGACY_ADAPTER_ENABLED`（默认 true），设为 false 时跳过所有桥接逻辑

## 2. 测试方案

### 2.1 单元测试

#### test_project_service.py（~200行）

| 测试用例 | 验证内容 |
|----------|---------|
| `test_create_import_project` | 导入创建 → project_meta.json 写入正确 |
| `test_create_normal_project` | 从转录任务创建 → 复用 job_dir |
| `test_get_project` | 查询 → 内存缓存和文件回退 |
| `test_list_projects` | 列表 → 扫描 JOBS_DIR |
| `test_scan_media_assets` | 媒体扫描 → 正确识别各类文件 |
| `test_project_id_generation` | ID 生成 → 格式正确、唯一性 |

#### test_subtitle_doc_service.py（~200行）

| 测试用例 | 验证内容 |
|----------|---------|
| `test_import_segments` | 导入 → subtitle_edits.json 格式正确 |
| `test_load_segments` | 加载 → 合并编辑和删除 |
| `test_update_segment` | 更新 → segment_id 到 sentence_index 映射正确 |
| `test_create_segment` | 新增 → 手动索引递减、segment_id 分配 |
| `test_delete_segment` | 删除 → deleted_indices 更新 |
| `test_segment_map_persistence` | _segment_map 持久化 → 重启后映射恢复 |
| `test_parse_srt` | SRT 解析 → 时间戳和文本正确 |
| `test_parse_ass` | ASS 解析 → 样式标签清理 |
| `test_export_srt` | SRT 导出 → 格式符合规范 |

#### test_legacy_projection.py（~200行）

| 测试用例 | 验证内容 |
|----------|---------|
| `test_resolve_new_migration` | 首次 resolve → 创建 project + 映射 |
| `test_resolve_idempotent` | 重复 resolve → 返回同一 project_id |
| `test_resolve_nonexistent` | 不存在的 job_id → 404 |
| `test_restore_from_edits_checkpoint` | 策略 1 恢复 → 合并正确 |
| `test_restore_from_checkpoint_only` | 策略 2 恢复 → 内容正确 |
| `test_restore_from_srt` | 策略 3 恢复 → 解析正确 |
| `test_legacy_project_id_deterministic` | 确定性派生 → 同 job_id 始终同 project_id |
| `test_media_assets_scanned` | 迁移后媒体资产正确挂载 |

#### test_flavor_guard.py（~150行）

| 测试用例 | 验证内容 |
|----------|---------|
| `test_flavor_full_default` | 默认 flavor=full |
| `test_flavor_lite_from_env` | ANCHORFLUX_FLAVOR=lite → IS_LITE=true |
| `test_flavor_lite_from_legacy_env` | ANCHORFLUX_LITE=true → IS_LITE=true |
| `test_lite_routes_not_registered` | Lite 模式 → 转录路由不注册 |
| `test_lite_services_not_initialized` | Lite 模式 → 重依赖服务不初始化 |
| `test_lite_capability_guard` | Lite 调用转录 API → 422 |

### 2.2 集成测试

| 测试场景 | 验证链路 |
|----------|---------|
| Lite 字幕导入闭环 | POST /import → GET /subtitles → PUT /subtitles → GET /export |
| 旧任务首次打开 | /editor/:jobId → resolve → 自动迁移 → 可编辑 → 可导出 |
| 旧任务二次打开 | 已迁移 → 直接返回 project_id → 编辑正常 |
| 无视频纯字幕编辑 | 导入 SRT（无视频） → 编辑 → 导出 |
| 媒体缺失自动补齐 | 有视频缺 peaks → 自动生成 → 不触发转录 |
| Full → Lite 降级 | 同项目数据 → Lite 可编辑不可转录 |

### 2.3 CI 守卫

```yaml
# .github/workflows/integration_ci.yml 新增步骤

- name: Lite 模式启动检查
  run: |
    ANCHORFLUX_LITE=true python -c "
    from backend.app.core.config import IS_LITE, FLAVOR
    assert IS_LITE == True
    assert FLAVOR == 'lite'
    "

- name: Lite 依赖检查
  run: |
    ANCHORFLUX_LITE=true python -c "
    # 验证 Lite 模式不 import 重依赖
    import sys
    sys.modules['torch'] = None  # 阻止 import
    sys.modules['onnxruntime'] = None
    sys.modules['faster_whisper'] = None
    from backend.app.main import create_lite_app
    "

- name: Project API 冒烟测试
  run: |
    pytest backend/tests/test_project_service.py -v
    pytest backend/tests/test_subtitle_doc_service.py -v
    pytest backend/tests/test_legacy_projection.py -v
```

### 2.4 用户验收标准

| 场景 | 验收标准 |
|------|---------|
| 历史任务 | 打开编辑器 → 字幕完整显示 → 编辑不丢失 → 导出正确 |
| 无媒体编辑 | 导入 SRT → 字幕列表可编辑 → 时间戳可调 → 导出正确 |
| 有媒体编辑 | 导入 SRT + 视频 → 波形显示 → 播放同步 → 导出正确 |
| 导出一致性 | 编辑器当前状态 === 导出文件内容 |
| Lite 闭环 | 导入 → 编辑 → 导出，全程不报错 |

## 3. 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| 映射漂移（project_id/job_id 不一致） | "能打开但不能保存" | 中 | resolve 幂等 + 启动时校验 _legacy_map 完整性 |
| 双 API 并存期语义不一致 | 旧端点与新端点返回字段差异 | 中 | 桥接层统一返回格式 + 字段别名适配 |
| SSE 双通道导致事件重复消费 | 字幕更新闪烁 | 低 | 事件携带 seq/updated_at，前端按序去重 |
| Lite 分支逻辑漂移导致构建包含重依赖 | Lite 包体积爆炸 | 中 | CI import 守卫 + 构建产物体积检查 |
| subtitle_edits.json 中 _segment_map 字段被旧代码意外清除 | segment_id 映射丢失 | 低 | SubtitleDocService 加载时自动重建映射 |
| 前端 Task 9 重构后 store API 变化 | Task 6-Lite 前端需要适配 | 高 | Task 6-Lite 前端严格在 Task 9 之后开始 |

## 4. 分阶段实施计划

### 4.1 Phase 1: 后端基础设施（可独立交付）

**目标**: 后端 API 可用，Lite 模式可启动

| 步骤 | 内容 | 涉及文件 | 前置条件 |
|------|------|---------|---------|
| 1.1 | flavor 配置 | `config.py` | 无 |
| 1.2 | 领域模型 | `project_models.py`(新) | 无 |
| 1.3 | 字幕文档服务 | `subtitle_doc_service.py`(新) | 1.2 |
| 1.4 | 项目服务 | `project_service.py`(新) | 1.2, 1.3 |
| 1.5 | 项目 API | `project_routes.py`(新) | 1.3, 1.4 |
| 1.6 | main.py 条件注册 | `main.py` | 1.1, 1.5 |
| 1.7 | 单测 | `test_*`(新) | 1.3-1.5 |

**验收**: `ANCHORFLUX_LITE=true` 启动 → `POST /api/projects/import` 返回 200 → `GET /api/projects/{id}/subtitles` 返回字幕

### 4.2 Phase 2: 旧任务兼容（可独立交付）

**目标**: 旧任务无缝迁移

| 步骤 | 内容 | 涉及文件 | 前置条件 |
|------|------|---------|---------|
| 2.1 | Legacy 投影服务 | `legacy_projection_service.py`(新) | Phase 1 |
| 2.2 | Legacy API | `legacy_compat_routes.py`(新) | 2.1 |
| 2.3 | 转录路由桥接 | `transcription_routes.py` | 2.1 |
| 2.4 | SSE 频道扩展 | `streaming_subtitle.py` | 2.1 |
| 2.5 | 媒体路由兼容 | `media_routes.py` | 2.1 |
| 2.6 | job_models 扩展 | `job_models.py` | 无 |
| 2.7 | 单测 | `test_legacy_projection.py`(新) | 2.1-2.5 |

**验收**: 旧任务 job_id → resolve → project_id → 字幕完整 → 可编辑 → 可导出

### 4.3 Phase 3: 前端适配（依赖 Task 9）

**目标**: 前端双入口 + Lite UI

| 步骤 | 内容 | 涉及文件 | 前置条件 |
|------|------|---------|---------|
| 3.1 | Vite 配置 | `vite.config.js` | 无 |
| 3.2 | flavor 模块 | `flavor.js`(新) | 3.1 |
| 3.3 | API 客户端 | `projectApi.js`(新), `legacyApi.js`(新) | Phase 1 API 就绪 |
| 3.4 | 路由改造 | `router/index.js` | 3.2 |
| 3.5 | 导入页面 | `ImportView.vue`(新) | 3.3, 3.4 |
| 3.6 | 编辑器适配 | `EditorView.vue` | 3.2, 3.3, Task 9 完成 |
| 3.7 | Store 适配 | `projectStore.js` | 3.2, Task 9 完成 |
| 3.8 | SSE 适配 | `sseChannelManager.js` | Phase 2 SSE 就绪 |
| 3.9 | 子组件适配 | Composables + 组件 | 3.6, 3.7 |

**验收**: `/import` 页面可用 → 导入 SRT → 跳转编辑器 → 编辑 → 导出

### 4.4 Phase 4: 集成验证与收敛

| 步骤 | 内容 |
|------|------|
| 4.1 | 端到端测试（Lite 导入闭环 + 旧任务迁移闭环） |
| 4.2 | CI 守卫添加（Lite 启动检查 + 依赖检查） |
| 4.3 | 性能验证（resolve 延迟 < 100ms，导入 1000 条字幕 < 2s） |
| 4.4 | 文档更新（llmdoc 同步） |

## 5. 与 Sprint 排布的对应

参考 `development-roadmap.md` 的 Sprint 排布：

```
Sprint 2:
  ├── Task 6-Lite 后端 = Phase 1 + Phase 2（本方案）
  ├── Task 9 前端状态重构（并行）
  └── Task 11 Electron（并行）

Sprint 2 → Sprint 3 过渡期:
  └── Task 6-Lite 前端 = Phase 3（在 Task 9 合入后）

Sprint 3:
  ├── Task 12 Lite 构建 profile
  ├── Phase 4 集成验证
  └── ★ Lite v1 交付
```

## 6. 分支策略

```
feature/edit-decouple（从 main 创建）
  ├── Phase 1 提交: feat(project): 项目领域模型与服务
  ├── Phase 2 提交: feat(legacy): 旧任务投影与兼容桥接
  ├── Phase 3 提交: feat(frontend): 编辑器双入口与 Lite 适配
  └── Phase 4 提交: test(integration): 端到端验证与 CI 守卫
```

**合入顺序**:
1. `refactor/frontend-stores`（Task 9）先合入 main
2. `feature/edit-decouple` 在 Task 9 之后合入 main
3. `feature/electron`（Task 11）随时可合入（与本分支无文件冲突）

## 7. 完成定义（DoD）

### Task 6-Lite 完成标准

- [ ] `ANCHORFLUX_LITE=true` 后端启动不加载转录相关服务
- [ ] `POST /api/projects/import` 可导入 SRT/ASS/VTT 并返回 project_id
- [ ] `GET /api/projects/{id}/subtitles` 返回完整字幕列表（含 segment_id）
- [ ] `PUT /api/projects/{id}/subtitles/{segment_id}` 更新字幕正常
- [ ] `GET /api/projects/{id}/export?format=srt` 导出正确
- [ ] 旧任务 job_id → resolve → project_id 映射幂等
- [ ] 旧任务字幕恢复三优先级策略正确
- [ ] 前端 `/import` 页面可用（Lite 主入口）
- [ ] 前端 `/editor/project/:projectId` 路由可用
- [ ] 前端 `/editor/:jobId` 路由 resolve 后重定向
- [ ] Lite 模式下跳过转录状态检查，直接进入编辑
- [ ] 所有单元测试通过（~750 行测试代码）
- [ ] CI Lite 启动检查通过
- [ ] `subtitle_edit_store.py` 零改动
