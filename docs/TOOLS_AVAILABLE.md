# 可用工具清单

本文档记录本次会话中实际可调用的工具列表与用法要点，具体可用范围以
当前会话的工具清单为准。

## 1. 命令执行

- `functions.shell_command`：执行 PowerShell 命令，支持设置工作目录。
  - 关键参数：`command`、`workdir`、`timeout_ms`
  - 示例：
    ```json
    {"command":"Get-ChildItem -Force","workdir":"f:/video_to_srt_gpu"}
    ```

## 2. MCP 资源访问

- `functions.list_mcp_resources`：列出 MCP 服务器可读资源。
  - 关键参数：`server`
- `functions.list_mcp_resource_templates`：列出 MCP 服务器资源模板。
  - 关键参数：`server`
- `functions.read_mcp_resource`：读取 MCP 资源内容。
  - 关键参数：`server`、`uri`

## 3. 任务计划

- `functions.update_plan`：更新任务计划步骤状态。
  - 关键参数：`plan`（数组，包含 `step` 与 `status`）

## 4. 图像查看

- `functions.view_image`：查看本地图片（必须提供完整路径）。
  - 关键参数：`path`

## 5. 文件补丁编辑

- `functions.apply_patch`：按补丁语法修改文件内容（非 JSON 输入）。
  - 适用场景：小范围编辑或新增文件
  - 备注：输入必须符合 `*** Begin Patch` / `*** End Patch` 语法

## 6. 并行执行

- `multi_tool_use.parallel`：并行调用多个 `functions.*` 工具。
  - 关键参数：`tool_uses`（数组，包含 `recipient_name` 与 `parameters`）
