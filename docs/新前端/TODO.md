1.任务监控

- [x]  阶段 1：数据层重构

- [x]  阶段 2：组件化与布局

- [x] 阶段 3：拖动排序集成

- [x] 点击跳转

2.播放区域

- [x] 视频播放经常加载，有时加载不出来
- [x] [media] 获取视频时长异常: [WinError 2] 系统找不到指定的文件。, 路径: F:\video_to_srt_gpu\jobs\ec01d30e44a54e0d881c1debbf6f5aad\p03-80.mp4，[media] 缩略图生成失败: 500: 缩略图生成失败: 500: 无法获取视频时长，视频路径: F:\video_to_srt_gpu\jobs\ec01d30e44a54e0d881c1debbf6f5aad\p03-80.mp4
- [ ] 演示字幕大小、背景
- [ ] 播放进度条样式

3.播放控制区域

- [ ] 倍速
- [ ] 

4.波形区域

- [x] 圆角
- [x] 波形无法左右拖动或滚动
- [x] wavesurfer.js波形图空隙异常大
- [ ] 字幕时间切分将空白处纳入

5.字幕区域

- [ ] 字幕检查过于严格
- [ ] 转录流式生成的字幕时长过长
- [ ] 全局搜索
- [ ] 单字幕滚动条

6.模型管理

7.全局设置相关

- [ ] 设置菜单
- [ ] 退出按钮和自动清理机制
- [ ] title美观

8.动态进度区

- [ ] 灵动岛动效
- [ ] 重启恢复任务出现阶段回退，有时甚至重新开始

9.其他

- [ ] 导出菜单
- [ ] 幽灵任务，队列完成后出现
- [ ] 快捷键
- [ ] 后端写入日志

---

10.翻译

11.校对





BGM检测完成: 比例=[0.01868760338502411, 0.01807172506509757, 0.04894797753079488], 平均=0.03, 最大=0.05
00:08:24.058 [INFO] [transcription_service] BGM检测结果: light, 比例=[0.01868760338502411, 0.01807172506509757, 0.04894797753079488], 最大=0.05
00:08:24.088 [INFO] [transcription_service] BGM检测完成: light



实时连接已断开，正在尝试重连





进度条回退





EditorView.vue:856  视频加载错误: Error: 视频加载失败，正在重试 (1/3)...
    at onError (index.vue:429:17)
    at callWithErrorHandling (chunk-256OB4QR.js?v=cf2fc920:2560:19)
    at callWithAsyncErrorHandling (chunk-256OB4QR.js?v=cf2fc920:2567:17)
    at HTMLVideoElement.invoker (chunk-256OB4QR.js?v=cf2fc920:11679:5)claude-sonnet-4-5-20250929[1m]
handleVideoError @ EditorView.vue:856
index.vue:332  [VideoStage] 视频源切换失败: Error: 加载失败
    at HTMLVideoElement.onError (index.vue:314:18)





curl -v -H "Authorization: Bearer wgh-gemini-c" "https://wgh-gb.zeabur.app/v1beta/models"


F:\video_to_srt_gpu\.venv\lib\site-packages\torch\nn\modules\activation.py:1230: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)
  return torch._native_multi_head_attention(



1.Post-VAD 智能合并层
2：强制关键复核
第2点特别注意：默认使用medium模型，但提供更高的配置选项，whisper模型统一接入现有的模型下载管理器；对于强制对齐的细节在文档末尾还有讨论仔细阅读分析。
3.过滤 Tags
4.VAD 边缘吸附
5.Word-Level Trigger（字级触发）

补充：1.理论上项目已经实现了对于字级置信度低的字在前端高亮的代码，但现在没有正常显示，分析是否忘记了在前端集成
2.进行所有上述的修改后，要对四层分句系统进行重新审视，看哪些部分需要做调整