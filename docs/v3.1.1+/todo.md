1. 将Whisper medium换成更好的模型并测试兼容性，拓展模型管理器，添加模型切换功能和前端界面
2. 修复由于pytorch版本问题导致的SenseVoice无法跑在GPU上
3. 新流程

**Fast Stream (SenseVoice + Punctuation)**:

- SenseVoice 快速输出无标点（或标点不可信）的 Draft 文本。
- **立即** 经过一个轻量级标点模型 (CT-Transformer ONNX)。
- **关键输出**：标点模型不仅输出标点，还告诉我们 **“哪里是句号/问号”**。

**Semantic Buffer (语义缓冲区)**:

- 不再傻傻地等 12s 或 30s。
- 系统监控 Fast Stream 的输出。一旦标点模型检测到 **强终止符（句号/问号/感叹号）**，且当前缓冲时长 > 15s（Whisper 舒适区），立刻触发 **“语义提交 (Semantic Commit)”**。
- 这生成了一个 **“Semantic Chunk”**。这个 Chunk 保证是 **语义完整** 的一句话或几句话。

**Slow Stream (Whisper)**:

- Whisper 接收到的不再是被 VAD 随机切断的音频，而是一个 **完整的语义闭环**。
- **结果**：Whisper 不需要“猜”结尾，因为它就是完整的。**幻觉率将呈指数级下降。**

**Alignment (基于句子的精准对齐)**:

- Needleman-Wunsch 算法不再需要在 30s 的长文本里“大海捞针”。
- 我们可以利用标点作为 **“超级锚点”**，将对齐任务拆解为 **句子级对齐**。准确率会大幅提升。

4. 针对SenseVoice大量丢字幕的问题进行优化，强制使用Whisper结果回写完全空白但检测到有人声的部分；重新设计SenseVoice和Whisper的相互补充机制，将目前的只使用SenseVoice兜底改为更加智能的双模型交流机制（相互兜底、相互纠错、添加将两者结果同时送给后续llm仲裁的功能）
5. 使用**NISQA** 或基于 **WavLM** 的质量评估模型替换YAMNet，不再判断是否有背景声，而是判断人声的清晰度，如果清晰度足够就直接跳过人声分离
6. 构建高质量 Prompt & 避免连锁反应（基于全新架构）
7. 重新设计快流的乱序执行机制（基于全新架构）
8. 集成 **Flash Attention 2** 技术，并实现真正的 **Dynamic Batching**（动态批处理），将 SequencedQueue 中的多个 Chunk 打包一次送入 GPU。

目标是让GPU尽可能全程全速运转

9. 实现说话人分离功能，使用ModelScope 3D-Speaker (CAM++) 或Fast-Whisper-Diarization (基于 ECAPA-TDNN)拓展
10. 实现CAPA (Context-Aware Phonetic Attention)用于后处理中的同音字筛查

该算法分为三个阶段：**粗筛（Recall）** -> **注意力评分（Scoring）** -> **聚类决策（Decision）**。

#### 阶段一：基于规则的稀疏掩码 (Sparse Masking)

为了保持极致速度，我们不能让所有词两两计算 Attention（那是 $O(N^2)$ 的复杂度）。我们利用 RapidFuzz 生成一个 **Attention Mask（注意力掩码）**。

- **输入**：全文字幕中提取的所有名词集合 $V = \{w_1, w_2, ..., w_n\}$。
- **操作**：
  1. 计算 $w_i$ 与 $w_j$ 的 **Pinyin Weighted Levenshtein**（拼音加权编辑距离）。
  2. 如果距离 > 阈值（例如差异太大），则在 Mask 矩阵中设为 $-\infty$（即不进行 Attention 计算）。
- **输出**：一个稀疏的候选邻接矩阵。

#### 阶段二：双模态注意力机制 (Dual-Modal Attention)

这是引入 Attention 的核心创新点。我们构建两个特征向量：

1. **声学特征 ($F_{audio}$)**：基于拼音或 Double Metaphone 的 One-hot 或 Embedding 向量。
2. **语义特征 ($F_{context}$)**：取该词前后各 2 个词，通过一个极小的微型 Transformer（或仅使用 FastText 词向量）生成的上下文向量。

注意力公式优化：

我们定义词 $i$ 和词 $j$ 的相似度分数 $S_{ij}$ 为：

$$S_{ij} = \alpha \cdot \text{Sim}_{edit}(P_i, P_j) + (1-\alpha) \cdot \text{Attention}(C_i, C_j)$$

其中：

- $P$ 为拼音/发音特征。
- $C$ 为上下文语义特征。
- **$\alpha$ (动态门控系数)**：这是关键。**Attention 机制被用来计算 $\alpha$**。

**$\alpha$ 的计算逻辑**：

- 如果上下文语义非常清晰（Attention Score 高），则降低拼写相似度的权重（容忍更大的拼写错误）。
- 如果上下文模糊（如孤立的词），则大幅提高拼写相似度的权重（依赖硬匹配）。

#### 阶段三：基于密度的聚类

- 根据计算出的 $S_{ij}$ 矩阵，使用 **DBSCAN** 或 **Affinity Propagation** 进行聚类。
- 簇中心（Cluster Centroid）即为标准词，簇中其他点即为错词，自动执行替换。

11. 实现接入llm进行语义校对+翻译（基于全新架构）
12. 实现本地llm语义校对+翻译
13. 添加直接导出为带字幕的预览视频的功能
14. 深入优化前端的性能问题，修复已知的前端bug
15. 设计在线更新系统（重要，应该列为首位）
16. 实现完整的前端参数配置界面，要求将所有重要的参数都开放给用户