# 本地更新测试说明

## 测试目的
模拟在线更新流程，验证 bootloader 的更新机制是否正常工作。

## 测试流程
1. 创建模拟更新包（zip文件）
2. 写入 update_signal.json 触发更新
3. 运行 bootloader 执行更新流程
4. 验证更新结果

## 文件说明
- `create_mock_update.py` - 创建模拟更新包
- `trigger_update.py` - 手动触发更新信号
- `mock_update.zip` - 模拟更新包（由脚本生成）
