# 贡献指南 / Contributing Guide

感谢您对 Whisper Turbo 项目的关注！我们欢迎各种形式的贡献。

## 🚀 快速开始

### 开发环境设置

```bash
# 1. Fork 并克隆项目
git clone https://github.com/tqpp/whisper-turbo.git
cd whisper-turbo

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装开发依赖
pip install -e .[dev]

# 4. 运行测试
python examples/quick_demo.py
```

## 📝 贡献类型

### 🐛 Bug 报告
- 使用 [Issue 模板](https://github.com/tqpp/whisper-turbo/issues/new)
- 提供详细的复现步骤
- 包含系统信息和错误日志

### ✨ 功能请求
- 描述功能的使用场景
- 说明预期的行为
- 考虑性能影响

### 🔧 代码贡献
- Fork 项目并创建功能分支
- 遵循代码风格指南
- 添加必要的测试
- 更新相关文档

## 📋 代码风格

### Python 代码规范
```bash
# 格式化代码
black whisper/
isort whisper/

# 检查代码质量
flake8 whisper/
```

### 提交信息规范
```
类型(范围): 简短描述

详细描述（可选）

- 修复了什么问题
- 添加了什么功能
- 影响范围

Closes #issue_number
```

类型：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关

## 🧪 测试指南

### 运行测试
```bash
# 基础功能测试
python examples/quick_demo.py

# 性能基准测试
python benchmark.py --quick-demo

# 单元测试
pytest tests/
```

### 添加测试
- 为新功能添加测试用例
- 确保测试覆盖率
- 测试边界情况

## 📚 文档贡献

### 文档类型
- API 文档：代码中的 docstring
- 使用指南：README 和示例
- 技术文档：算法说明和优化细节

### 文档规范
- 中英文双语支持
- 包含代码示例
- 保持简洁明了

## 🔄 Pull Request 流程

1. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **开发和测试**
   - 实现功能
   - 添加测试
   - 更新文档

3. **提交代码**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

4. **创建 PR**
   - 填写 PR 模板
   - 关联相关 Issue
   - 等待代码审查

## 🎯 优化贡献

### 性能优化
- 算法优化
- 内存优化
- GPU 加速
- 批处理优化

### 优化指南
1. **基准测试**：先建立性能基线
2. **分析瓶颈**：使用 profiler 找到热点
3. **实现优化**：保持代码可读性
4. **验证效果**：确保精度不下降

## 🏷️ 发布流程

### 版本号规范
- 主版本：不兼容的 API 变更
- 次版本：向后兼容的功能性新增
- 修订版本：向后兼容的问题修正

### 发布检查清单
- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] 性能基准测试
- [ ] 更新 CHANGELOG.md

## 🤝 社区准则

### 行为准则
- 尊重他人
- 建设性讨论
- 包容不同观点
- 帮助新贡献者

### 沟通渠道
- GitHub Issues：Bug 报告和功能请求
- GitHub Discussions：技术讨论
- Pull Requests：代码审查

## 📞 联系方式

- 项目维护者：[@tqpp](https://github.com/tqpp)
- 项目主页：https://github.com/tqpp/whisper-turbo
- 问题反馈：https://github.com/tqpp/whisper-turbo/issues

## 🙏 致谢

感谢所有贡献者的努力！您的贡献让 Whisper Turbo 变得更好。

---

**让语音识别更快更强！** 🚀