# Time-LLM 项目 GitHub 上传指南

> 本指南帮助你将 Time-LLM 项目上传到 GitHub 私有仓库，同时排除大型模型/数据文件，并保留目录结构。

---

## 📁 一、项目结构概览

```
Time-LLM/
├── base_models/          ❌ 排除内容（保留结构）
│   ├── Qwen2.5-3B/
│   └── gpt2/
├── dataset/              ❌ 排除内容（保留结构）
│   ├── ETT-small/
│   └── prompt_bank/
├── scripts/              ❌ 排除内容（保留结构）
├── data_provider/        ✅ 完整上传
├── layers/               ✅ 完整上传
├── models/               ✅ 完整上传
├── utils/                ✅ 完整上传
├── *.md, *.py, etc.      ✅ 完整上传
└── node_modules/         ❌ 排除（npm依赖，不需要上传）
```

---

## 🚀 二、上传步骤（图形化操作）

### 步骤 1：更新 .gitignore 文件

在 VS Code 或 Antigravity 中打开 `.gitignore` 文件，替换为以下内容：

```gitignore
# 环境变量
.env

# Node.js 依赖
node_modules/

# 排除大型模型文件夹内容（保留目录结构）
base_models/**
!base_models/
!base_models/**/
!base_models/**/.gitkeep

# 排除数据集内容
dataset/**
!dataset/
!dataset/**/
!dataset/**/.gitkeep

# 排除脚本内容
scripts/**
!scripts/
!scripts/**/.gitkeep

# Python 缓存
__pycache__/
*.pyc
*.pyo

# IDE 配置（可选）
.vscode/
.idea/

# 模型检查点和输出
checkpoints/
outputs/
*.pt
*.pth
*.bin
*.safetensors
```

### 步骤 2：创建 .gitkeep 文件保留目录结构

在以下位置创建空的 `.gitkeep` 文件：

| 路径 | 作用 |
|------|------|
| `base_models/Qwen2.5-3B/.gitkeep` | 保留 Qwen 模型目录 |
| `base_models/gpt2/.gitkeep` | 保留 GPT2 模型目录 |
| `dataset/ETT-small/.gitkeep` | 保留 ETT 数据目录 |
| `dataset/prompt_bank/.gitkeep` | 保留 prompt 目录 |
| `scripts/.gitkeep` | 保留脚本目录 |

**图形化操作**：右键对应文件夹 → 新建文件 → 命名为 `.gitkeep` → 保持空白保存

### 步骤 3：在 GitHub 创建远程仓库

1. 打开浏览器，登录 [GitHub](https://github.com)
2. 点击右上角 **+** → **New repository**
3. 填写仓库名称（如 `Time-LLM`）
4. 选择 **Private**（私有）
5. **不要**勾选 "Add a README file"（因为你已有）
6. 点击 **Create repository**
7. 复制仓库地址（类似 `https://github.com/你的用户名/Time-LLM.git`）

### 步骤 4：使用 VS Code 图形化上传

**方法 A：使用"发布到 GitHub"功能（推荐新手）**

1. 打开 VS Code，进入源代码管理视图（左侧栏第三个图标，或按 `Ctrl+Shift+G`）
2. 点击 **"发布到 GitHub"** 或 **"Publish Branch"**
3. 选择 **Private repository**
4. VS Code 会自动处理

**方法 B：如果已有本地 Git 仓库**

1. 打开源代码管理视图
2. 点击 **"..."** 菜单 → **Remote** → **Add Remote...**
3. 输入远程仓库地址
4. 在"更改"栏看到所有文件变更
5. 在顶部输入框输入提交信息（如：`初次上传：完整项目结构`）
6. 点击 ✓ **提交** 按钮
7. 点击 **同步更改** 或 **Push** 按钮

### 步骤 5：创建版本标签（Tag）

用于标记这个初始版本，方便以后回溯：

**图形化操作（VS Code）**：
1. 安装扩展 **GitLens** 或 **Git Graph**（推荐）
2. 在 GitLens 侧边栏 → 右键当前提交 → **Create Tag**
3. 输入标签名：`v1.0-initial` 或 `baseline`
4. 推送标签：点击 **Push Tags**

---

## 🔐 三、私有仓库与 AI 工具访问

### 问题说明
GitHub 私有仓库的代码，网页版 ChatGPT/Gemini **无法直接访问**。

### 解决方案

| 方案 | 说明 | 推荐度 |
|------|------|--------|
| **本地 AI 工具** | 继续使用 VS Code Copilot / Antigravity，可直接读取本地代码 | ⭐⭐⭐⭐⭐ |
| **临时公开** | 需要用网页 AI 分析时，临时切换仓库为 Public | ⭐⭐⭐ |
| **复制粘贴** | 将需要分析的代码片段直接粘贴给 AI | ⭐⭐⭐ |
| **GitHub Fine-grained PAT** | 生成只读 Token，但需手动提供给 AI（有风险） | ⭐⭐ |

**推荐做法**：日常使用本地 Antigravity（已经可以访问你的所有代码），只在特殊情况临时公开仓库。

### 如何临时切换仓库可见性

1. GitHub → 你的仓库 → **Settings**
2. 滚动到底部 **Danger Zone**
3. 点击 **Change visibility** → 切换为 Public/Private
4. 使用完后记得切回 Private

---

## 📝 四、日常操作指南

### 添加/修改文件后提交

**图形化操作**：
1. 修改文件后，在源代码管理视图看到变更
2. 点击文件旁边的 **+** 暂存更改
3. 输入提交信息（简短描述做了什么）
4. 点击 ✓ 提交
5. 点击 **同步更改** 推送到 GitHub

### 撤销修改

| 操作 | 方法 |
|------|------|
| 撤销未暂存的修改 | 右键文件 → **Discard Changes** |
| 撤销已暂存的修改 | 右键文件 → **Unstage Changes**，然后 Discard |
| 撤销最近一次提交 | 源代码管理 → … → **Undo Last Commit** |

### 查看历史版本

**推荐安装扩展**：**Git Graph** 或 **GitLens**
- Git Graph：可视化显示所有分支和提交历史
- 点击任意提交可查看当时的文件状态

### 回到某个版本/标签

1. 打开 Git Graph
2. 右键目标提交或标签
3. 选择 **Checkout** 或 **Create Branch from Here**

---

## 🏷️ 五、分支和提交命名规范

### 提交信息格式建议

```
<类型>: <简短描述>

类型包括：
- feat: 新功能
- fix: 修复问题
- docs: 文档更新
- refactor: 代码重构
- test: 测试相关
- chore: 杂项（配置等）
```

**示例**：
```
docs: 更新 README 添加环境配置说明
feat: 添加 Qwen2.5 模型支持
fix: 修复数据加载路径问题
```

### 标签版本号建议

- `v1.0-initial` - 初次上传基准版本
- `v1.1-qwen` - 添加 Qwen 模型后
- `v1.2-experiment1` - 第一次实验结果
- `v2.0-major` - 重大更新

---

## 💻 六、命令行操作参考

如果你更喜欢命令行，以下是对应的 Git 命令：

```bash
# === 初始设置 ===
# 初始化 Git 仓库（如果还没有）
git init

# 添加远程仓库
git remote add origin https://github.com/你的用户名/Time-LLM.git

# === 日常操作 ===
# 查看状态
git status

# 添加所有更改
git add .

# 添加特定文件
git add 文件名

# 提交
git commit -m "feat: 添加新功能描述"

# 推送到远程
git push origin main

# === 标签操作 ===
# 创建标签
git tag -a v1.0-initial -m "初始版本"

# 推送标签到远程
git push origin v1.0-initial

# 推送所有标签
git push origin --tags

# 查看所有标签
git tag -l

# === 版本回溯 ===
# 查看提交历史
git log --oneline

# 切换到某个标签
git checkout v1.0-initial

# 回到最新版本
git checkout main

# === 撤销操作 ===
# 撤销工作目录的修改（未暂存）
git checkout -- 文件名

# 取消暂存
git reset HEAD 文件名

# 撤销最近一次提交（保留更改）
git reset --soft HEAD~1

# === 创建 .gitkeep 文件 ===
# 在 PowerShell 中
New-Item -ItemType File -Path "base_models/Qwen2.5-3B/.gitkeep" -Force
New-Item -ItemType File -Path "base_models/gpt2/.gitkeep" -Force
New-Item -ItemType File -Path "dataset/ETT-small/.gitkeep" -Force
New-Item -ItemType File -Path "dataset/prompt_bank/.gitkeep" -Force
New-Item -ItemType File -Path "scripts/.gitkeep" -Force
```

---

## ✅ 快速清单

- [ ] 更新 `.gitignore` 文件
- [ ] 创建 `.gitkeep` 空文件保留目录结构
- [ ] GitHub 创建私有仓库
- [ ] VS Code 连接远程仓库
- [ ] 提交并推送代码
- [ ] 创建 `v1.0-initial` 标签
- [ ] 推送标签到远程

---

*文档创建时间：2024-12-07*
*项目：Time-LLM*
