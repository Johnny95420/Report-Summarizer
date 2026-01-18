# UV 可移植環境使用說明

本專案使用 [uv](https://github.com/astral-sh/uv) 管理 Python 環境，確保在不同機器上能還原完全相同的依賴版本。

## 前置需求

1. 安裝 uv：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. 或使用 pip 安裝：
   ```bash
   pip install uv
   ```

## 快速開始

### 1. 還原環境（在新機器上）

```bash
cd /path/to/pdf_parser
uv sync
```

這會：
- 創建 `.venv` 虛擬環境
- 根據 `uv.lock` 安裝所有依賴（178 個套件）

### 2. 執行程式

```bash
# 使用 uv run 自動使用虛擬環境
uv run python run_research_report.py

# 或啟動 shell 進入虛擬環境
uv shell
python run_research_report.py
```

## 專案結構

| 檔案 | 用途 |
|------|------|
| `pyproject.toml` | 專案配置與直接依賴（18 個核心套件） |
| `uv.lock` | 鎖定所有依賴的精確版本（含間接依賴） |
| `.venv/` | 虛擬環境目錄（自動生成） |

## 常用命令

```bash
# 同步環境（安裝/更新依賴）
uv sync

# 執行 Python 檔案
uv run python <檔案名>.py

# 安裝新套件
uv add <套件名>

# 移除套件
uv remove <套件名>

# 更新所有依賴並重新鎖定
uv lock --upgrade

# 查看已安裝套件
uv pip list

# 執行測試
uv run pytest
```

## 環境變數

專案需要 `.env` 檔案配置 API 金鑰：

```bash
cp .env.example .env  # 編輯 .env 填入金鑰
```

## 依賴摘要

- **直接依賴**: 18 個核心套件
- **總依賴數**: 178 個套件（含間接依賴）
- **Python 版本**: >= 3.12

核心套件包括：LangChain 生態系、PyTorch、ChromaDB、Tavily 等。
