# 部署指南（Streamlit 托管平台）

## 为什么不使用 Vercel

Vercel 是无状态 Serverless 平台，**原生无法运行 Streamlit 这类需要常驻进程、监听端口、维持 WebSocket 的 Python 应用**（其函数最多运行数秒、不保留长进程）。在 Vercel 上跑 Streamlit 只能靠 Node 包装 Python 子进程的 hack，不稳定且违反其长时间进程条款。

因此本项目改用对 Python/Streamlit **原生支持**的托管平台，部署最顺畅、最贴合技术栈。

## 方案 A：Streamlit Community Cloud（推荐，免费）

专为 Streamlit 设计，零配置。

1. 把本项目推到 GitHub 仓库（需先 `git init` + 关联 remote + push）。
2. 打开 https://share.streamlit.io ，用 GitHub 登录。
3. 点击 **New app** ，选择本仓库。
4. **Main file path** 填：`app.py`（根目录入口，已创建并转发到 `retailquant/webapp.py`）。
5. Python version 选 **3.11**（与本地一致），Branch 选含代码的分支。
6. 点击 **Deploy** 。平台会自动按 `requirements.txt` 安装依赖并启动。

> `requirements.txt` 已包含 `streamlit / pandas / numpy / akshare / matplotlib` 等全部依赖，无需额外配置。

## 方案 B：Hugging Face Spaces（免费，备选）

1. 在 https://huggingface.co/spaces 新建 Space，SDK 选 **Streamlit**。
2. 将仓库内容（含 `app.py`、`requirements.txt`、`retailquant/`）上传。
3. 平台自动识别 `app.py` 并运行，几分钟内上线，获得 `*.hf.space` 公网地址。

## 本地入口说明

- 根目录 `app.py` 仅作部署约定入口，转发到 `retailquant/webapp.py:main()`。
- 本地照常运行：`python -m streamlit run retailquant/webapp.py --server.port 8501`
  或直接 `python -m streamlit run app.py`。

## 注意事项

- 持仓助手依赖本地 `portfolio.json`，部署后请通过页面"保存持仓"功能在线上维护，或重新上传该文件。
- `data_cache/`、`output/` 在免费平台为临时存储，重启可能清空；回测数据需按需重新拉取。
- 免费平台对 `akshare` 高频访问可能受限，属正常现象。
