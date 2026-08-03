"""部署入口：Streamlit Cloud / Hugging Face Spaces 默认约定为根目录 app.py。

本文件仅转发到真实的 Web 应用 retailquant.webapp，保持源码结构不变。
"""

from retailquant.webapp import main  # noqa: F401

if __name__ == "__main__":
    main()
