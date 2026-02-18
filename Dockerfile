FROM nvcr.io/nvidia/pytorch:26.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim ca-certificates \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Node.js 18 LTS + md-to-pdf (markdown → PDF export)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g md-to-pdf \
    && rm -rf /var/lib/apt/lists/*

# Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

# Poetry (no virtualenv — use container's system Python)
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

# ML packages — installed via pip (NOT poetry) to preserve NGC's PyTorch.
# Any package with torch in its dep tree must use --no-deps to prevent
# pip from pulling in PyPI's torch and overwriting NGC's custom build.
# Their sub-deps (transformers, huggingface-hub, etc.) are installed explicitly.
RUN pip install --no-cache-dir \
    "transformers>=4.41.0,<5.0.0" \
    "huggingface-hub>=0.33.4,<1.0.0" \
    && pip install --no-cache-dir --no-deps \
    "sentence-transformers" \
    "marker-pdf" \
    && pip install --no-cache-dir \
    "surya-ocr" \
    "pdftext" \
    "ftfy" \
    "filetype" \
    "pypdfium2==4.30.0" \
    "markdownify" \
    "google-genai"

WORKDIR /root/pdf_parser

CMD ["bash"]
