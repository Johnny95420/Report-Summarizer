# %%
import glob
import re

import tqdm
from langchain.schema import Document
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# %%
converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
# %%
target_folder = "/pdf_parser/parsed_file_6937"
files = glob.glob("/pdf_parser/files_6937/*.pdf")
documents = []
for file in tqdm.tqdm(files):
    rendered = converter(file)
    text, _, images = text_from_rendered(rendered)
    pattern = r"!\[\]\((.*?)\)"
    text = re.sub(pattern, "", text)
    documents.append(Document(text, metadata={"file": file}))

for doc in documents:
    path = doc.metadata["file"]
    new_path = path.split("/")[-1].replace(".pdf", ".txt")
    with open(f"{target_folder}/{new_path}", "w", encoding="utf-8") as file:
        file.write(doc.page_content)

# %%
