# %%
import dotenv

dotenv.load_dotenv(".env")
import glob

from Utils.pdf_processor import PDFProcessor

# %%
if __name__ == "__main__":
    target_folder = "/root/pdf_parser/raw_md"
    files = glob.glob("/root/pdf_parser/raw_pdf/*.*")
    pdf_processor = PDFProcessor(files=files, target_folder=target_folder)
    pdf_processor.run_parse()
    # audio_trans = AudioTranscription(
    #     glob.glob("/pdf_parser/test_audio/*.*"), "/pdf_parser/test_audio_md"
    # )
    # audio_trans.run_parse()
# %%
