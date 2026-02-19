# %%
import glob
import logging

import dotenv
import typer

dotenv.load_dotenv(".env")

from Utils.pdf_processor import PDFProcessor

logger = logging.getLogger("preprocess_files")

app = typer.Typer()


@app.command()
def main(
    input_dir: str = typer.Argument(..., help="Directory containing PDF files to process"),
    output_dir: str = typer.Argument(..., help="Directory to write JSON output files"),
    model: str = typer.Option("deepseek/deepseek-chat", help="LLM model name for metadata extraction and table summarization"),
    no_table: bool = typer.Option(False, "--no-table", help="Disable table extraction"),
):
    files = glob.glob(f"{input_dir}/*.*")
    if not files:
        logger.warning("No files found in %s. Exiting.", input_dir)
        raise typer.Exit(code=1)

    pdf_processor = PDFProcessor(
        files=files,
        target_folder=output_dir,
        do_extract_table=not no_table,
        model_name=model,
    )
    pdf_processor.run_parse()


if __name__ == "__main__":
    app()
