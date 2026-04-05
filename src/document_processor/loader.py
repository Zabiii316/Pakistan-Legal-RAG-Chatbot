from pathlib import Path


class DocumentLoader:
    def __init__(self, raw_docs_dir: str):
        self.raw_docs_dir = Path(raw_docs_dir)

    def load_txt_documents(self) -> list[dict]:
        documents = []

        if not self.raw_docs_dir.exists():
            return documents

        for file_path in self.raw_docs_dir.glob("*.txt"):
            text = file_path.read_text(encoding="utf-8")
            documents.append(
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "text": text,
                }
            )

        return documents