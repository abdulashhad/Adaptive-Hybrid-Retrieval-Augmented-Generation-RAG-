import os

class DocumentLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_txt_files(self):
        documents = []

        # Debug: check folder
        if not os.path.exists(self.data_path):
            print(f"[ERROR] Path not found: {self.data_path}")
            return documents

        files = os.listdir(self.data_path)

        print(f"[INFO] Found {len(files)} files")

        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(self.data_path, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    documents.append({
                        "content": text,
                        "source": file
                    })

                    print(f"[LOADED] {file}")

                except Exception as e:
                    print(f"[ERROR] Failed to read {file}: {e}")

        return documents