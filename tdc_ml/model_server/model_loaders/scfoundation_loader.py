import os


class scFoundationLoader:

    GDRIVE_FILE_ID = "1DK_QrNfCUQU17uWuVI-0ZXr9G3wRJXdc"
    GENE_VOCAB_URL = (
        "https://raw.githubusercontent.com/biomap-research/scFoundation/"
        "main/model/OS_scRNA_gene_index.19264.tsv"
    )
    DEFAULT_SAVE_DIR = "scfoundation_model"
    CHECKPOINT_NAME = "models.ckpt"

    def __init__(self):
        pass

    def load(self, checkpoint_path=None):
        canonical_path = os.path.join(self.DEFAULT_SAVE_DIR, self.CHECKPOINT_NAME)

        search_paths = []
        if checkpoint_path:
            search_paths.append(checkpoint_path)
        search_paths += [
            canonical_path,
            os.path.expanduser("~/scFoundation/model/models/models.ckpt"),
        ]

        for path in search_paths:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print(f"scFoundation checkpoint found at {path}")
                return path

        if self.GDRIVE_FILE_ID is None:
            raise FileNotFoundError(
                "scFoundation checkpoint not found at "
                f"{os.path.abspath(canonical_path)}.\n"
                "Upload models.ckpt to Google Drive, set GDRIVE_FILE_ID, and retry."
            )

        print("scFoundation checkpoint not found locally. Downloading from Google Drive...")
        return self._download_from_gdrive(canonical_path)

    def _download_from_gdrive(self, output_path):
        try:
            import gdown
        except ImportError:
            raise ImportError("gdown is required: pip install gdown")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        url = f"https://drive.google.com/uc?id={self.GDRIVE_FILE_ID}"
        print(f"  File ID: {self.GDRIVE_FILE_ID}")
        print(f"  Dest   : {os.path.abspath(output_path)}")

        gdown.download(url, output_path, quiet=False, fuzzy=True)

        size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        if size < 100 * 1024 * 1024:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(
                f"Download failed — file too small ({size / 1024**2:.1f}MB). "
                "Make sure the Google Drive file is shared as 'Anyone with the link'."
            )

        print(f"scFoundation checkpoint saved to {output_path} ({size / 1024**2:.0f}MB)")
        return output_path

    def load_gene_vocab(self, save_dir="scfoundation_model"):
        import requests

        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "OS_scRNA_gene_index.19264.tsv")

        if os.path.exists(output_path):
            return output_path

        print("Downloading scFoundation gene vocabulary...")
        try:
            response = requests.get(self.GENE_VOCAB_URL, timeout=30)
            response.raise_for_status()
            with open(output_path, "w") as f:
                f.write(response.text)
            print(f"Gene vocabulary saved to {output_path}")
            return output_path

        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"Failed to download gene vocabulary: {e}")
