import os


class scFoundationLoader:
    """
    Loader for scFoundation model weights and gene vocabulary.

    Downloads pre-trained checkpoint from Figshare and gene vocabulary
    from the scFoundation GitHub repository.

    Reference pattern: scvi_loader.py
    """

    CHECKPOINT_URL = "https://figshare.com/ndownloader/files/42466884"
    GENE_VOCAB_URL = "https://raw.githubusercontent.com/biomap-research/scFoundation/main/model/OS_scRNA_gene_index.19264.tsv"

    def __init__(self):
        pass

    def load(self, save_dir="scfoundation_model"):
        """
        Download scFoundation checkpoint.

        Args:
            save_dir: Directory to save the checkpoint.

        Returns:
            Path to the downloaded checkpoint file.
        """
        import requests

        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "models.ckpt")

        if os.path.exists(output_path):
            print(f"scFoundation checkpoint already exists at {output_path}")
            return output_path

        print("Downloading scFoundation checkpoint from Figshare...")
        print("Note: This is a ~400MB file and may take a few minutes.")

        try:
            response = requests.get(self.CHECKPOINT_URL, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\rProgress: {pct:.1f}%", end="", flush=True)

            print(f"\nscFoundation checkpoint downloaded to {output_path}")
            return output_path

        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"Failed to download scFoundation checkpoint: {e}")

    def load_gene_vocab(self, save_dir="scfoundation_model"):
        """
        Download scFoundation gene vocabulary.

        Args:
            save_dir: Directory to save the vocabulary.

        Returns:
            Path to the downloaded vocabulary file.
        """
        import requests

        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "OS_scRNA_gene_index.19264.tsv")

        if os.path.exists(output_path):
            print(f"Gene vocabulary already exists at {output_path}")
            return output_path

        print("Downloading scFoundation gene vocabulary...")

        try:
            response = requests.get(self.GENE_VOCAB_URL)
            response.raise_for_status()

            with open(output_path, "w") as f:
                f.write(response.text)

            print(f"Gene vocabulary downloaded to {output_path}")
            return output_path

        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(f"Failed to download gene vocabulary: {e}")
