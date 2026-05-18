from pathlib import Path
import re

def cleanup_output_folder(base_folder="Output", limit=10):
    base_path = Path(base_folder)

    pattern = re.compile(r"ligand_(\d+)")

    for subfolder in ["pdb", "sdf"]:
        folder_path = base_path / subfolder

        if not folder_path.exists():
            continue

        for file in folder_path.iterdir():
            if file.is_file():
                match = pattern.search(file.stem)

                if match:
                    number = int(match.group(1))

                    if number > limit:
                        file.unlink()
                        print(f"Deleted: {file}")