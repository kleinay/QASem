from pathlib import Path
package_homedir = Path(__file__).parent
data_dir = package_homedir / "data"
# version
with open(package_homedir / "version.txt", "r") as f:
    __version__ = f.read().strip()
