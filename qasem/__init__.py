from pathlib import Path
package_homedir = Path(__file__).parent
data_dir = package_homedir / "data"
# version
with open("qasem/version.txt", "r") as f:
    version = f.read().strip()
__version__ = version