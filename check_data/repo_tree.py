import os

def print_tree(start_path, prefix=""):
    files = sorted(os.listdir(start_path))
    pointers = {
        "branch": "├── ",
        "tee":    "│   ",
        "last":   "└── ",
        "empty":  "    ",
    }

    for index, name in enumerate(files):
        path = os.path.join(start_path, name)
        connector = (
            pointers["last"] if index == len(files) - 1 else pointers["branch"]
        )

        print(prefix + connector + name)

        if os.path.isdir(path):
            extension = (
                pointers["empty"] if index == len(files) - 1 else pointers["tee"]
            )
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    root = os.getcwd()   # runs from current folder
    print(f"\n📁 Project Structure: {root}\n")
    print_tree(root)
