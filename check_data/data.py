import os
import pandas as pd
import json

# -------------------------------------------------------------------
# Utility: Safe CSV load
# -------------------------------------------------------------------
def load_csv_safe(path):
    try:
        df = pd.read_csv(path)
        print(f"\n[✔] Loaded CSV: {path}")
        return df
    except Exception as e:
        print(f"[✘] Failed to read CSV {path}: {e}")
        return None

# -------------------------------------------------------------------
# Inspect a CSV file
# -------------------------------------------------------------------
def inspect_csv(path):
    df = load_csv_safe(path)
    if df is None:
        return

    print("=" * 80)
    print(f"📌 CSV FILE: {path}")

    print("\n📊 Shape:", df.shape)
    print("\n🔍 Columns:", df.columns.tolist())

    print("\n📌 Missing values:")
    print(df.isnull().sum())

    print("\n📈 Data types:")
    print(df.dtypes)

    print("\n📌 Numerical summary:")
    print(df.describe())

    print("\n📌 Categorical summary:")
    print(df.describe(include='object'))

# -------------------------------------------------------------------
# Inspect image/PDF folder
# -------------------------------------------------------------------
def inspect_image_folder(path):
    if not os.path.exists(path):
        print(f"[✘] Folder not found: {path}")
        return

    files = os.listdir(path)
    images = [
        f for f in files 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff'))
    ]

    print("=" * 80)
    print(f"📁 FOLDER: {path}")
    print(f"📦 Total files: {len(files)}")
    print(f"🖼️ Total documents/images: {len(images)}")

    if images:
        print("\n🔍 Sample files:", images[:10])

        # Count by type
        type_count = {}
        for f in images:
            ext = f.split('.')[-1].lower()
            type_count[ext] = type_count.get(ext, 0) + 1

        print("\n📌 File type distribution:", type_count)
    else:
        print("❗ No images/PDF files found.")

# -------------------------------------------------------------------
# Inspect JSON labels folder
# -------------------------------------------------------------------
def inspect_json_labels(folder):
    if not os.path.exists(folder):
        print(f"[✘] JSON folder not found: {folder}")
        return

    json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    print("=" * 80)
    print(f"📁 JSON LABELS: {folder}")
    print(f"📦 Total JSON files: {len(json_files)}")

    if not json_files:
        return

    sample_file = os.path.join(folder, json_files[0])
    print(f"\n🔍 Sample JSON file: {sample_file}")

    with open(sample_file, 'r') as f:
        data = json.load(f)

    print("\n📝 JSON keys:", data.keys())
    print("\n📌 Sample content:")
    print(json.dumps(data, indent=2)[:1000], "...")

# -------------------------------------------------------------------
# Auto-detect CSVs, images, and JSONs
# -------------------------------------------------------------------
def auto_detect_and_inspect(root):
    print("=" * 80)
    print(f"🔍 AUTO SCANNING DATASET UNDER: {root}")
    print("=" * 80)

    for subdir, dirs, files in os.walk(root):

        # Inspect CSV files
        for file in files:
            if file.endswith('.csv'):
                inspect_csv(os.path.join(subdir, file))

        # Inspect image/PDF folders
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')) for f in files):
            inspect_image_folder(subdir)

        # Inspect JSON label folders
        if any(f.endswith('.json') for f in files):
            inspect_json_labels(subdir)

# -------------------------------------------------------------------
# Run script
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Your actual dataset path
    auto_detect_and_inspect("../data")
