import os
from pathlib import Path

mapping = {
    0: 1,  # mouse_bite -> mouse_bite
    1: 4,  # spur -> spur
    2: 0,  # missing_hole -> missing_hole
    3: 3,  # short -> short
    4: 2,  # open_circuit -> open_circuit
    5: 5   # spurious_copper -> spurious_copper
}

base_dir = Path("data/raw/pcb2/pcb-defect-dataset")
splits = ["train", "val"]

for split in splits:
    labels_dir = base_dir / split / "labels"
    if not labels_dir.exists():
        continue
    
    count = 0
    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            old_id = int(parts[0])
            new_id = mapping.get(old_id, old_id)
            new_lines.append(f"{new_id} " + " ".join(parts[1:]))
        label_file.write_text("\n".join(new_lines) + "\n")
        count += 1
    print(f"Remapped {count} label files in {split} split.")
