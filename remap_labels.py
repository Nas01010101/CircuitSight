import os
from pathlib import Path

# Original dataset (pcb2) class IDs -> Our model class IDs
# pcb2 names:
# 0: mouse_bite
# 1: spur
# 2: missing_hole
# 3: short
# 4: open_circuit
# 5: spurious_copper

# Our model names:
# 0: missing_hole
# 1: mouse_bite
# 2: open_circuit
# 3: short
# 4: spur
# 5: spurious_copper

mapping = {
    0: 1,  # mouse_bite -> mouse_bite
    1: 4,  # spur -> spur
    2: 0,  # missing_hole -> missing_hole
    3: 3,  # short -> short
    4: 2,  # open_circuit -> open_circuit
    5: 5   # spurious_copper -> spurious_copper
}

labels_dir = Path("data/raw/pcb2/pcb-defect-dataset/test/labels")

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

print("Remapped test labels!")

# Also update the data.yaml
yaml_path = Path("data/raw/pcb2/pcb-defect-dataset/data.yaml")
yaml_content = f"""path: {Path.cwd() / "data" / "raw" / "pcb2" / "pcb-defect-dataset"}
train: train
val: val
test: test

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
"""
yaml_path.write_text(yaml_content)
print("Updated data.yaml")
