import os
from pathlib import Path
import shutil
import cv2

# Mapping DeepPCB classes (1-indexed) to our YOLO classes (0-indexed)
# DeepPCB: 1:open, 2:short, 3:mousebite, 4:spur, 5:copper, 6:pin-hole
# Ours: 0:missing_hole, 1:mouse_bite, 2:open_circuit, 3:short_circuit, 4:spur, 5:spurious_copper
CLASS_MAP = {
    6: 0, # pin-hole -> missing_hole
    3: 1, # mousebite -> mouse_bite
    1: 2, # open -> open_circuit
    2: 3, # short -> short_circuit
    4: 4, # spur -> spur
    5: 5  # copper -> spurious_copper
}

def convert():
    base_dir = Path("data/raw/deep_pcb")
    out_dir = Path("data/processed/deep_pcb")
    
    # We only care about testing zero-shot generalization, so let's just parse test.txt
    test_txt = base_dir / "test.txt"
    if not test_txt.exists():
        print("DeepPCB test.txt not found.")
        return
        
    img_out = out_dir / "test" / "images"
    lbl_out = out_dir / "test" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    with open(test_txt, "r") as f:
        lines = f.read().splitlines()
        
    print(f"Converting {len(lines)} DeepPCB test images to YOLO format...")
    
    for line in lines:
        if not line.strip(): continue
        
        # DeepPCB format: "group20085/20085/20085291.jpg group20085/20085_not/20085291.txt"
        parts = line.split()
        img_path_str = parts[0].replace(".jpg", "_test.jpg")
        img_path = base_dir / img_path_str
        lbl_path = base_dir / parts[1]
        
        if not img_path.exists() or not lbl_path.exists():
            continue
            
        # Copy image
        out_img_path = img_out / img_path.name
        shutil.copy(img_path, out_img_path)
        
        # Parse labels
        yolo_lines = []
        # All DeepPCB images are 640x640, but let's read the first one to be sure if we wanted to
        w, h = 640.0, 640.0 
        
        with open(lbl_path, "r") as lf:
            for l_line in lf:
                l_parts = l_line.strip().split()
                if len(l_parts) < 5: continue
                
                x1, y1, x2, y2, cls_id = map(int, l_parts[:5])
                
                # Convert to our ontology
                our_cls = CLASS_MAP.get(cls_id)
                if our_cls is None: continue
                
                # Convert to YOLO norm (cx, cy, w, h)
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                yolo_lines.append(f"{our_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
        # Write YOLO label
        out_lbl_path = lbl_out / (img_path.stem + ".txt")
        with open(out_lbl_path, "w") as out_f:
            out_f.write("\n".join(yolo_lines))
            
    print("Done converting DeepPCB.")

if __name__ == "__main__":
    convert()
