import os
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

def main():
    # Set up directories
    input_dir = 'assets'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading DDColor pipeline...")
    colorizer = pipeline(Tasks.image_colorization, model='damo/cv_ddcolor_image-colorization')
    
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    print(f"Running inference on {len(image_files)} image(s)...\n")
    
    for filename in image_files:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, f"colorized_{filename}")
        
        try:
            result = colorizer(in_path)
            cv2.imwrite(out_path, result[OutputKeys.OUTPUT_IMG])
            print(f"[Done] {filename} -> {out_path}")
        except Exception as e:
            print(f"[Error] Could not process {filename}: {e}")

if __name__ == "__main__":
    main()
