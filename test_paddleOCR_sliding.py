from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import ssl

# Disable SSL verification (add these lines at the start)
ssl._create_default_https_context = ssl._create_unverified_context

# Paddleocr supports Chinese, English, French, German, Korean and Japanese
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)  # Enable GPU if installed

img_path = (
    r"C:\Users\pcadmin\Desktop\YSA AI\Traceability\Data"
    r"\19-11-2024\20241119_144922.jpg"
)

slice = {
    'horizontal_stride': 500,
    'vertical_stride': 500,
    'merge_x_thres': 50,
    'merge_y_thres': 50
}

results = ocr.ocr(img_path, cls=True, slice=slice)

# Load image
image = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("C:/Windows/Fonts/Arial.ttf", size=50)  # Adjust size as needed

# Process and draw results
for res in results:
    for line in res:
        box = [tuple(point) for point in line[0]]
        # Finding the bounding box
        box = [(min(point[0] for point in box), min(point[1] for point in box)),
               (max(point[0] for point in box), max(point[1] for point in box))]
        txt = line[1][0]
        draw.rectangle(box, outline="red", width=2)  # Draw rectangle
        draw.text((box[0][0], box[0][1] - 25), txt, fill="blue", font=font)  # Draw text above the box

# Save the processed image
image.save(r"C:\Users\pcadmin\Desktop\YSA AI\Traceability\Processed\output_image.jpg")  # Specify the output path
        