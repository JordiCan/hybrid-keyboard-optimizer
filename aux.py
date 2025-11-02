import os
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Configuración
texts = ["wizard_oz", "moby_dick"]
subfolder = "QWERTY"
images = [
    "scheduler_comparison_detailed.png",
    "fitness_evolution_random.png",
    "fitness_evolution_local.png"
]
base_input = "results/sa"
output_dir = "slides/recursos"
os.makedirs(output_dir, exist_ok=True)

for text in texts:
    input_path = os.path.join(base_input, text, subfolder)
    
    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        if os.path.exists(img_path):
            pdf_name = f"{text}_{img_name.replace('.png', '.pdf')}"
            pdf_path = os.path.join(output_dir, pdf_name)
            
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            
            c = canvas.Canvas(pdf_path, pagesize=(iw, ih))
            c.drawImage(img, 0, 0, width=iw, height=ih)
            c.showPage()
            c.save()
            
            print(f"PDF generado: {pdf_path}")
        else:
            print(f"No encontrado: {img_path}")
