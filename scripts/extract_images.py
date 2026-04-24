import os
import fitz  # PyMuPDF

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            file_name = f"page_{page_index + 1}_img_{img_index}.{image_ext}"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "wb") as f:
                f.write(image_bytes)

    doc.close()

extract_images_from_pdf(r"C:\Users\user\working\MDM\data\raw\230630_자동차사고 과실비율 인정기준_최종.pdf", r"C:\Users\user\working\MDM\data\images")