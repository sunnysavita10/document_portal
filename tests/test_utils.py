from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


def create_dummy_pdf(file_path: str):
    """
    Creates a dummy PDF file.
    """
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(100, 750, "This is a dummy PDF file.")
    c.save()
