"""Generate test fixture files for ingest tests."""
import os
os.makedirs("tests/fixtures", exist_ok=True)

def make_sample_pdf(path="tests/fixtures/sample.pdf"):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "FINANCIAL ANALYSIS")
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "The borrower demonstrates strong cash flow coverage.")
    c.drawString(72, 685, "DSCR is 1.35x based on trailing twelve months.")
    c.drawString(72, 670, "The covenant threshold is 1.20x per the credit agreement.")
    c.save()

if __name__ == "__main__":
    make_sample_pdf()
    print("Fixtures created.")
