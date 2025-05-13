import fitz #PyMuPDF module

def extract_pdf_text(filepath):
    pdf_file=fitz.open(filepath)

    overall_text=""

    for page_no in range(len(pdf_file)):
        
        page=pdf_file[page_no]

        text=page.get_text()

        overall_text+=text

    pdf_file.close()

    return overall_text

if __name__=="__main__":
    filepath="sample_pdf.pdf"
    text=extract_pdf_text(filepath)
    print(text)
