import PyPDF2
import pyttsx3

def extract_text_from_pdf(pdf_path):
    # Abre o arquivo PDF
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        # Itera sobre todas as páginas
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Caminho para o arquivo PDF
pdf_path = 'arquivo1.pdf'

# Extrai o texto do PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Converte o texto em áudio
text_to_speech(pdf_text)
