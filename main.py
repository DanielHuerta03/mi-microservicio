from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import requests
import pdfplumber
import re
import os

app = FastAPI()

# Inicializa solo una vez el modelo

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = AutoModel.from_pretrained('distilbert-base-multilingual-cased')


# ------------------- Embedding y Similarity --------------------
class TextRequest(BaseModel):
    text: str

class SimilarityRequest(BaseModel):
    jd_embedding: list  # El embedding ya generado del puesto de trabajo
    candidates: list    # Lista de embeddings de los candidatos

def get_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    emb = output.last_hidden_state.mean(dim=1).numpy()
    return emb[0].tolist()

@app.post("/embedding")
def embedding(req: TextRequest):
    emb = get_embedding(req.text)
    return {"embedding": emb}

@app.post("/similarity")
def similarity(req: SimilarityRequest):
    jd_emb = np.array(req.jd_embedding)
    candidate_embs = np.array(req.candidates)
    scores = cosine_similarity([jd_emb], candidate_embs)[0]
    return {"scores": scores.tolist()}

# ------------------- Parseo de CV PDF desde URL --------------------
section_titles = {
    "habilidades": [
        "Habilidades", "Skills", "Habilidades Técnicas", "Technical Skills",
        "Competencias", "Competencias Técnicas", "Technical Competencies",
        "Habilidades Blandas", "Soft Skills", "Tecnologías", "Technologies", "Aptitudes"
    ],
    "educacion": [
        "Formación", "Educación", "Education", "Formación Académica", "Academic Background",
        "FORMACIÓN", "Estudios", "Studies", "Certificados"
    ],
    "experiencia": [
        "Experiencia", "Experience", "Experiencia Laboral", "Work Experience",
        "Trayectoria Profesional", "Professional Background", "Experiencia Profesional", "Proyectos Académicos"
    ]
}

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += "\n" + page_text
    return re.sub(r'\n+', '\n', text).strip()

def extract_section(text, section_names, other_sections):
    for sec in section_names:
        pattern = rf"^\s*{sec}[:\-\s]*\n?([\s\S]+?)(?=^\s*(?:{'|'.join(other_sections)})[:\-\s]*\n|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            section_text = match.group(1).strip()
            clean_lines = [
                line for line in section_text.splitlines()
                if line.strip() and not re.search(r'\b\d{7,}\b|@|www\.|facebook\.com|linkedin\.com', line, re.IGNORECASE)
            ]
            cleaned_section = "\n".join(clean_lines)
            if len(cleaned_section.split()) > 2:
                return cleaned_section
    return None

@app.post("/parse_cv_url")
async def parse_cv_url(cv_url: str = Form(...)):
    temp_file = "temp_cv.pdf"
    try:
        # Descarga el PDF desde la URL
        with requests.get(cv_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_file, "wb") as f:
                f.write(r.content)
        # Procesa el PDF
        text = extract_text_from_pdf(temp_file)
        habilidades = extract_section(text, section_titles["habilidades"], section_titles["educacion"] + section_titles["experiencia"])
        educacion = extract_section(text, section_titles["educacion"], section_titles["habilidades"] + section_titles["experiencia"])
        experiencia = extract_section(text, section_titles["experiencia"], section_titles["habilidades"] + section_titles["educacion"])
        os.remove(temp_file)
        return JSONResponse({
            "texto_extraido": text,
            "habilidades": habilidades,
            "educacion": educacion,
            "experiencia": experiencia
        })
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return JSONResponse({"error": str(e)}, status_code=400)
