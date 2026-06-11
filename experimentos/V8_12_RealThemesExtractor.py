import os
import json
from bs4 import BeautifulSoup

INPUT_FOLDER = r"F:\themes it html"
OUTPUT_RAW = r"F:\themes_dataset\raw"
OUTPUT_NORM = r"F:\themes_dataset\normalized"
os.makedirs(OUTPUT_RAW, exist_ok=True)
os.makedirs(OUTPUT_NORM, exist_ok=True)

def clean_html(element):
    # Clona o elemento para não alterar o raw
    import copy
    el = copy.copy(element)
    
    # Remove scripts e estilos antigos (Level 2 - Normalized)
    for tag in el(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
        
    # Mantém classes semânticas! Remove apenas lixo como ids longos/aleatórios ou estilos inline
    for tag in el.find_all(True):
        if 'id' in tag.attrs:
            del tag['id']
        if 'style' in tag.attrs:
            del tag['style']
            
    return str(el)

def main():
    print("Iniciando Extração de Componentes Reais (V8.12)...")
    metadata = []
    
    for site_name in os.listdir(INPUT_FOLDER):
        site_path = os.path.join(INPUT_FOLDER, site_name)
        if not os.path.isdir(site_path) or site_name.startswith("."):
            continue
            
        html_file = os.path.join(site_path, "index.html")
        if not os.path.exists(html_file):
            continue
            
        print(f"Analisando: {site_name}")
        with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        
        # Procurar seções semanticamente
        sections = {
            "header": soup.find_all("header"),
            "footer": soup.find_all("footer"),
            "form": soup.find_all("form"),
            "navbar": soup.find_all("nav"),
            "section": soup.find_all("section") # Seções genéricas para features/hero
        }
        
        for section_name, elements in sections.items():
            if not elements:
                continue
                
            for i, el in enumerate(elements):
                # Filtra componentes muito vazios (ex: só uma tag sem nada dentro)
                if len(el.get_text(strip=True)) < 10 and section_name != "navbar":
                    continue
                    
                # Level 1: Raw
                raw_html = str(el)
                filename = f"{site_name}_{section_name}_{i}.html"
                
                with open(os.path.join(OUTPUT_RAW, filename), "w", encoding="utf-8") as f_out:
                    f_out.write(raw_html)
                    
                # Level 2: Normalized
                cleaned_html = clean_html(el)
                with open(os.path.join(OUTPUT_NORM, filename), "w", encoding="utf-8") as f_out:
                    f_out.write(cleaned_html)
                    
                metadata.append({
                    "site": site_name,
                    "component": section_name,
                    "file": filename
                })

    meta_path = os.path.join(r"F:\themes_dataset", "metadata_v8_12.json")
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)
        
    print(f"\nExtração Concluída! {len(metadata)} componentes extraídos para {OUTPUT_RAW} e {OUTPUT_NORM}")

if __name__ == "__main__":
    main()
