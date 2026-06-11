import os
import json
import re
from playwright.sync_api import sync_playwright

THEMES_FOLDER = r"F:\themes it html"
OUT_DIR = r"F:\themes_dataset_ouro"
OUT_IMG = os.path.join(OUT_DIR, "images")
os.makedirs(OUT_IMG, exist_ok=True)

def generate_react(html_str):
    react_str = html_str.replace('class="', 'className="')
    return f"<LANG_REACT> export default function Component() {{\n  return (\n    {react_str}\n  );\n}}"

def generate_nextjs(html_str):
    react_str = html_str.replace('class="', 'className="')
    return f"<LANG_NEXTJS> 'use client';\nexport default function Component() {{\n  return (\n    {react_str}\n  );\n}}"

def infer_metadata(html_str):
    # Heurísticas simples para extrair metadados das classes reais
    theme = "light"
    if re.search(r'dark|bg-gray-9|bg-black|text-white', html_str, re.IGNORECASE):
        theme = "dark"
        
    layout = "vertical"
    if re.search(r'grid|row', html_str, re.IGNORECASE):
        layout = "grid"
    elif re.search(r'flex|space-between|inline', html_str, re.IGNORECASE):
        layout = "horizontal"
        
    style = "flat"
    if re.search(r'shadow|elevation', html_str, re.IGNORECASE):
        style = "shadow"
    if re.search(r'glass|blur|backdrop', html_str, re.IGNORECASE):
        style = "glassmorphism"
        
    return theme, layout, style

def main():
    print("Iniciando Extração do DATASET OURO (V8.14)...")
    print("Vamos renderizar os sites inteiros com CSS original e capturar as bounding boxes!")
    
    metadata = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        for site_name in os.listdir(THEMES_FOLDER):
            site_path = os.path.join(THEMES_FOLDER, site_name)
            if not os.path.isdir(site_path) or site_name.startswith("."):
                continue
                
            index_path = os.path.join(site_path, "index.html")
            if not os.path.exists(index_path):
                continue
                
            print(f"Renderizando site completo: {site_name}")
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            
            try:
                page.goto(f"file://{index_path}", wait_until="networkidle")
            except Exception as e:
                print(f"Erro ao carregar {site_name}: {e}")
                continue
                
            # Procurar componentes visíveis
            for component_tag in ["header", "footer", "nav", "form"]:
                locators = page.locator(component_tag)
                count = locators.count()
                
                for i in range(count):
                    loc = locators.nth(i)
                    
                    if not loc.is_visible():
                        continue
                        
                    box = loc.bounding_box()
                    if not box or box["width"] < 50 or box["height"] < 20:
                        continue
                        
                    raw_html = loc.evaluate("el => el.outerHTML")
                    
                    # Remover scripts/styles inline do HTML capturado para ficar limpo
                    raw_html = re.sub(r'<script.*?>.*?</script>', '', raw_html, flags=re.DOTALL)
                    raw_html = re.sub(r'<style.*?>.*?</style>', '', raw_html, flags=re.DOTALL)
                    
                    theme, layout, style = infer_metadata(raw_html)
                    
                    base_name = f"{site_name}_{component_tag}_{i}"
                    img_filename = f"{base_name}.png"
                    img_path = os.path.join(OUT_IMG, img_filename)
                    
                    try:
                        loc.screenshot(path=img_path)
                    except Exception as e:
                        print(f"Erro ao capturar {base_name}: {e}")
                        continue
                        
                    # Gerar as linguagens
                    fw_html = f"<LANG_HTML> {raw_html}"
                    fw_react = generate_react(raw_html)
                    fw_next = generate_nextjs(raw_html)
                    fw_tail = f"<LANG_TAILWIND> {raw_html}" # Assume original has tailwind or semantic classes
                    
                    for fw_name, fw_code in [("html", fw_html), ("react", fw_react), ("nextjs", fw_next), ("tailwind", fw_tail)]:
                        metadata.append({
                            "original_site": site_name,
                            "component": component_tag,
                            "theme": theme,
                            "style": style,
                            "layout": layout,
                            "framework": fw_name,
                            "image_file": img_filename,
                            "target_code": fw_code
                        })
                        
        browser.close()
        
    meta_path = os.path.join(OUT_DIR, "metadata_ouro.json")
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)
        
    print(f"\nExtração Concluída! {len(metadata)} amostras de altíssima fidelidade extraídas em {OUT_DIR}.")

if __name__ == "__main__":
    main()
