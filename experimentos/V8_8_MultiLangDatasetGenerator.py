import os
import json
import random
import pathlib
from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path("dataset_multilang")
IMG_DIR = OUT_DIR / "images"
CODE_DIR = OUT_DIR / "code"
META_DIR = OUT_DIR / "metadata"

for d in [IMG_DIR, CODE_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Vocabulários sintéticos
LOGOS = ["Logo", "Brand", "MySite", "App", "System"]
LINKS = ["Home", "About", "Services", "Contact", "Dashboard"]

def random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

def generate_header_flavors(idx_str):
    bg_color = random_color()
    text_color = "#ffffff" if random.choice([True, False]) else "#000000"
    padding = f"{random.randint(10, 30)}px"
    
    logo = random.choice(LOGOS)
    link1, link2 = random.sample(LINKS, 2)
    
    # --- 1. HTML Base (Apenas para renderizar a imagem) ---
    style = f"width: 100%; box-sizing: border-box; display: flex; justify-content: space-between; align-items: center; padding: {padding}; background-color: {bg_color}; color: {text_color}; font-family: sans-serif;"
    html_pure = f'<header style="{style}"><h1>{logo}</h1><nav><span>{link1}</span> | <span>{link2}</span></nav></header>'
    
    # --- 2. Variantes de Linguagem ---
    
    # A. HTML
    target_html = f"<LANG_HTML> <header>\n  <h1>{logo}</h1>\n  <nav>\n    <span>{link1}</span>\n    <span>{link2}</span>\n  </nav>\n</header>"
    
    # B. React (JSX Puro)
    target_react = f"<LANG_REACT> export default function Header() {{\n  return (\n    <header style={{{{ display: 'flex', justifyContent: 'space-between', padding: '{padding}', backgroundColor: '{bg_color}', color: '{text_color}' }}}}>\n      <h1>{logo}</h1>\n      <nav>\n        <span>{link1}</span>\n        <span>{link2}</span>\n      </nav>\n    </header>\n  );\n}}"
    
    # C. Next.js (com Link)
    target_next = f"<LANG_NEXTJS> import Link from 'next/link';\n\nexport default function Header() {{\n  return (\n    <header style={{{{ display: 'flex', justifyContent: 'space-between', padding: '{padding}', backgroundColor: '{bg_color}', color: '{text_color}' }}}}>\n      <h1>{logo}</h1>\n      <nav>\n        <Link href=\"/\">{link1}</Link>\n        <Link href=\"/about\">{link2}</Link>\n      </nav>\n    </header>\n  );\n}}"
    
    # D. Tailwind (Classes Utilitárias)
    target_tailwind = f"<LANG_TAILWIND> export default function Header() {{\n  return (\n    <header className=\"flex justify-between items-center p-4 bg-gray-800 text-white\">\n      <h1 className=\"text-xl font-bold\">{logo}</h1>\n      <nav className=\"flex gap-4\">\n        <span>{link1}</span>\n        <span>{link2}</span>\n      </nav>\n    </header>\n  );\n}}"

    return html_pure, {
        "html": target_html,
        "react": target_react,
        "nextjs": target_next,
        "tailwind": target_tailwind
    }

def main():
    print("Gerando Dataset V8.8: Múltiplas Linguagens para o mesmo Componente (Headers)...")
    samples = 500 
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1024, "height": 400}) 
        
        for idx in range(1, samples + 1):
            idx_str = f"{idx:06d}"
            
            html_pure, flavors = generate_header_flavors(idx_str)
            
            # HTML para a screenshot
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head><meta charset="utf-8"></head>
            <body style="margin: 0; padding: 20px; background-color: #f0f0f0;">
                {html_pure}
            </body>
            </html>
            """
            
            html_path = CODE_DIR / f"{idx_str}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            # Gerar screenshot
            page.goto(f"file://{html_path.resolve()}")
            element = page.locator("body > *").first
            if element.count() > 0:
                element.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
            else:
                page.screenshot(path=str(IMG_DIR / f"{idx_str}.png"))
                
            # Salvar 4 arquivos de metadata (uma para cada linguagem)
            for lang, target_code in flavors.items():
                meta = {
                    "id": idx_str,
                    "image_file": f"{idx_str}.png",
                    "language": lang,
                    "target_code": target_code
                }
                meta_filename = f"{idx_str}_{lang}.json"
                with open(META_DIR / meta_filename, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                    
            if idx % 50 == 0:
                print(f"{idx}/{samples} Headers gerados (com 4 linguagens cada = {idx*4} amostras de treino)...")
                
        browser.close()
    print(f"\nDataset MultiLang gerado! Total de Imagens: {samples} | Total de Pares de Treino: {samples * 4}")

if __name__ == "__main__":
    main()
