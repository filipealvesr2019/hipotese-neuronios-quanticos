import os
import json
import random
import re
from playwright.sync_api import sync_playwright

NORM_FOLDER = r"F:\themes_dataset\normalized"
OUT_IMG = r"F:\themes_dataset\images"
os.makedirs(OUT_IMG, exist_ok=True)

THEMES = [
    {"name": "dark", "classes": "bg-gray-900 text-white"},
    {"name": "light", "classes": "bg-white text-gray-800"},
    {"name": "brand", "classes": "bg-blue-600 text-white"}
]

STYLES = [
    {"name": "minimal", "classes": "p-4 border-b border-gray-200"},
    {"name": "glassmorphism", "classes": "p-6 backdrop-blur-md bg-white/30 shadow-lg border border-white/20"},
    {"name": "flat", "classes": "p-8 shadow-md rounded-xl"}
]

LAYOUTS = [
    {"name": "horizontal", "classes": "flex flex-row justify-between items-center"},
    {"name": "vertical", "classes": "flex flex-col gap-4"},
    {"name": "grid", "classes": "grid grid-cols-1 md:grid-cols-2 gap-6"}
]

def generate_react(html_str):
    react_str = html_str.replace('class="', 'className="')
    return f"<LANG_REACT> export default function Component() {{\n  return (\n    {react_str}\n  );\n}}"

def generate_nextjs(html_str):
    react_str = html_str.replace('class="', 'className="')
    return f"<LANG_NEXTJS> 'use client';\nexport default function Component() {{\n  return (\n    {react_str}\n  );\n}}"

def generate_tailwind(html_str):
    return f"<LANG_TAILWIND> {html_str}"

def inject_classes(html_str, classes_str):
    # Procura a primeira tag que possui "class" e injeta no inicio dela
    if 'class="' in html_str:
        return html_str.replace('class="', f'class="{classes_str} ', 1)
    else:
        # Se nao tiver class, cria uma
        return re.sub(r'^(<[a-z0-9]+)', r'\1 class="' + classes_str + '"', html_str, count=1)

def main():
    print("Iniciando Geração de Variantes com Screenshots (V8.13)...")
    
    with open(r"F:\themes_dataset\metadata_v8_12.json", "r", encoding="utf-8") as f:
        base_meta = json.load(f)
        
    variants_meta = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        
        # Injeta Tailwind via CDN para as screenshots funcionarem
        tailwind_cdn = '<script src="https://cdn.tailwindcss.com"></script>'
        
        for item in base_meta:
            file_path = os.path.join(NORM_FOLDER, item["file"])
            if not os.path.exists(file_path): continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                base_html = f.read()
                
            base_name = item["file"].replace(".html", "")
                
            for i in range(3): # 3 variantes visuais por componente
                theme = random.choice(THEMES)
                style = random.choice(STYLES)
                layout = random.choice(LAYOUTS)
                
                combined_classes = f"{theme['classes']} {style['classes']} {layout['classes']}"
                variant_html = inject_classes(base_html, combined_classes)
                
                # HTML para renderização (Screenshot)
                render_html = f"<!DOCTYPE html><html><head>{tailwind_cdn}</head><body class='p-10'>{variant_html}</body></html>"
                temp_html_path = os.path.join(r"F:\themes_dataset", "temp_render.html")
                with open(temp_html_path, "w", encoding="utf-8") as f_temp:
                    f_temp.write(render_html)
                
                # Tirar Screenshot
                page.goto(f"file://{temp_html_path}")
                img_filename = f"{base_name}_v{i}.png"
                img_path = os.path.join(OUT_IMG, img_filename)
                
                try:
                    # Tenta capturar só o elemento raiz
                    page.locator("body > *").first.screenshot(path=img_path)
                except:
                    # Fallback para capturar a página inteira
                    page.screenshot(path=img_path)
                    
                # Gerar códigos alvos
                fw_html = f"<LANG_HTML> {variant_html}"
                fw_react = generate_react(variant_html)
                fw_next = generate_nextjs(variant_html)
                fw_tail = generate_tailwind(variant_html)
                
                # Salvar 4 metadados (uma para cada framework)
                for fw_name, fw_code in [("html", fw_html), ("react", fw_react), ("nextjs", fw_next), ("tailwind", fw_tail)]:
                    variants_meta.append({
                        "original_site": item["site"],
                        "component": item["component"],
                        "theme": theme["name"],
                        "style": style["name"],
                        "layout": layout["name"],
                        "framework": fw_name,
                        "image_file": img_filename,
                        "target_code": fw_code
                    })
                    
        browser.close()
        
    meta_path = os.path.join(r"F:\themes_dataset", "metadata_v8_13_variants.json")
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(variants_meta, f_meta, indent=2)
        
    print(f"Geração Concluída! {len(variants_meta)} pares Imagem-Código criados com metadados semânticos!")

if __name__ == "__main__":
    main()
