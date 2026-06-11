import os
import json
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

OUT_DIR = r"F:\themes_dataset_ultralux"
OUT_IMG = os.path.join(OUT_DIR, "images")
os.makedirs(OUT_IMG, exist_ok=True)

THEMES = {
    "light": {"bg": "bg-white", "text_h": "text-gray-900", "text_p": "text-gray-500", "bg_sec": "bg-gray-100", "btn": "bg-indigo-600 text-white"},
    "dark": {"bg": "bg-slate-900", "text_h": "text-white", "text_p": "text-slate-300", "bg_sec": "bg-slate-800", "btn": "bg-indigo-500 text-white"},
    "brand": {"bg": "bg-indigo-600", "text_h": "text-white", "text_p": "text-indigo-100", "bg_sec": "bg-indigo-500", "btn": "bg-white text-indigo-600"}
}

STYLES = {
    "minimal": {"border": "border border-current opacity-20", "shadow": "shadow-sm", "glass": "", "rounded": "rounded-none", "pad": "p-4"},
    "flat": {"border": "border-0", "shadow": "shadow-xl", "glass": "", "rounded": "rounded-2xl", "pad": "p-8"},
    "glassmorphism": {"border": "border border-white/30", "shadow": "shadow-2xl", "glass": "backdrop-blur-xl bg-white/5", "rounded": "rounded-3xl", "pad": "p-10"}
}

LAYOUTS = {
    "vertical": {"dir": "flex flex-col gap-6"},
    "horizontal": {"dir": "flex flex-row items-center justify-between gap-8"},
    "grid": {"dir": "grid grid-cols-2 gap-8"}
}

SKELETONS = [
    {
        "component": "card",
        "html": """
<article class="{bg} {border} {shadow} {rounded} {glass} {pad} w-full max-w-3xl mx-auto overflow-hidden">
    <div class="{dir}">
        <div class="flex-1">
            <h3 class="{text_h} text-2xl font-bold tracking-tight mb-2">Enterprise Plan</h3>
            <p class="{text_p} text-sm leading-relaxed mb-6">Everything you need to scale your application to millions of users.</p>
            <ul class="space-y-3 mb-8">
                <li class="flex items-center {text_p}"><svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path></svg>Unlimited API Calls</li>
                <li class="flex items-center {text_p}"><svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path></svg>24/7 Dedicated Support</li>
                <li class="flex items-center {text_p}"><svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path></svg>Custom Analytics Dashboard</li>
            </ul>
        </div>
        <div class="{bg_sec} p-6 {rounded} flex flex-col justify-center items-center text-center">
            <span class="{text_h} text-4xl font-extrabold">$299</span>
            <span class="{text_p} text-sm mt-1 mb-4">/month billed annually</span>
            <button class="{btn} w-full py-3 px-4 rounded-lg font-medium transition-transform hover:scale-105">Get Started</button>
        </div>
    </div>
</article>
        """
    },
    {
        "component": "header",
        "html": """
<header class="{bg} {border} {shadow} {rounded} {glass} {pad} w-full mb-8">
    <div class="{dir}">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 {bg_sec} {rounded} flex items-center justify-center">
                <svg class="w-6 h-6 {text_h}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            </div>
            <span class="{text_h} text-xl font-black tracking-widest">NEXUS</span>
        </div>
        <nav class="hidden md:flex gap-6">
            <a href="#" class="{text_p} hover:{text_h} font-medium transition-colors">Products</a>
            <a href="#" class="{text_p} hover:{text_h} font-medium transition-colors">Solutions</a>
            <a href="#" class="{text_p} hover:{text_h} font-medium transition-colors">Pricing</a>
            <a href="#" class="{text_p} hover:{text_h} font-medium transition-colors">Docs</a>
        </nav>
        <div class="flex items-center gap-4">
            <a href="#" class="{text_p} hover:{text_h} font-medium">Log in</a>
            <button class="{btn} px-5 py-2 {rounded} font-semibold shadow-md">Sign up</button>
        </div>
    </div>
</header>
        """
    },
    {
        "component": "form",
        "html": """
<section class="{bg} {border} {shadow} {rounded} {glass} {pad} w-full max-w-4xl mx-auto">
    <div class="mb-8 text-center">
        <h2 class="{text_h} text-3xl font-bold mb-3">Contact Sales</h2>
        <p class="{text_p} max-w-md mx-auto">We're here to help and answer any question you might have. We look forward to hearing from you.</p>
    </div>
    <form class="{dir}">
        <div class="flex flex-col gap-5">
            <div>
                <label class="block {text_h} text-sm font-semibold mb-2">Full Name</label>
                <input type="text" class="w-full {bg_sec} {border} {text_h} {rounded} px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="John Doe">
            </div>
            <div>
                <label class="block {text_h} text-sm font-semibold mb-2">Work Email</label>
                <input type="email" class="w-full {bg_sec} {border} {text_h} {rounded} px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="john@company.com">
            </div>
        </div>
        <div class="flex flex-col gap-5 h-full">
            <div class="flex-1">
                <label class="block {text_h} text-sm font-semibold mb-2">How can we help?</label>
                <textarea rows="4" class="w-full h-[88%] {bg_sec} {border} {text_h} {rounded} px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Tell us about your project..."></textarea>
            </div>
            <button type="submit" class="{btn} w-full py-4 {rounded} font-bold text-lg shadow-lg hover:opacity-90 transition-opacity">Send Message</button>
        </div>
    </form>
</section>
        """
    },
    {
        "component": "footer",
        "html": """
<footer class="{bg} {border} {shadow} {rounded} {glass} {pad} w-full mt-10">
    <div class="{dir}">
        <div class="col-span-1">
            <div class="flex items-center gap-2 mb-4">
                <div class="w-8 h-8 {bg_sec} rounded-full"></div>
                <span class="{text_h} text-lg font-bold">Acme Corp</span>
            </div>
            <p class="{text_p} text-sm mb-6 max-w-xs">Making the world a better place through constructing elegant hierarchies.</p>
            <div class="flex space-x-4">
                <div class="w-6 h-6 {bg_sec} rounded"></div>
                <div class="w-6 h-6 {bg_sec} rounded"></div>
                <div class="w-6 h-6 {bg_sec} rounded"></div>
            </div>
        </div>
        <div class="col-span-1 grid grid-cols-2 gap-4">
            <div>
                <h4 class="{text_h} font-semibold mb-4">Solutions</h4>
                <ul class="space-y-2">
                    <li><a href="#" class="{text_p} hover:{text_h} text-sm">Marketing</a></li>
                    <li><a href="#" class="{text_p} hover:{text_h} text-sm">Analytics</a></li>
                    <li><a href="#" class="{text_p} hover:{text_h} text-sm">Commerce</a></li>
                </ul>
            </div>
            <div>
                <h4 class="{text_h} font-semibold mb-4">Support</h4>
                <ul class="space-y-2">
                    <li><a href="#" class="{text_p} hover:{text_h} text-sm">Pricing</a></li>
                    <li><a href="#" class="{text_p} hover:{text_h} text-sm">Documentation</a></li>
                    <li><a href="#" class="{text_p} hover:{text_h} text-sm">Guides</a></li>
                </ul>
            </div>
        </div>
    </div>
    <div class="mt-8 pt-8 {border} border-t">
        <p class="{text_p} text-center text-sm">© 2026 Acme Corp. All rights reserved.</p>
    </div>
</footer>
        """
    },
    {
        "component": "hero",
        "html": """
<section class="{bg} {border} {shadow} {rounded} {glass} {pad} w-full overflow-hidden relative">
    <div class="absolute -top-24 -right-24 w-96 h-96 {bg_sec} rounded-full opacity-50 blur-3xl"></div>
    <div class="{dir} relative z-10">
        <div class="flex flex-col justify-center">
            <span class="inline-block py-1 px-3 {bg_sec} {text_h} rounded-full text-xs font-semibold mb-4 w-max border {border}">New Feature Release v2.0</span>
            <h1 class="{text_h} text-5xl md:text-6xl font-extrabold tracking-tight mb-6 leading-tight">Data to enrich your <span class="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-600">online business</span></h1>
            <p class="{text_p} text-lg mb-8 max-w-lg">Anim aute id magna aliqua ad ad non deserunt sunt. Qui irure qui lorem cupidatat commodo. Elit sunt amet fugiat veniam occaecat fugiat aliqua.</p>
            <div class="flex flex-wrap gap-4">
                <button class="{btn} px-8 py-4 {rounded} font-bold text-lg shadow-xl hover:shadow-2xl transition-shadow">Get started</button>
                <button class="{bg_sec} {text_h} {border} px-8 py-4 {rounded} font-bold text-lg hover:bg-opacity-80 transition-colors">Live demo</button>
            </div>
        </div>
        <div class="flex items-center justify-center relative">
            <div class="w-full h-80 {bg_sec} {rounded} {border} {shadow} relative overflow-hidden group">
                <div class="absolute inset-0 bg-gradient-to-br from-indigo-500/20 to-purple-500/20"></div>
                <div class="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-{bg} to-transparent"></div>
            </div>
        </div>
    </div>
</section>
        """
    }
]

def generate_react(html_str):
    react_str = html_str.replace('class="', 'className="')
    return f"<LANG_REACT> export default function Component() {{\n  return (\n    {react_str}\n  );\n}}"

def generate_nextjs(html_str):
    react_str = html_str.replace('class="', 'className="')
    return f"<LANG_NEXTJS> 'use client';\nexport default function Component() {{\n  return (\n    {react_str}\n  );\n}}"

def get_dom_depth(soup_el, current_depth=1):
    if not hasattr(soup_el, 'children') or not list(soup_el.children):
        return current_depth
    max_depth = current_depth
    for child in soup_el.children:
        if child.name is not None:
            max_depth = max(max_depth, get_dom_depth(child, current_depth + 1))
    return max_depth

def main():
    print("Iniciando Geração do DATASET ULTRA-LUXO (V8.16) com FÁBRICA DE ESQUELETOS...")
    variants_meta = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        tailwind_cdn = '<script src="https://cdn.tailwindcss.com"></script>'
        
        for skel_idx, skeleton in enumerate(SKELETONS):
            comp_type = skeleton["component"]
            raw_template = skeleton["html"].strip()
            
            for theme_name, theme_vars in THEMES.items():
                for style_name, style_vars in STYLES.items():
                    for layout_name, layout_vars in LAYOUTS.items():
                        
                        # Injeta as variáveis no template
                        format_dict = {**theme_vars, **style_vars, **layout_vars}
                        # O background do esqueleto brand/dark precisa que a página seja escura para contraste?
                        # Para manter limpo, vamos usar um fundo cinza que contraste bem
                        page_bg = "bg-gray-200" if theme_name == "light" else "bg-gray-900"
                        
                        try:
                            variant_html = raw_template.format(**format_dict)
                        except KeyError as e:
                            print(f"Erro de variável faltando: {e}")
                            continue
                            
                        # Calcula os metadados estruturais
                        soup = BeautifulSoup(variant_html, "html.parser")
                        root_el = list(soup.children)[0] if list(soup.children) else soup
                        dom_depth = get_dom_depth(root_el)
                        num_tags = len(soup.find_all(True))
                        
                        base_name = f"{comp_type}_{theme_name}_{style_name}_{layout_name}"
                        
                        render_html = f"<!DOCTYPE html><html><head>{tailwind_cdn}</head><body class='p-10 {page_bg} min-h-screen flex items-center justify-center'>{variant_html}</body></html>"
                        temp_html_path = os.path.join(OUT_DIR, "temp_render.html")
                        with open(temp_html_path, "w", encoding="utf-8") as f_temp:
                            f_temp.write(render_html)
                        
                        page.goto(f"file://{temp_html_path}", wait_until="networkidle")
                        img_filename = f"{base_name}.png"
                        img_path = os.path.join(OUT_IMG, img_filename)
                        
                        try:
                            # Tira a screenshot com padding automático
                            loc = page.locator("body > *").first
                            loc.screenshot(path=img_path)
                        except Exception as e:
                            print(f"Erro na captura {base_name}: {e}")
                            continue
                            
                        # Gera as 4 linguagens
                        fw_html = f"<LANG_HTML> {variant_html}"
                        fw_react = generate_react(variant_html)
                        fw_next = generate_nextjs(variant_html)
                        fw_tail = f"<LANG_TAILWIND> {variant_html}"
                        
                        for fw_name, fw_code in [("html", fw_html), ("react", fw_react), ("nextjs", fw_next), ("tailwind", fw_tail)]:
                            variants_meta.append({
                                "component": comp_type,
                                "theme": theme_name,
                                "style": style_name,
                                "layout": layout_name,
                                "framework": fw_name,
                                "depth_dom": dom_depth,
                                "num_tags": num_tags,
                                "image_file": img_filename,
                                "target_code": fw_code
                            })
                            
            print(f"Esqueleto '{comp_type}' processado. ({len(THEMES)*len(STYLES)*len(LAYOUTS)} variantes x 4 frameworks)")
            
        browser.close()
        
    meta_path = os.path.join(OUT_DIR, "metadata_ultralux.json")
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(variants_meta, f_meta, indent=2)
        
    print(f"\nGeração Concluída! {len(variants_meta)} amostras Ultra-Luxo geradas em {OUT_DIR}.")

if __name__ == "__main__":
    main()
