"""
ROTOLINE - Aplicador de Marca D'agua
Versao 2.0 - Interface moderna, acessivel e com funcionalidades de automacao

Melhorias:
- Interface mais intuitiva para usuarios leigos
- Posicionamento interativo com arrastar e soltar
- Perfis de configuracao (salvar/carregar)
- Modo monitoramento de pasta (watch)
- Suporte a linha de comando (CLI)
- Acessibilidade melhorada
"""

import os
import sys
import json
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageDraw

# ============================================================
# CONSTANTES E CONFIGURACAO
# ============================================================

OUT_SIZE = (1200, 1200)
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
APP_VERSION = "2.0"


def get_base_path():
    """Retorna o diretorio base (funciona com PyInstaller)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    else:
        return Path(os.path.dirname(__file__))


SCRIPT_DIR = get_base_path()
WATERMARK_FILE = SCRIPT_DIR / "rotoline_watermark_only_1200.png"
PROFILES_DIR = SCRIPT_DIR / "profiles"
CONFIG_FILE = SCRIPT_DIR / "config.json"

# Verifica se rembg esta disponivel
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


# ============================================================
# TEMA MODERNO - DESIGN SYSTEM
# ============================================================

COLORS = {
    # Backgrounds (Dark Theme)
    "bg": "#0f172a",
    "bg_card": "#1e293b",
    "bg_dark": "#0c1222",
    "bg_hover": "#334155",

    # Primary (Ciano vibrante)
    "primary": "#06b6d4",
    "primary_hover": "#0891b2",
    "primary_light": "#164e63",
    "primary_dark": "#0e7490",

    # Success (Verde esmeralda)
    "success": "#10b981",
    "success_hover": "#059669",
    "success_light": "#064e3b",

    # Warning (Ambar)
    "warning": "#f59e0b",
    "warning_hover": "#d97706",
    "warning_light": "#78350f",

    # Danger (Vermelho)
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
    "danger_light": "#7f1d1d",

    # Text (claro para dark theme)
    "text": "#f1f5f9",
    "text_secondary": "#cbd5e1",
    "text_muted": "#64748b",
    "text_light": "#94a3b8",
    "border": "#334155",
    "border_focus": "#06b6d4",
    "white": "#ffffff",

    # Special
    "accent": "#a78bfa",
    "accent_light": "#4c1d95",
    "muted": "#475569",
    "muted_hover": "#64748b",
}

FONTS = {
    "title": ("Segoe UI Semibold", 26, "bold"),
    "subtitle": ("Segoe UI", 12),
    "heading": ("Segoe UI Semibold", 14, "bold"),
    "body": ("Segoe UI", 11),
    "small": ("Segoe UI", 10),
    "tiny": ("Segoe UI", 9),
    "mono": ("Cascadia Code", 10),
}


# ============================================================
# PROCESSAMENTO DE IMAGEM
# ============================================================

def boost_watermark_alpha(wm_rgba: Image.Image, factor: float = 1.8) -> Image.Image:
    """Aumenta a opacidade da marca d'agua."""
    wm = wm_rgba.copy().convert("RGBA")
    r, g, b, a = wm.split()
    a = a.point(lambda p: min(255, int(p * factor)))
    wm.putalpha(a)
    return wm


def remove_bg_auto(img_rgba: Image.Image, thr: int) -> Image.Image:
    """Remove fundo usando analise de cantos."""
    arr = np.array(img_rgba).astype(np.int16)
    h, w, _ = arr.shape

    def corner(x0: int, y0: int) -> np.ndarray:
        patch = arr[y0:y0+10, x0:x0+10, :3]
        return np.median(patch.reshape(-1, 3), axis=0)

    bg = np.median(np.stack([
        corner(0, 0),
        corner(w - 10, 0),
        corner(0, h - 10),
        corner(w - 10, h - 10),
    ]), axis=0)

    rgb = arr[:, :, :3]
    dist = np.sqrt(np.sum((rgb - bg) ** 2, axis=2))

    out = arr.copy()
    out[:, :, 3] = np.where(dist <= thr, 0, out[:, :, 3])
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def remove_bg_ai(img_rgba: Image.Image) -> Image.Image:
    """Remove fundo usando IA (rembg)."""
    if not REMBG_AVAILABLE:
        raise RuntimeError("rembg nao esta instalado")
    return rembg_remove(img_rgba)


def crop_to_alpha(img_rgba: Image.Image, pad: int = 2) -> Image.Image:
    """Recorta imagem para o conteudo nao-transparente."""
    arr = np.array(img_rgba)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img_rgba
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(arr.shape[1] - 1, x2 + pad)
    y2 = min(arr.shape[0] - 1, y2 + pad)
    return img_rgba.crop((x1, y1, x2 + 1, y2 + 1))


def build_white_bg_with_wm(wm_rgba: Image.Image, wm_x: int = 0, wm_y: int = 0) -> Image.Image:
    """Cria fundo branco com marca d'agua em posicao especificada."""
    base = Image.new("RGBA", OUT_SIZE, (255, 255, 255, 255))
    base.alpha_composite(wm_rgba, (wm_x, wm_y))
    return base


def process_image(
    img: Image.Image,
    wm_rgba: Image.Image,
    padding: float,
    thr: int,
    remove_bg: bool,
    bg_method: str = "simple",
    pos_x: float = 0.5,  # 0.0 = marca d'agua a esquerda, 0.5 = centro, 1.0 = direita
    pos_y: float = 0.5,  # 0.0 = marca d'agua no topo, 0.5 = centro, 1.0 = baixo
) -> Image.Image:
    """Processa uma imagem com posicionamento da marca d'agua."""
    img = img.convert("RGBA")

    if remove_bg:
        alpha_min = np.array(img)[:, :, 3].min()
        if alpha_min == 255:
            if bg_method == "ai":
                img = remove_bg_ai(img)
            else:
                img = remove_bg_auto(img, thr=thr)
            img = crop_to_alpha(img, pad=2)

    pad_px = int(min(OUT_SIZE) * padding)
    max_w = OUT_SIZE[0] - 2 * pad_px
    max_h = OUT_SIZE[1] - 2 * pad_px

    img = ImageOps.contain(img, (max_w, max_h))

    # Calcula offset da marca d'agua baseado em pos_x e pos_y
    wm_w, wm_h = wm_rgba.size
    max_offset_x = wm_w // 2
    max_offset_y = wm_h // 2

    wm_offset_x = int((pos_x - 0.5) * 2 * max_offset_x)
    wm_offset_y = int((pos_y - 0.5) * 2 * max_offset_y)

    bg = build_white_bg_with_wm(wm_rgba, wm_offset_x, wm_offset_y)

    # Produto sempre centralizado
    available_x = OUT_SIZE[0] - img.size[0] - 2 * pad_px
    available_y = OUT_SIZE[1] - img.size[1] - 2 * pad_px

    x = pad_px + int(available_x * 0.5)
    y = pad_px + int(available_y * 0.5)

    bg.alpha_composite(img, (x, y))
    return bg


def process_one(
    in_path: Path,
    out_path: Path,
    wm_rgba: Image.Image,
    padding: float,
    thr: int,
    remove_bg: bool,
    out_format: str = "jpg",
    bg_method: str = "simple",
    pos_x: float = 0.5,
    pos_y: float = 0.5,
    quality: int = 95,
):
    """Processa e salva uma imagem."""
    img = Image.open(in_path).convert("RGBA")
    result = process_image(img, wm_rgba, padding, thr, remove_bg, bg_method, pos_x, pos_y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_format.lower() in ("jpg", "jpeg"):
        result.convert("RGB").save(out_path, "JPEG", quality=quality, optimize=True)
    else:
        result.save(out_path, "PNG", compress_level=6)


def iter_images(folder: Path):
    """Itera sobre imagens em uma pasta (recursivo)."""
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


# ============================================================
# COMPONENTES DE UI MODERNOS
# ============================================================

class Tooltip:
    """Tooltip com visual moderno."""

    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip = None
        self.scheduled = None
        widget.bind("<Enter>", self._schedule, add='+')
        widget.bind("<Leave>", self._hide, add='+')
        widget.bind("<Button-1>", self._hide, add='+')

    def _schedule(self, event=None):
        self._hide()
        self.scheduled = self.widget.after(self.delay, self._show)

    def _show(self):
        if not self.widget.winfo_exists():
            return

        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        self.tooltip.attributes("-topmost", True)

        frame = tk.Frame(
            self.tooltip,
            bg=COLORS["bg_hover"],
            padx=1,
            pady=1
        )
        frame.pack()

        label = tk.Label(
            frame,
            text=self.text,
            bg=COLORS["bg_hover"],
            fg=COLORS["text"],
            font=FONTS["small"],
            padx=10,
            pady=6,
            wraplength=300,
            justify="left"
        )
        label.pack()

    def _hide(self, event=None):
        if self.scheduled:
            self.widget.after_cancel(self.scheduled)
            self.scheduled = None
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class ModernButton(tk.Frame):
    """Botao moderno com estados visuais."""

    def __init__(self, parent, text, command=None, style="primary", icon=None,
                 size="normal", full_width=False, tooltip=None, **kwargs):
        bg_color = self._get_parent_bg(parent)
        super().__init__(parent, bg=bg_color)

        self.style = style
        self.command = command
        self._enabled = True
        self._pressed = False
        self._original_text = f"{icon}  {text}" if icon else text

        # Cores por estilo
        styles = {
            "primary": (COLORS["primary"], COLORS["primary_hover"], COLORS["white"]),
            "success": (COLORS["success"], COLORS["success_hover"], COLORS["white"]),
            "warning": (COLORS["warning"], COLORS["warning_hover"], COLORS["white"]),
            "danger": (COLORS["danger"], COLORS["danger_hover"], COLORS["white"]),
            "secondary": (COLORS["bg_dark"], COLORS["bg_hover"], COLORS["text"]),
            "outline": (COLORS["bg_card"], COLORS["bg_hover"], COLORS["text"]),
        }

        self.color_normal, self.color_hover, self.fg_color = styles.get(style, styles["primary"])

        # Tamanhos
        sizes = {
            "small": (12, 6, FONTS["small"]),
            "normal": (20, 10, FONTS["body"]),
            "large": (28, 14, ("Segoe UI Semibold", 12, "bold")),
        }
        padx, pady, font = sizes.get(size, sizes["normal"])

        self.btn = tk.Label(
            self,
            text=self._original_text,
            bg=self.color_normal,
            fg=self.fg_color,
            font=font,
            padx=padx,
            pady=pady,
            cursor="hand2"
        )

        if full_width:
            self.btn.pack(fill="x")
        else:
            self.btn.pack()

        self.btn.bind("<Enter>", self._on_enter)
        self.btn.bind("<Leave>", self._on_leave)
        self.btn.bind("<Button-1>", self._on_press)
        self.btn.bind("<ButtonRelease-1>", self._on_release)

        if tooltip:
            Tooltip(self.btn, tooltip)

    def _get_parent_bg(self, parent):
        try:
            return parent["bg"]
        except:
            return COLORS["bg"]

    def _on_enter(self, e):
        if self._enabled:
            self.btn.config(bg=self.color_hover)

    def _on_leave(self, e):
        if self._enabled:
            self.btn.config(bg=self.color_normal)

    def _on_press(self, e):
        if self._enabled:
            self._pressed = True
            self.btn.config(bg=self.color_hover)

    def _on_release(self, e):
        if self._enabled and self._pressed and self.command:
            self.command()
        self._pressed = False

    def config(self, **kwargs):
        if "state" in kwargs:
            self._enabled = kwargs["state"] != "disabled"
            if self._enabled:
                self.btn.config(bg=self.color_normal, cursor="hand2", fg=self.fg_color)
            else:
                self.btn.config(bg=COLORS["muted"], cursor="arrow", fg=COLORS["white"])
        if "text" in kwargs:
            self.btn.config(text=kwargs["text"])

    def set_loading(self, loading=True, text="Processando..."):
        if loading:
            self._enabled = False
            self.btn.config(text=f"   {text}", bg=COLORS["muted"], cursor="wait")
        else:
            self._enabled = True
            self.btn.config(text=self._original_text, bg=self.color_normal, cursor="hand2")


class Card(tk.Frame):
    """Card container com visual moderno."""

    def __init__(self, parent, title=None, subtitle=None, collapsible=False, **kwargs):
        super().__init__(parent, bg=COLORS["bg_card"], padx=20, pady=16,
                        highlightthickness=1, highlightbackground=COLORS["border"], **kwargs)

        self.collapsed = False
        self.content_frame = None

        if title:
            header = tk.Frame(self, bg=COLORS["bg_card"])
            header.pack(fill="x", pady=(0, 12))

            title_frame = tk.Frame(header, bg=COLORS["bg_card"])
            title_frame.pack(side="left", fill="x", expand=True)

            tk.Label(
                title_frame,
                text=title,
                font=FONTS["heading"],
                bg=COLORS["bg_card"],
                fg=COLORS["text"],
                anchor="w"
            ).pack(side="left")

            if subtitle:
                tk.Label(
                    title_frame,
                    text=f"  |  {subtitle}",
                    font=FONTS["small"],
                    bg=COLORS["bg_card"],
                    fg=COLORS["text_muted"],
                    anchor="w"
                ).pack(side="left")


class ModernEntry(tk.Frame):
    """Campo de entrada moderno com label."""

    def __init__(self, parent, label=None, placeholder="", variable=None,
                 readonly=False, browse_command=None, tooltip=None):
        super().__init__(parent, bg=COLORS["bg_card"])

        if label:
            label_frame = tk.Frame(self, bg=COLORS["bg_card"])
            label_frame.pack(fill="x", pady=(0, 4))

            lbl = tk.Label(
                label_frame,
                text=label,
                font=FONTS["small"],
                bg=COLORS["bg_card"],
                fg=COLORS["text_secondary"]
            )
            lbl.pack(side="left")

            if tooltip:
                help_lbl = tk.Label(
                    label_frame,
                    text=" ?",
                    font=FONTS["tiny"],
                    bg=COLORS["bg_card"],
                    fg=COLORS["primary"],
                    cursor="question_arrow"
                )
                help_lbl.pack(side="left", padx=(4, 0))
                Tooltip(help_lbl, tooltip)

        entry_frame = tk.Frame(self, bg=COLORS["border"])
        entry_frame.pack(fill="x")

        inner = tk.Frame(entry_frame, bg=COLORS["bg_dark"], padx=1, pady=1)
        inner.pack(fill="x", padx=1, pady=1)

        self.entry = tk.Entry(
            inner,
            textvariable=variable,
            font=FONTS["body"],
            relief="flat",
            bg=COLORS["bg_dark"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            state="readonly" if readonly else "normal",
            readonlybackground=COLORS["bg_dark"]
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=8, padx=8)

        if browse_command:
            browse_btn = tk.Label(
                inner,
                text=" Procurar ",
                font=FONTS["small"],
                bg=COLORS["primary"],
                fg=COLORS["white"],
                cursor="hand2",
                padx=10,
                pady=4
            )
            browse_btn.pack(side="right", padx=4, pady=4)
            browse_btn.bind("<Button-1>", lambda e: browse_command())
            browse_btn.bind("<Enter>", lambda e: browse_btn.config(bg=COLORS["primary_hover"]))
            browse_btn.bind("<Leave>", lambda e: browse_btn.config(bg=COLORS["primary"]))


class ModernSlider(tk.Frame):
    """Slider moderno com valor exibido."""

    def __init__(self, parent, label, variable, from_, to_, format_func=None,
                 tooltip=None, show_markers=False):
        super().__init__(parent, bg=COLORS["bg_card"])

        self.format_func = format_func or (lambda x: str(x))

        # Header
        header = tk.Frame(self, bg=COLORS["bg_card"])
        header.pack(fill="x")

        label_frame = tk.Frame(header, bg=COLORS["bg_card"])
        label_frame.pack(side="left")

        lbl = tk.Label(
            label_frame,
            text=label,
            font=FONTS["small"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_secondary"]
        )
        lbl.pack(side="left")

        if tooltip:
            help_lbl = tk.Label(
                label_frame,
                text=" ?",
                font=FONTS["tiny"],
                bg=COLORS["bg_card"],
                fg=COLORS["primary"],
                cursor="arrow"
            )
            help_lbl.pack(side="left", padx=(4, 0))
            Tooltip(help_lbl, tooltip)

        self.value_label = tk.Label(
            header,
            text=self.format_func(variable.get()),
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["bg_card"],
            fg=COLORS["primary"]
        )
        self.value_label.pack(side="right")

        # Slider
        style = ttk.Style()
        style.configure("Modern.Horizontal.TScale",
                       background=COLORS["bg_card"],
                       troughcolor=COLORS["bg_dark"])

        self.slider = ttk.Scale(
            self,
            from_=from_,
            to=to_,
            variable=variable,
            orient="horizontal",
            style="Modern.Horizontal.TScale"
        )
        self.slider.pack(fill="x", pady=(6, 0))

        variable.trace_add("write", self._update_value)

    def _update_value(self, *args):
        try:
            val = self.slider.get()
            self.value_label.config(text=self.format_func(val))
        except:
            pass


class ModernCheckbox(tk.Frame):
    """Checkbox moderno com descricao."""

    def __init__(self, parent, text, variable, description=None, tooltip=None):
        super().__init__(parent, bg=COLORS["bg_card"])

        self.variable = variable

        main_frame = tk.Frame(self, bg=COLORS["bg_card"])
        main_frame.pack(fill="x")

        self.checkbox = tk.Checkbutton(
            main_frame,
            text=text,
            variable=variable,
            font=FONTS["body"],
            bg=COLORS["bg_card"],
            fg=COLORS["text"],
            activebackground=COLORS["bg_card"],
            selectcolor=COLORS["bg_dark"],
            activeforeground=COLORS["text"],
            cursor="hand2"
        )
        self.checkbox.pack(side="left")

        if tooltip:
            help_lbl = tk.Label(
                main_frame,
                text=" ?",
                font=FONTS["tiny"],
                bg=COLORS["bg_card"],
                fg=COLORS["primary"],
                cursor="arrow"
            )
            help_lbl.pack(side="left", padx=(4, 0))
            Tooltip(help_lbl, tooltip)

        if description:
            tk.Label(
                self,
                text=description,
                font=FONTS["tiny"],
                bg=COLORS["bg_card"],
                fg=COLORS["text_muted"],
                anchor="w"
            ).pack(fill="x", padx=(24, 0))


class InteractivePositionSelector(tk.Frame):
    """Seletor de posicao interativo com arrastar e soltar."""

    def __init__(self, parent, pos_x_var, pos_y_var, preview_callback=None):
        super().__init__(parent, bg=COLORS["bg_card"])

        self.pos_x_var = pos_x_var
        self.pos_y_var = pos_y_var
        self.preview_callback = preview_callback
        self.dragging = False

        # Titulo
        header = tk.Frame(self, bg=COLORS["bg_card"])
        header.pack(fill="x", pady=(0, 8))

        tk.Label(
            header,
            text="Posicao da Marca D'agua",
            font=FONTS["heading"],
            bg=COLORS["bg_card"],
            fg=COLORS["text"]
        ).pack(side="left")

        # Botao reset
        reset_btn = tk.Label(
            header,
            text="[ Centralizar ]",
            font=FONTS["tiny"],
            bg=COLORS["bg_card"],
            fg=COLORS["primary"],
            cursor="hand2"
        )
        reset_btn.pack(side="right")
        reset_btn.bind("<Button-1>", self._reset_position)
        reset_btn.bind("<Enter>", lambda e: reset_btn.config(fg=COLORS["primary_hover"]))
        reset_btn.bind("<Leave>", lambda e: reset_btn.config(fg=COLORS["primary"]))

        tk.Label(
            self,
            text="Arraste o ponto ou clique para posicionar a marca d'agua",
            font=FONTS["tiny"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_muted"]
        ).pack(anchor="w", pady=(0, 8))

        # Canvas interativo
        self.canvas_size = 180
        self.canvas = tk.Canvas(
            self,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=COLORS["bg_dark"],
            highlightthickness=2,
            highlightbackground=COLORS["border"],
            cursor="crosshair"
        )
        self.canvas.pack()

        # Bind eventos
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Coordenadas (criar ANTES de _draw_position)
        self.coords_label = tk.Label(
            self,
            text="Centro (50%, 50%)",
            font=FONTS["tiny"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_secondary"]
        )

        # Grid de referencia
        self._draw_grid()
        self._draw_position()

        # Posicionar label de coordenadas
        self.coords_label.pack(pady=(8, 0))

        # Quick positions
        quick_frame = tk.Frame(self, bg=COLORS["bg_card"])
        quick_frame.pack(fill="x", pady=(8, 0))

        tk.Label(
            quick_frame,
            text="Posicoes rapidas:",
            font=FONTS["tiny"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_muted"]
        ).pack(side="left")

        positions = [
            ("Esq", 0.1, 0.5), ("Centro", 0.5, 0.5), ("Dir", 0.9, 0.5),
            ("Topo", 0.5, 0.1), ("Baixo", 0.5, 0.9),
        ]

        for label_text, px, py in positions:
            btn = tk.Label(
                quick_frame,
                text=label_text,
                font=FONTS["tiny"],
                bg=COLORS["bg_dark"],
                fg=COLORS["text"],
                padx=6,
                pady=2,
                cursor="hand2"
            )
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", lambda e, x=px, y=py: self._set_position(x, y))
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=COLORS["primary_light"]))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=COLORS["bg_dark"]))

    def _draw_grid(self):
        """Desenha grid de referencia."""
        s = self.canvas_size
        # Linhas de grade
        for i in range(1, 3):
            x = s * i // 3
            self.canvas.create_line(x, 0, x, s, fill=COLORS["border"], dash=(2, 2))
        for i in range(1, 3):
            y = s * i // 3
            self.canvas.create_line(0, y, s, y, fill=COLORS["border"], dash=(2, 2))

        # Borda interna (area util)
        margin = 15
        self.canvas.create_rectangle(
            margin, margin, s - margin, s - margin,
            outline=COLORS["text_muted"], dash=(4, 4)
        )

    def _draw_position(self):
        """Desenha indicador de posicao."""
        self.canvas.delete("position")

        s = self.canvas_size
        margin = 15

        # Converter posicao para coordenadas do canvas
        x = margin + (s - 2 * margin) * self.pos_x_var.get()
        y = margin + (s - 2 * margin) * self.pos_y_var.get()

        # Circulo externo
        r = 12
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill=COLORS["primary_light"],
            outline=COLORS["primary"],
            width=2,
            tags="position"
        )

        # Circulo interno
        r2 = 5
        self.canvas.create_oval(
            x - r2, y - r2, x + r2, y + r2,
            fill=COLORS["primary"],
            outline="",
            tags="position"
        )

        # Atualizar label
        px = int(self.pos_x_var.get() * 100)
        py = int(self.pos_y_var.get() * 100)

        if px == 50 and py == 50:
            text = "Marca centralizada (50%, 50%)"
        else:
            h = "Esquerda" if px < 40 else ("Direita" if px > 60 else "Centro")
            v = "Topo" if py < 40 else ("Baixo" if py > 60 else "Centro")
            text = f"Marca: {v} {h} ({px}%, {py}%)"

        self.coords_label.config(text=text)

    def _canvas_to_position(self, cx, cy):
        """Converte coordenadas do canvas para posicao (0-1)."""
        s = self.canvas_size
        margin = 15

        px = (cx - margin) / (s - 2 * margin)
        py = (cy - margin) / (s - 2 * margin)

        return max(0, min(1, px)), max(0, min(1, py))

    def _on_click(self, event):
        px, py = self._canvas_to_position(event.x, event.y)
        self.pos_x_var.set(px)
        self.pos_y_var.set(py)
        self._draw_position()
        self.dragging = True
        if self.preview_callback:
            self.preview_callback()

    def _on_drag(self, event):
        if self.dragging:
            px, py = self._canvas_to_position(event.x, event.y)
            self.pos_x_var.set(px)
            self.pos_y_var.set(py)
            self._draw_position()

    def _on_release(self, event):
        self.dragging = False
        if self.preview_callback:
            self.preview_callback()

    def _set_position(self, px, py):
        self.pos_x_var.set(px)
        self.pos_y_var.set(py)
        self._draw_position()
        if self.preview_callback:
            self.preview_callback()

    def _reset_position(self, event=None):
        self._set_position(0.5, 0.5)

    def update_display(self):
        """Atualiza display (chamar apos mudar variaveis externamente)."""
        self._draw_position()


class ProfileManager:
    """Gerenciador de perfis de configuracao."""

    def __init__(self, profiles_dir: Path):
        self.profiles_dir = profiles_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self):
        """Lista perfis disponiveis."""
        return [p.stem for p in self.profiles_dir.glob("*.json")]

    def save_profile(self, name: str, settings: dict):
        """Salva um perfil."""
        path = self.profiles_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

    def load_profile(self, name: str) -> dict:
        """Carrega um perfil."""
        path = self.profiles_dir / f"{name}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def delete_profile(self, name: str):
        """Remove um perfil."""
        path = self.profiles_dir / f"{name}.json"
        if path.exists():
            path.unlink()


# ============================================================
# APLICACAO PRINCIPAL
# ============================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(f"ROTOLINE - Marca D'agua v{APP_VERSION}")
        self.geometry("980x920")
        self.minsize(920, 870)
        self.configure(bg=COLORS["bg"])

        # Gerenciador de perfis
        self.profile_manager = ProfileManager(PROFILES_DIR)

        # Variaveis de configuracao
        self._init_variables()

        # Construir UI
        self._build_ui()

        # Verificar marca d'agua
        self._check_watermark()

        # Watch mode
        self.watching = False
        self.watch_thread = None

    def _init_variables(self):
        """Inicializa variaveis de controle."""
        # Pastas
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.single_file = tk.StringVar()

        # Imagens temporarias
        self.single_original_img = None
        self.single_result_img = None
        self.single_preview_original = None
        self.single_preview_result = None

        # Configuracoes compartilhadas
        self.remove_bg = tk.BooleanVar(value=True)
        self.padding = tk.DoubleVar(value=0.10)
        self.thr = tk.IntVar(value=20)
        self.wm_strength = tk.DoubleVar(value=1.8)
        self.bg_method = tk.StringVar(value="simple")
        self.pos_x = tk.DoubleVar(value=0.5)  # 0.0 a 1.0
        self.pos_y = tk.DoubleVar(value=0.5)
        self.out_format = tk.StringVar(value="jpg")
        self.quality = tk.IntVar(value=95)
        self.auto_preview = tk.BooleanVar(value=True)

        # Para modo unico (copia separada)
        self.single_remove_bg = tk.BooleanVar(value=True)
        self.single_padding = tk.DoubleVar(value=0.10)
        self.single_thr = tk.IntVar(value=20)
        self.single_wm_strength = tk.DoubleVar(value=1.8)
        self.single_bg_method = tk.StringVar(value="simple")
        self.single_pos_x = tk.DoubleVar(value=0.5)
        self.single_pos_y = tk.DoubleVar(value=0.5)

    def _check_watermark(self):
        """Verifica se o arquivo de marca d'agua existe."""
        if not WATERMARK_FILE.exists():
            self.after(100, lambda: messagebox.showwarning(
                "Marca d'agua nao encontrada",
                f"O arquivo de marca d'agua nao foi encontrado.\n\n"
                f"Verifique se o arquivo 'rotoline_watermark_only_1200.png' "
                f"esta na mesma pasta do programa.\n\n"
                f"Local esperado:\n{WATERMARK_FILE}"
            ))

    def _build_ui(self):
        """Constroi a interface principal."""
        # Header
        self._build_header()

        # Notebook (Abas)
        self._build_notebook()

    def _build_header(self):
        """Constroi o cabecalho."""
        header = tk.Frame(self, bg=COLORS["bg"])
        header.pack(fill="x", pady=(20, 10), padx=30)

        # Logo e titulo
        title_frame = tk.Frame(header, bg=COLORS["bg"])
        title_frame.pack()

        tk.Label(
            title_frame,
            text="ROTOLINE",
            font=FONTS["title"],
            bg=COLORS["bg"],
            fg=COLORS["primary"]
        ).pack()

        tk.Label(
            title_frame,
            text="Aplicador de Marca D'agua Profissional",
            font=FONTS["subtitle"],
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"]
        ).pack(pady=(2, 0))

        # Info bar
        info_frame = tk.Frame(header, bg=COLORS["bg"])
        info_frame.pack(fill="x", pady=(10, 0))

        # Status do rembg
        if REMBG_AVAILABLE:
            status_text = "[OK] Remocao por IA disponivel"
            status_color = COLORS["success"]
        else:
            status_text = "[!] Remocao basica (instale rembg para IA)"
            status_color = COLORS["text_muted"]

        tk.Label(
            info_frame,
            text=status_text,
            font=FONTS["tiny"],
            bg=COLORS["bg"],
            fg=status_color
        ).pack(side="left")

        tk.Label(
            info_frame,
            text=f"v{APP_VERSION}",
            font=FONTS["tiny"],
            bg=COLORS["bg"],
            fg=COLORS["text_muted"]
        ).pack(side="right")

    def _build_notebook(self):
        """Constroi o notebook com abas."""
        style = ttk.Style()
        style.theme_use('default')

        style.configure('Modern.TNotebook',
                       background=COLORS["bg"],
                       borderwidth=0,
                       tabmargins=[0, 0, 0, 0])

        style.configure('Modern.TNotebook.Tab',
                       font=FONTS["body"],
                       padding=[24, 12],
                       background=COLORS["bg_dark"],
                       foreground=COLORS["text"])

        style.map('Modern.TNotebook.Tab',
                 background=[('selected', COLORS["bg_card"])],
                 foreground=[('selected', COLORS["primary"])])

        # Estilo dark para Combobox
        style.configure('TCombobox',
                       fieldbackground=COLORS["bg_dark"],
                       background=COLORS["bg_dark"],
                       foreground=COLORS["text"],
                       arrowcolor=COLORS["text"])

        style.map('TCombobox',
                 fieldbackground=[('readonly', COLORS["bg_dark"])],
                 selectbackground=[('readonly', COLORS["primary_light"])],
                 selectforeground=[('readonly', COLORS["text"])])

        self.notebook = ttk.Notebook(self, style='Modern.TNotebook')
        self.notebook.pack(fill="both", expand=True, padx=30, pady=10)

        # Criar abas
        self.tab_single = tk.Frame(self.notebook, bg=COLORS["bg"])
        self.tab_batch = tk.Frame(self.notebook, bg=COLORS["bg"])
        self.tab_watch = tk.Frame(self.notebook, bg=COLORS["bg"])
        self.tab_settings = tk.Frame(self.notebook, bg=COLORS["bg"])

        self.notebook.add(self.tab_single, text="  Uma Foto  ")
        self.notebook.add(self.tab_batch, text="  Varias Fotos  ")
        self.notebook.add(self.tab_watch, text="  Monitorar Pasta  ")
        self.notebook.add(self.tab_settings, text="  Configuracoes  ")

        # Construir conteudo das abas
        self._build_single_tab()
        self._build_batch_tab()
        self._build_watch_tab()
        self._build_settings_tab()

    def _build_single_tab(self):
        """Aba para processar uma unica foto."""
        main = tk.Frame(self.tab_single, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Layout em duas colunas
        left_col = tk.Frame(main, bg=COLORS["bg"])
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_col = tk.Frame(main, bg=COLORS["bg"])
        right_col.pack(side="right", fill="y")

        # === COLUNA ESQUERDA ===

        # Card: Selecionar imagem
        card1 = Card(left_col, title="1. Escolha a Foto",
                    subtitle="Formatos: JPG, PNG, WEBP, BMP, TIFF")
        card1.pack(fill="x", pady=(0, 10))

        ModernEntry(
            card1,
            label="Arquivo de imagem",
            variable=self.single_file,
            readonly=True,
            browse_command=self._pick_single_file,
            tooltip="Clique em Procurar para escolher uma imagem do seu computador"
        ).pack(fill="x")

        # Card: Preview
        card2 = Card(left_col, title="2. Visualizacao", subtitle="Antes e depois")
        card2.pack(fill="both", expand=True, pady=(0, 10))

        preview_frame = tk.Frame(card2, bg=COLORS["bg_card"])
        preview_frame.pack(fill="both", expand=True)

        # Original
        orig_frame = tk.Frame(preview_frame, bg=COLORS["bg_card"])
        orig_frame.pack(side="left", fill="both", expand=True)

        tk.Label(
            orig_frame,
            text="ORIGINAL",
            font=FONTS["tiny"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_muted"]
        ).pack()

        self.canvas_original = tk.Canvas(
            orig_frame,
            width=280,
            height=280,
            bg=COLORS["bg_dark"],
            highlightthickness=2,
            highlightbackground=COLORS["border"]
        )
        self.canvas_original.pack(pady=5)

        # Seta
        tk.Label(
            preview_frame,
            text=">>>",
            font=("Segoe UI", 20, "bold"),
            bg=COLORS["bg_card"],
            fg=COLORS["primary"]
        ).pack(side="left", padx=15)

        # Resultado
        result_frame = tk.Frame(preview_frame, bg=COLORS["bg_card"])
        result_frame.pack(side="left", fill="both", expand=True)

        tk.Label(
            result_frame,
            text="RESULTADO",
            font=FONTS["tiny"],
            bg=COLORS["bg_card"],
            fg=COLORS["success"]
        ).pack()

        self.canvas_result = tk.Canvas(
            result_frame,
            width=280,
            height=280,
            bg=COLORS["bg_dark"],
            highlightthickness=2,
            highlightbackground=COLORS["border"]
        )
        self.canvas_result.pack(pady=5)

        # Botoes
        btn_frame = tk.Frame(left_col, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(0, 10))

        self.btn_single_process = ModernButton(
            btn_frame,
            text="PROCESSAR",
            icon="▶",
            command=self._process_single,
            style="warning",
            size="large",
            tooltip="Aplica a marca d'agua na foto"
        )
        self.btn_single_process.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_single_save = ModernButton(
            btn_frame,
            text="SALVAR",
            icon="💾",
            command=self._save_single,
            style="success",
            size="large",
            tooltip="Salva a foto processada"
        )
        self.btn_single_save.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.btn_single_save.config(state="disabled")

        # Status
        self.single_status = tk.Label(
            left_col,
            text="Escolha uma foto para comecar",
            font=FONTS["small"],
            bg=COLORS["bg"],
            fg=COLORS["text_secondary"]
        )
        self.single_status.pack(fill="x")

        # === COLUNA DIREITA (Configuracoes) ===

        # Card: Ajustes
        card3 = Card(right_col, title="3. Ajustes")
        card3.pack(fill="x", pady=(0, 10))

        # Remover fundo
        ModernCheckbox(
            card3,
            "Remover fundo",
            self.single_remove_bg,
            tooltip="Remove automaticamente o fundo da imagem para destacar o produto"
        ).pack(fill="x", pady=(0, 8))

        # Metodo
        method_frame = tk.Frame(card3, bg=COLORS["bg_card"])
        method_frame.pack(fill="x", pady=(0, 8))

        tk.Label(
            method_frame,
            text="Metodo de remocao:",
            font=FONTS["small"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_secondary"]
        ).pack(side="left")

        methods = ["Simples (rapido)"]
        if REMBG_AVAILABLE:
            methods.append("IA (melhor)")

        self.single_method_combo = ttk.Combobox(
            method_frame,
            values=methods,
            state="readonly",
            width=15,
            font=FONTS["small"]
        )
        self.single_method_combo.current(0)
        self.single_method_combo.pack(side="right")

        # Sliders
        ModernSlider(
            card3,
            "Margem",
            self.single_padding,
            0.05, 0.18,
            lambda v: f"{int(float(v)*100)}%",
            tooltip="Espaco entre o produto e as bordas da imagem"
        ).pack(fill="x", pady=4)

        ModernSlider(
            card3,
            "Intensidade da marca",
            self.single_wm_strength,
            1.0, 3.0,
            lambda v: f"{float(v):.1f}x",
            tooltip="Quao visivel sera a marca d'agua (maior = mais visivel)"
        ).pack(fill="x", pady=4)

        # Posicao interativa
        self.single_position_selector = InteractivePositionSelector(
            card3,
            self.single_pos_x,
            self.single_pos_y,
            preview_callback=self._auto_preview_single
        )
        self.single_position_selector.pack(fill="x", pady=(10, 0))

    def _build_batch_tab(self):
        """Aba para processar varias fotos."""
        main = tk.Frame(self.tab_batch, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Layout em duas colunas
        left_col = tk.Frame(main, bg=COLORS["bg"])
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_col = tk.Frame(main, bg=COLORS["bg"])
        right_col.pack(side="right", fill="y")

        # === COLUNA ESQUERDA ===

        # Card: Pastas
        card1 = Card(left_col, title="1. Escolha as Pastas")
        card1.pack(fill="x", pady=(0, 10))

        ModernEntry(
            card1,
            label="Pasta de origem (fotos originais)",
            variable=self.in_dir,
            browse_command=self._pick_in_dir,
            tooltip="Pasta onde estao as fotos que voce quer processar"
        ).pack(fill="x", pady=(0, 10))

        ModernEntry(
            card1,
            label="Pasta de destino (fotos processadas)",
            variable=self.out_dir,
            browse_command=self._pick_out_dir,
            tooltip="Pasta onde as fotos com marca d'agua serao salvas"
        ).pack(fill="x")

        # Card: Progresso
        card2 = Card(left_col, title="2. Progresso")
        card2.pack(fill="x", pady=(0, 10))

        self.batch_status = tk.Label(
            card2,
            text="Aguardando...",
            font=FONTS["body"],
            bg=COLORS["bg_card"],
            fg=COLORS["text"]
        )
        self.batch_status.pack(anchor="w")

        style = ttk.Style()
        style.configure(
            "Modern.Horizontal.TProgressbar",
            thickness=24,
            troughcolor=COLORS["bg_dark"],
            background=COLORS["success"]
        )

        self.progress = ttk.Progressbar(
            card2,
            mode="determinate",
            style="Modern.Horizontal.TProgressbar"
        )
        self.progress.pack(fill="x", pady=(10, 0))

        # Botoes
        btn_frame = tk.Frame(left_col, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(0, 10))

        self.btn_batch_run = ModernButton(
            btn_frame,
            text="PROCESSAR TODAS",
            icon="▶",
            command=self._run_batch,
            style="success",
            size="large",
            tooltip="Processa todas as fotos da pasta de origem"
        )
        self.btn_batch_run.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.btn_open_folder = ModernButton(
            btn_frame,
            text="ABRIR PASTA",
            icon="📂",
            command=self._open_out_dir,
            style="primary",
            size="large",
            tooltip="Abre a pasta de destino no explorador"
        )
        self.btn_open_folder.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Log
        card3 = Card(left_col, title="Historico")
        card3.pack(fill="both", expand=True)

        log_frame = tk.Frame(card3, bg=COLORS["bg_dark"])
        log_frame.pack(fill="both", expand=True)

        self.log = tk.Text(
            log_frame,
            height=8,
            wrap="word",
            font=FONTS["mono"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text"],
            relief="flat",
            padx=10,
            pady=10
        )
        self.log.pack(fill="both", expand=True)

        self.log.tag_configure("success", foreground=COLORS["success"])
        self.log.tag_configure("error", foreground=COLORS["danger"])
        self.log.tag_configure("info", foreground=COLORS["primary"])

        # === COLUNA DIREITA (Configuracoes) ===

        card4 = Card(right_col, title="Configuracoes")
        card4.pack(fill="x", pady=(0, 10))

        ModernCheckbox(
            card4,
            "Remover fundo",
            self.remove_bg,
            tooltip="Remove automaticamente o fundo das imagens"
        ).pack(fill="x", pady=(0, 8))

        # Metodo
        method_frame = tk.Frame(card4, bg=COLORS["bg_card"])
        method_frame.pack(fill="x", pady=(0, 8))

        tk.Label(
            method_frame,
            text="Metodo:",
            font=FONTS["small"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_secondary"]
        ).pack(side="left")

        methods = ["Simples"]
        if REMBG_AVAILABLE:
            methods.append("IA")

        self.batch_method_combo = ttk.Combobox(
            method_frame,
            values=methods,
            state="readonly",
            width=10,
            font=FONTS["small"]
        )
        self.batch_method_combo.current(0)
        self.batch_method_combo.pack(side="right")

        ModernSlider(
            card4,
            "Margem",
            self.padding,
            0.05, 0.18,
            lambda v: f"{int(float(v)*100)}%"
        ).pack(fill="x", pady=4)

        ModernSlider(
            card4,
            "Intensidade",
            self.wm_strength,
            1.0, 3.0,
            lambda v: f"{float(v):.1f}x"
        ).pack(fill="x", pady=4)

        # Posicao
        self.batch_position_selector = InteractivePositionSelector(
            card4,
            self.pos_x,
            self.pos_y
        )
        self.batch_position_selector.pack(fill="x", pady=(10, 0))

    def _build_watch_tab(self):
        """Aba para monitoramento de pasta."""
        main = tk.Frame(self.tab_watch, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Explicacao
        info_card = Card(main, title="Modo Monitoramento",
                        subtitle="Processa fotos automaticamente")
        info_card.pack(fill="x", pady=(0, 10))

        tk.Label(
            info_card,
            text="Este modo monitora uma pasta e processa automaticamente\n"
                 "novas imagens assim que elas aparecerem.\n\n"
                 "Ideal para integracao com cameras ou outros softwares.",
            font=FONTS["body"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_secondary"],
            justify="left"
        ).pack(anchor="w")

        # Configuracao
        config_card = Card(main, title="Configuracao")
        config_card.pack(fill="x", pady=(0, 10))

        self.watch_in_dir = tk.StringVar()
        self.watch_out_dir = tk.StringVar()

        ModernEntry(
            config_card,
            label="Pasta a monitorar",
            variable=self.watch_in_dir,
            browse_command=lambda: self._pick_dir(self.watch_in_dir),
            tooltip="Pasta onde novas imagens aparecerao"
        ).pack(fill="x", pady=(0, 10))

        ModernEntry(
            config_card,
            label="Pasta de saida",
            variable=self.watch_out_dir,
            browse_command=lambda: self._pick_dir(self.watch_out_dir),
            tooltip="Pasta onde as imagens processadas serao salvas"
        ).pack(fill="x")

        # Opcoes
        options_frame = tk.Frame(config_card, bg=COLORS["bg_card"])
        options_frame.pack(fill="x", pady=(10, 0))

        self.watch_delete_original = tk.BooleanVar(value=False)
        ModernCheckbox(
            options_frame,
            "Mover originais para lixeira apos processar",
            self.watch_delete_original,
            tooltip="Remove as imagens originais apos o processamento"
        ).pack(fill="x")

        # Botao
        btn_frame = tk.Frame(main, bg=COLORS["bg"])
        btn_frame.pack(fill="x", pady=(0, 10))

        self.btn_watch = ModernButton(
            btn_frame,
            text="INICIAR MONITORAMENTO",
            icon="▶",
            command=self._toggle_watch,
            style="primary",
            size="large"
        )
        self.btn_watch.pack(fill="x")

        # Status
        status_card = Card(main, title="Status")
        status_card.pack(fill="both", expand=True)

        self.watch_status = tk.Label(
            status_card,
            text="[PARADO]",
            font=FONTS["heading"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_muted"]
        )
        self.watch_status.pack(pady=10)

        self.watch_log = tk.Text(
            status_card,
            height=10,
            wrap="word",
            font=FONTS["mono"],
            bg=COLORS["bg_dark"],
            fg=COLORS["text"],
            relief="flat",
            padx=10,
            pady=10
        )
        self.watch_log.pack(fill="both", expand=True)
        self.watch_log.tag_configure("success", foreground=COLORS["success"])
        self.watch_log.tag_configure("error", foreground=COLORS["danger"])
        self.watch_log.tag_configure("info", foreground=COLORS["primary"])

    def _build_settings_tab(self):
        """Aba de configuracoes gerais."""
        main = tk.Frame(self.tab_settings, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Perfis
        profile_card = Card(main, title="Perfis de Configuracao",
                           subtitle="Salve e carregue suas configuracoes")
        profile_card.pack(fill="x", pady=(0, 10))

        profile_frame = tk.Frame(profile_card, bg=COLORS["bg_card"])
        profile_frame.pack(fill="x")

        tk.Label(
            profile_frame,
            text="Perfil:",
            font=FONTS["body"],
            bg=COLORS["bg_card"],
            fg=COLORS["text"]
        ).pack(side="left")

        self.profile_combo = ttk.Combobox(
            profile_frame,
            values=self.profile_manager.list_profiles(),
            state="readonly",
            width=20,
            font=FONTS["body"]
        )
        self.profile_combo.pack(side="left", padx=(10, 10))

        ModernButton(
            profile_frame,
            text="Carregar",
            command=self._load_profile,
            style="primary",
            size="small"
        ).pack(side="left", padx=(0, 5))

        ModernButton(
            profile_frame,
            text="Salvar",
            command=self._save_profile,
            style="success",
            size="small"
        ).pack(side="left", padx=(0, 5))

        ModernButton(
            profile_frame,
            text="Excluir",
            command=self._delete_profile,
            style="danger",
            size="small"
        ).pack(side="left")

        # Formato de saida
        format_card = Card(main, title="Formato de Saida")
        format_card.pack(fill="x", pady=(0, 10))

        format_frame = tk.Frame(format_card, bg=COLORS["bg_card"])
        format_frame.pack(fill="x")

        tk.Label(
            format_frame,
            text="Formato:",
            font=FONTS["body"],
            bg=COLORS["bg_card"],
            fg=COLORS["text"]
        ).pack(side="left")

        for fmt, label in [("jpg", "JPG (menor tamanho)"), ("png", "PNG (sem perdas)")]:
            rb = tk.Radiobutton(
                format_frame,
                text=label,
                variable=self.out_format,
                value=fmt,
                font=FONTS["body"],
                bg=COLORS["bg_card"],
                fg=COLORS["text"],
                activebackground=COLORS["bg_card"],
                activeforeground=COLORS["text"],
                selectcolor=COLORS["bg_dark"]
            )
            rb.pack(side="left", padx=(20, 0))

        ModernSlider(
            format_card,
            "Qualidade JPG",
            self.quality,
            50, 100,
            lambda v: f"{int(v)}%",
            tooltip="Qualidade da compressao JPG (maior = melhor qualidade, maior arquivo)"
        ).pack(fill="x", pady=(10, 0))

        # Sobre
        about_card = Card(main, title="Sobre")
        about_card.pack(fill="x", pady=(0, 10))

        tk.Label(
            about_card,
            text=f"ROTOLINE Marca D'agua v{APP_VERSION}\n\n"
                 f"Desenvolvido para facilitar a aplicacao de marcas d'agua\n"
                 f"em imagens de produtos de forma rapida e profissional.\n\n"
                 f"Tamanho de saida: {OUT_SIZE[0]}x{OUT_SIZE[1]} pixels\n"
                 f"Formatos suportados: {', '.join(sorted(SUPPORTED_EXT))}",
            font=FONTS["body"],
            bg=COLORS["bg_card"],
            fg=COLORS["text_secondary"],
            justify="left"
        ).pack(anchor="w")

        # Ajuda
        help_card = Card(main, title="Dicas de Uso")
        help_card.pack(fill="x")

        tips = [
            "-> Use o modo 'Uma Foto' para testar as configuracoes",
            "-> Arraste o ponto para ajustar a posicao da marca d'agua",
            "-> Salve perfis para diferentes tipos de produtos",
            "-> O modo Monitorar e ideal para fluxos automaticos",
            "-> Use 'IA' para fundos complexos (requer rembg instalado)",
        ]

        for tip in tips:
            tk.Label(
                help_card,
                text=tip,
                font=FONTS["small"],
                bg=COLORS["bg_card"],
                fg=COLORS["text_secondary"],
                anchor="w"
            ).pack(fill="x", pady=2)

    # ============================================================
    # METODOS - UMA FOTO
    # ============================================================

    def _pick_single_file(self):
        """Abre dialogo para escolher uma foto."""
        filetypes = [
            ("Imagens", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
            ("Todos os arquivos", "*.*")
        ]
        path = filedialog.askopenfilename(title="Escolha uma foto", filetypes=filetypes)
        if path:
            self.single_file.set(path)
            self._load_single_preview(path)

    def _load_single_preview(self, path: str):
        """Carrega preview da imagem original."""
        try:
            self.single_original_img = Image.open(path).convert("RGBA")
            self._update_original_preview()
            self.single_result_img = None
            self.canvas_result.delete("all")
            self.btn_single_save.config(state="disabled")
            self.single_status.config(
                text="[OK] Foto carregada! Clique em PROCESSAR",
                fg=COLORS["primary"]
            )
        except Exception as e:
            messagebox.showerror("Erro", f"Nao foi possivel abrir a foto:\n{e}")

    def _update_original_preview(self):
        """Atualiza preview da imagem original."""
        if self.single_original_img is None:
            return

        img = self.single_original_img.copy()
        img.thumbnail((280, 280), Image.LANCZOS)

        self.single_preview_original = ImageTk.PhotoImage(img)
        self.canvas_original.delete("all")
        self.canvas_original.create_image(140, 140, image=self.single_preview_original, anchor="center")

    def _update_result_preview(self):
        """Atualiza preview do resultado."""
        if self.single_result_img is None:
            return

        img = self.single_result_img.copy()
        img.thumbnail((280, 280), Image.LANCZOS)

        self.single_preview_result = ImageTk.PhotoImage(img)
        self.canvas_result.delete("all")
        self.canvas_result.create_image(140, 140, image=self.single_preview_result, anchor="center")

    def _auto_preview_single(self):
        """Preview automatico ao mudar posicao."""
        pass  # Pode ser implementado para preview em tempo real

    def _process_single(self):
        """Processa uma unica foto."""
        if self.single_original_img is None:
            messagebox.showwarning("Atencao", "Primeiro escolha uma foto para processar.")
            return

        if not WATERMARK_FILE.exists():
            messagebox.showerror("Erro", "Arquivo de marca d'agua nao encontrado.")
            return

        self.btn_single_process.set_loading(True)
        self.single_status.config(text="Processando...", fg=COLORS["warning"])
        self.update()

        def worker():
            try:
                wm = Image.open(WATERMARK_FILE).convert("RGBA")
                if wm.size != OUT_SIZE:
                    wm = wm.resize(OUT_SIZE, Image.LANCZOS)

                strength = float(self.single_wm_strength.get())
                wm = boost_watermark_alpha(wm, factor=strength)

                method_text = self.single_method_combo.get()
                bg_method = "ai" if "IA" in method_text else "simple"

                result = process_image(
                    img=self.single_original_img.copy(),
                    wm_rgba=wm,
                    padding=float(self.single_padding.get()),
                    thr=int(self.single_thr.get()),
                    remove_bg=bool(self.single_remove_bg.get()),
                    bg_method=bg_method,
                    pos_x=float(self.single_pos_x.get()),
                    pos_y=float(self.single_pos_y.get())
                )

                self.single_result_img = result

                self.after(0, self._update_result_preview)
                self.after(0, lambda: self.btn_single_save.config(state="normal"))
                self.after(0, lambda: self.single_status.config(
                    text="[OK] Pronto! Clique em SALVAR",
                    fg=COLORS["success"]
                ))

            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Erro", f"Erro ao processar:\n{e}"))
                self.after(0, lambda: self.single_status.config(
                    text="[ERRO] Falha no processamento",
                    fg=COLORS["danger"]
                ))
            finally:
                self.after(0, lambda: self.btn_single_process.set_loading(False))

        threading.Thread(target=worker, daemon=True).start()

    def _save_single(self):
        """Salva a foto processada."""
        if self.single_result_img is None:
            messagebox.showwarning("Atencao", "Primeiro processe uma foto.")
            return

        original_path = Path(self.single_file.get())
        suggested_name = f"{original_path.stem}_watermark.{self.out_format.get()}"

        filetypes = [("JPEG", "*.jpg"), ("PNG", "*.png")]
        if self.out_format.get() == "png":
            filetypes.reverse()

        path = filedialog.asksaveasfilename(
            title="Salvar foto processada",
            defaultextension=f".{self.out_format.get()}",
            initialfile=suggested_name,
            filetypes=filetypes
        )

        if path:
            try:
                if path.lower().endswith(".png"):
                    self.single_result_img.save(path, "PNG", compress_level=6)
                else:
                    self.single_result_img.convert("RGB").save(
                        path, "JPEG",
                        quality=self.quality.get(),
                        optimize=True
                    )

                self.single_status.config(
                    text=f"[OK] Salvo: {Path(path).name}",
                    fg=COLORS["success"]
                )
                messagebox.showinfo("Sucesso", "Foto salva com sucesso!")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar:\n{e}")

    # ============================================================
    # METODOS - VARIAS FOTOS
    # ============================================================

    def _pick_in_dir(self):
        """Escolhe pasta de origem."""
        p = filedialog.askdirectory(title="Escolha a pasta com as fotos originais")
        if p:
            self.in_dir.set(p)

    def _pick_out_dir(self):
        """Escolhe pasta de destino."""
        p = filedialog.askdirectory(title="Escolha a pasta para salvar os resultados")
        if p:
            self.out_dir.set(p)

    def _pick_dir(self, var):
        """Escolhe uma pasta generica."""
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _open_out_dir(self):
        """Abre pasta de destino no explorador."""
        p = self.out_dir.get().strip()
        if not p:
            messagebox.showinfo("Atencao", "Primeiro escolha uma pasta de destino.")
            return
        if not Path(p).exists():
            messagebox.showinfo("Atencao", "A pasta de destino ainda nao existe.")
            return
        os.startfile(p)

    def _append_log(self, msg: str, tag: str = None):
        """Adiciona mensagem ao log."""
        self.log.insert("end", msg + "\n", tag)
        self.log.see("end")

    def _validate_batch(self):
        """Valida configuracoes do processamento em lote."""
        in_dir = Path(self.in_dir.get().strip())
        out_dir = Path(self.out_dir.get().strip())

        if not in_dir.exists():
            return False, "Escolha uma pasta de origem valida."
        if not out_dir.exists():
            return False, "Escolha uma pasta de destino valida."
        if not WATERMARK_FILE.exists():
            return False, "Arquivo de marca d'agua nao encontrado."

        return True, ""

    def _run_batch(self):
        """Executa processamento em lote."""
        ok, msg = self._validate_batch()
        if not ok:
            messagebox.showerror("Erro", msg)
            return

        self.btn_batch_run.set_loading(True)
        self.log.delete("1.0", "end")
        self.batch_status.config(text="Iniciando...")
        self.progress.config(value=0)

        threading.Thread(target=self._batch_worker, daemon=True).start()

    def _batch_worker(self):
        """Worker thread para processamento em lote."""
        try:
            in_dir = Path(self.in_dir.get().strip())
            out_dir = Path(self.out_dir.get().strip())

            wm = Image.open(WATERMARK_FILE).convert("RGBA")
            if wm.size != OUT_SIZE:
                wm = wm.resize(OUT_SIZE, Image.LANCZOS)

            strength = float(self.wm_strength.get())
            wm = boost_watermark_alpha(wm, factor=strength)

            files = list(iter_images(in_dir))
            total = len(files)

            if total == 0:
                self.after(0, lambda: self.batch_status.config(text="Nenhuma foto encontrada"))
                self.after(0, lambda: messagebox.showinfo("Atencao", "Nenhuma foto encontrada na pasta de origem."))
                return

            self.after(0, lambda: self.progress.config(maximum=total, value=0))

            remove_bg_flag = bool(self.remove_bg.get())
            padding = float(self.padding.get())
            thr = int(self.thr.get())
            out_format = self.out_format.get()
            quality = self.quality.get()

            method_text = self.batch_method_combo.get()
            bg_method = "ai" if "IA" in method_text else "simple"

            pos_x = float(self.pos_x.get())
            pos_y = float(self.pos_y.get())

            done = 0
            errors = 0
            start_time = time.time()

            for src in files:
                rel = src.relative_to(in_dir)
                dst = (out_dir / rel).with_suffix(f".{out_format}")

                try:
                    process_one(
                        in_path=src,
                        out_path=dst,
                        wm_rgba=wm,
                        padding=padding,
                        thr=thr,
                        remove_bg=remove_bg_flag,
                        out_format=out_format,
                        bg_method=bg_method,
                        pos_x=pos_x,
                        pos_y=pos_y,
                        quality=quality
                    )
                    self.after(0, lambda s=str(src.name): self._append_log(f"[OK] {s}", "success"))
                except Exception as e:
                    errors += 1
                    self.after(0, lambda s=str(src.name), er=str(e): self._append_log(f"[ERRO] {s}: {er}", "error"))

                done += 1
                self.after(0, lambda v=done, t=total: (
                    self.progress.config(value=v),
                    self.batch_status.config(text=f"Processando: {v}/{t}")
                ))

            elapsed = time.time() - start_time

            if errors == 0:
                final_msg = f"[OK] Concluido! {done} fotos em {elapsed:.1f}s"
                self.after(0, lambda: self.batch_status.config(text=f"Concluido: {done} fotos"))
            else:
                final_msg = f"Concluido: {done - errors} OK, {errors} erros"
                self.after(0, lambda: self.batch_status.config(text=final_msg))

            self.after(0, lambda: self._append_log(f"\n{final_msg}", "info"))
            self.after(0, lambda m=final_msg: messagebox.showinfo("Concluido", m))

        finally:
            self.after(0, lambda: self.btn_batch_run.set_loading(False))

    # ============================================================
    # METODOS - MONITORAMENTO
    # ============================================================

    def _toggle_watch(self):
        """Inicia ou para o monitoramento."""
        if self.watching:
            self._stop_watch()
        else:
            self._start_watch()

    def _start_watch(self):
        """Inicia monitoramento de pasta."""
        in_dir = self.watch_in_dir.get().strip()
        out_dir = self.watch_out_dir.get().strip()

        if not in_dir or not Path(in_dir).exists():
            messagebox.showerror("Erro", "Escolha uma pasta valida para monitorar.")
            return
        if not out_dir:
            messagebox.showerror("Erro", "Escolha uma pasta de destino.")
            return

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        self.watching = True
        self.btn_watch.config(text="⏹  PARAR MONITORAMENTO")
        self.watch_status.config(text="[MONITORANDO...]", fg=COLORS["success"])
        self._append_watch_log(f"Iniciado monitoramento de: {in_dir}", "info")

        self.watch_thread = threading.Thread(target=self._watch_worker, daemon=True)
        self.watch_thread.start()

    def _stop_watch(self):
        """Para o monitoramento."""
        self.watching = False
        self.btn_watch.config(text="▶  INICIAR MONITORAMENTO")
        self.watch_status.config(text="[PARADO]", fg=COLORS["text_muted"])
        self._append_watch_log("Monitoramento parado.", "info")

    def _append_watch_log(self, msg: str, tag: str = None):
        """Adiciona mensagem ao log do watch."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.watch_log.insert("end", f"[{timestamp}] {msg}\n", tag)
        self.watch_log.see("end")

    def _watch_worker(self):
        """Worker thread para monitoramento."""
        in_dir = Path(self.watch_in_dir.get().strip())
        out_dir = Path(self.watch_out_dir.get().strip())
        processed = set()

        # Carrega marca d'agua
        wm = Image.open(WATERMARK_FILE).convert("RGBA")
        if wm.size != OUT_SIZE:
            wm = wm.resize(OUT_SIZE, Image.LANCZOS)
        wm = boost_watermark_alpha(wm, factor=float(self.wm_strength.get()))

        while self.watching:
            try:
                for src in in_dir.iterdir():
                    if not self.watching:
                        break
                    if not src.is_file():
                        continue
                    if src.suffix.lower() not in SUPPORTED_EXT:
                        continue
                    if src in processed:
                        continue

                    # Espera arquivo estar completo
                    time.sleep(0.5)

                    dst = out_dir / f"{src.stem}_watermark.{self.out_format.get()}"

                    try:
                        process_one(
                            in_path=src,
                            out_path=dst,
                            wm_rgba=wm,
                            padding=float(self.padding.get()),
                            thr=int(self.thr.get()),
                            remove_bg=bool(self.remove_bg.get()),
                            out_format=self.out_format.get(),
                            bg_method="simple",
                            pos_x=float(self.pos_x.get()),
                            pos_y=float(self.pos_y.get()),
                            quality=self.quality.get()
                        )

                        processed.add(src)
                        self.after(0, lambda s=src.name: self._append_watch_log(f"[OK] {s}", "success"))

                        if self.watch_delete_original.get():
                            try:
                                src.unlink()
                            except:
                                pass

                    except Exception as e:
                        self.after(0, lambda s=src.name, e=str(e): self._append_watch_log(f"[ERRO] {s}: {e}", "error"))
                        processed.add(src)

            except Exception as e:
                self.after(0, lambda e=str(e): self._append_watch_log(f"Erro: {e}", "error"))

            time.sleep(1)

    # ============================================================
    # METODOS - PERFIS
    # ============================================================

    def _get_current_settings(self) -> dict:
        """Retorna configuracoes atuais como dicionario."""
        return {
            "remove_bg": self.remove_bg.get(),
            "padding": self.padding.get(),
            "thr": self.thr.get(),
            "wm_strength": self.wm_strength.get(),
            "pos_x": self.pos_x.get(),
            "pos_y": self.pos_y.get(),
            "out_format": self.out_format.get(),
            "quality": self.quality.get(),
        }

    def _apply_settings(self, settings: dict):
        """Aplica configuracoes de um dicionario."""
        if "remove_bg" in settings:
            self.remove_bg.set(settings["remove_bg"])
            self.single_remove_bg.set(settings["remove_bg"])
        if "padding" in settings:
            self.padding.set(settings["padding"])
            self.single_padding.set(settings["padding"])
        if "thr" in settings:
            self.thr.set(settings["thr"])
            self.single_thr.set(settings["thr"])
        if "wm_strength" in settings:
            self.wm_strength.set(settings["wm_strength"])
            self.single_wm_strength.set(settings["wm_strength"])
        if "pos_x" in settings:
            self.pos_x.set(settings["pos_x"])
            self.single_pos_x.set(settings["pos_x"])
        if "pos_y" in settings:
            self.pos_y.set(settings["pos_y"])
            self.single_pos_y.set(settings["pos_y"])
        if "out_format" in settings:
            self.out_format.set(settings["out_format"])
        if "quality" in settings:
            self.quality.set(settings["quality"])

        # Atualiza seletores de posicao
        self.single_position_selector.update_display()
        self.batch_position_selector.update_display()

    def _refresh_profiles_combo(self):
        """Atualiza lista de perfis no combo."""
        self.profile_combo["values"] = self.profile_manager.list_profiles()

    def _save_profile(self):
        """Salva perfil atual."""
        name = simpledialog.askstring("Salvar Perfil", "Nome do perfil:")
        if name:
            name = name.strip()
            if name:
                self.profile_manager.save_profile(name, self._get_current_settings())
                self._refresh_profiles_combo()
                self.profile_combo.set(name)
                messagebox.showinfo("Sucesso", f"Perfil '{name}' salvo!")

    def _load_profile(self):
        """Carrega perfil selecionado."""
        name = self.profile_combo.get()
        if not name:
            messagebox.showwarning("Atencao", "Selecione um perfil para carregar.")
            return

        settings = self.profile_manager.load_profile(name)
        if settings:
            self._apply_settings(settings)
            messagebox.showinfo("Sucesso", f"Perfil '{name}' carregado!")
        else:
            messagebox.showerror("Erro", f"Nao foi possivel carregar o perfil '{name}'.")

    def _delete_profile(self):
        """Exclui perfil selecionado."""
        name = self.profile_combo.get()
        if not name:
            messagebox.showwarning("Atencao", "Selecione um perfil para excluir.")
            return

        if messagebox.askyesno("Confirmar", f"Excluir o perfil '{name}'?"):
            self.profile_manager.delete_profile(name)
            self._refresh_profiles_combo()
            self.profile_combo.set("")
            messagebox.showinfo("Sucesso", f"Perfil '{name}' excluido!")


# ============================================================
# CLI - LINHA DE COMANDO
# ============================================================

def run_cli():
    """Executa em modo linha de comando."""
    parser = argparse.ArgumentParser(
        description="ROTOLINE - Aplicador de Marca D'agua",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python rotoline_watermark_app.py --input foto.jpg --output foto_wm.jpg
  python rotoline_watermark_app.py --input pasta/ --output saida/ --batch
  python rotoline_watermark_app.py --input pasta/ --output saida/ --batch --no-remove-bg
        """
    )

    parser.add_argument("--input", "-i", required=True, help="Arquivo ou pasta de entrada")
    parser.add_argument("--output", "-o", required=True, help="Arquivo ou pasta de saida")
    parser.add_argument("--batch", "-b", action="store_true", help="Processa pasta inteira")
    parser.add_argument("--format", "-f", choices=["jpg", "png"], default="jpg", help="Formato de saida")
    parser.add_argument("--quality", "-q", type=int, default=95, help="Qualidade JPG (50-100)")
    parser.add_argument("--padding", "-p", type=float, default=0.10, help="Margem (0.05-0.18)")
    parser.add_argument("--strength", "-s", type=float, default=1.8, help="Intensidade da marca (1.0-3.0)")
    parser.add_argument("--pos-x", type=float, default=0.5, help="Posicao X da marca d'agua (0.0-1.0)")
    parser.add_argument("--pos-y", type=float, default=0.5, help="Posicao Y da marca d'agua (0.0-1.0)")
    parser.add_argument("--no-remove-bg", action="store_true", help="Nao remove fundo")
    parser.add_argument("--ai", action="store_true", help="Usa IA para remocao de fundo")

    args = parser.parse_args()

    if not WATERMARK_FILE.exists():
        print(f"ERRO: Marca d'agua nao encontrada: {WATERMARK_FILE}")
        sys.exit(1)

    # Carrega marca d'agua
    wm = Image.open(WATERMARK_FILE).convert("RGBA")
    if wm.size != OUT_SIZE:
        wm = wm.resize(OUT_SIZE, Image.LANCZOS)
    wm = boost_watermark_alpha(wm, factor=args.strength)

    in_path = Path(args.input)
    out_path = Path(args.output)

    remove_bg = not args.no_remove_bg
    bg_method = "ai" if args.ai else "simple"

    if args.batch:
        if not in_path.is_dir():
            print(f"ERRO: {in_path} nao e uma pasta")
            sys.exit(1)

        out_path.mkdir(parents=True, exist_ok=True)
        files = list(iter_images(in_path))
        total = len(files)

        if total == 0:
            print("Nenhuma imagem encontrada.")
            sys.exit(0)

        print(f"Processando {total} imagens...")

        for i, src in enumerate(files, 1):
            rel = src.relative_to(in_path)
            dst = (out_path / rel).with_suffix(f".{args.format}")

            try:
                process_one(
                    in_path=src,
                    out_path=dst,
                    wm_rgba=wm,
                    padding=args.padding,
                    thr=20,
                    remove_bg=remove_bg,
                    out_format=args.format,
                    bg_method=bg_method,
                    pos_x=args.pos_x,
                    pos_y=args.pos_y,
                    quality=args.quality
                )
                print(f"[{i}/{total}] OK: {src.name}")
            except Exception as e:
                print(f"[{i}/{total}] ERRO: {src.name} - {e}")

        print("Concluido!")

    else:
        if not in_path.is_file():
            print(f"ERRO: {in_path} nao e um arquivo")
            sys.exit(1)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            process_one(
                in_path=in_path,
                out_path=out_path,
                wm_rgba=wm,
                padding=args.padding,
                thr=20,
                remove_bg=remove_bg,
                out_format=args.format,
                bg_method=bg_method,
                pos_x=args.pos_x,
                pos_y=args.pos_y,
                quality=args.quality
            )
            print(f"OK: {out_path}")
        except Exception as e:
            print(f"ERRO: {e}")
            sys.exit(1)


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    # Verifica se ha argumentos CLI
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        run_cli()
    else:
        # Modo GUI
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        app = App()
        app.mainloop()
