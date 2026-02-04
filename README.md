# ROTOLINE Watermark

Aplicativo desktop para aplicar marca d'agua em imagens de produtos de forma rapida e profissional.

## Funcionalidades

- **Modo Foto Unica** - Processa uma imagem com preview em tempo real
- **Modo Lote** - Processa multiplas imagens de uma pasta
- **Modo Monitoramento** - Monitora pasta e processa automaticamente novas imagens
- **Remocao de Fundo** - Remove fundo automaticamente (simples ou IA)
- **Posicionamento da Marca D'agua** - Arraste para posicionar a marca d'agua
- **Perfis** - Salve e carregue configuracoes

## Requisitos

- Python 3.8+
- Windows 10/11

## Instalacao

### Usando o executavel (recomendado)

Baixe o `ROTOLINE_Watermark.exe` da pasta `dist/` e execute.

### Desenvolvimento

```bash
# Clone o repositorio
git clone https://github.com/gitrotoline/add_watermark_desktop.git
cd add_watermark_desktop

# Crie ambiente virtual
python -m venv venv
venv\Scripts\activate

# Instale dependencias
pip install pillow numpy

# (Opcional) Para remocao de fundo por IA
pip install rembg

# Execute
python rotoline_watermark_app.py
```

## Uso via CLI

```bash
# Processar uma imagem
python rotoline_watermark_app.py -i foto.jpg -o foto_wm.jpg

# Processar pasta inteira
python rotoline_watermark_app.py -i pasta_entrada/ -o pasta_saida/ --batch

# Opcoes disponiveis
python rotoline_watermark_app.py --help
```

## Compilar Executavel

```bash
pip install pyinstaller
pyinstaller ROTOLINE_Watermark.spec --noconfirm
```

O executavel sera gerado em `dist/ROTOLINE_Watermark.exe`

## Estrutura

```
├── rotoline_watermark_app.py    # Aplicacao principal
├── rotoline_watermark_only_1200.png  # Imagem da marca d'agua
├── ROTOLINE_Watermark.spec      # Configuracao PyInstaller
├── profiles/                    # Perfis salvos
├── dist/                        # Executavel compilado
└── README.md
```

## Configuracoes

| Opcao | Descricao |
|-------|-----------|
| Margem | Espaco entre produto e bordas (5-18%) |
| Intensidade | Visibilidade da marca d'agua (1.0-3.0x) |
| Posicao | Posicao da marca d'agua (arraste no seletor) |
| Formato | JPG (menor) ou PNG (sem perdas) |
| Qualidade | Qualidade JPG (50-100%) |

## Licenca

Uso interno ROTOLINE.
