"""
Prépare le dataset d'entraînement LoRA à partir du dossier `banque d'image`.

Ce que fait le script :
  1. Recadre chaque image en carré (center crop) puis la redimensionne en 1024x1024
     (résolution native attendue par SDXL).
  2. Convertit en RGB / PNG (supprime canal alpha, EXIF, etc.).
  3. Génère un fichier caption `.txt` par image, avec le mot-déclencheur (trigger word)
     en tête, au format attendu par kohya_ss / ai-toolkit.

Usage :
    python training/prepare_dataset.py
    python training/prepare_dataset.py --trigger sollechar --size 1024

Après exécution, RELIS les .txt générés : ils contiennent une description de base
identique pour toutes les images. Affine-les à la main (pose, tenue, fond, expression)
pour que le modèle apprenne ce qui VARIE et fige le reste (le perso + le style).
"""

import argparse
from pathlib import Path

from PIL import Image, ImageOps

# Dossier source (relatif à la racine du projet)
SOURCE_DIR = Path("banque d'image")
OUTPUT_DIR = Path("training/dataset")

# Caption de base appliquée à chaque image. Le trigger word est ajouté devant.
# Décris ici UNIQUEMENT ce qui est commun à toutes les images (perso + style).
# Le reste (pose, tenue, fond...) devra être ajouté/édité manuellement par image.
BASE_CAPTION = (
    "a purple furry one-eyed cartoon character, comic illustration style, "
    "thick black outlines, bold flat colors"
)

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_square(img: Image.Image, size: int) -> Image.Image:
    """Recadre l'image au centre en carré, puis redimensionne en size x size."""
    img = ImageOps.exif_transpose(img)  # respecte l'orientation EXIF
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare le dataset LoRA.")
    parser.add_argument("--trigger", default="sollechar",
                        help="Mot-déclencheur unique du personnage (def: sollechar)")
    parser.add_argument("--size", type=int, default=1024,
                        help="Résolution carrée de sortie (def: 1024 pour SDXL)")
    parser.add_argument("--source", default=str(SOURCE_DIR),
                        help="Dossier source des images")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="Dossier de sortie")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if not source.is_dir():
        raise SystemExit(f"Dossier source introuvable : {source.resolve()}")

    images = sorted(p for p in source.iterdir() if p.suffix.lower() in VALID_EXT)
    if not images:
        raise SystemExit(f"Aucune image trouvée dans {source.resolve()}")

    caption = f"{args.trigger}, {BASE_CAPTION}"
    print(f"{len(images)} images trouvées. Trigger='{args.trigger}', taille={args.size}px")

    for i, path in enumerate(images, start=1):
        try:
            with Image.open(path) as img:
                square = make_square(img, args.size)
        except Exception as exc:  # image corrompue, format exotique...
            print(f"  [SKIP] {path.name} : {exc}")
            continue

        stem = f"{args.trigger}_{i:03d}"
        square.save(output / f"{stem}.png")
        (output / f"{stem}.txt").write_text(caption, encoding="utf-8")
        print(f"  [OK] {path.name} -> {stem}.png (+ .txt)")

    print(f"\nTerminé. Dataset prêt dans : {output.resolve()}")
    print("Pense à RELIRE et affiner les fichiers .txt avant l'entraînement.")


if __name__ == "__main__":
    main()
