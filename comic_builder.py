import os
import textwrap
from PIL import Image, ImageDraw, ImageFont

def create_comic_bubble(image_path, output_path, text, bubble_position, tail_target, max_width=20):
    """
    Ajoute une bulle de dialogue façon "comic book" sur une image.
    
    :param image_path: Chemin de l'image d'origine.
    :param output_path: Chemin de sauvegarde de l'image modifiée.
    :param text: Le texte à écrire dans la bulle.
    :param bubble_position: Tuple (x, y) du centre de la bulle.
    :param tail_target: Tuple (x, y) de la pointe de la queue de la bulle (vers le personnage).
    :param max_width: Nombre maximum de caractères par ligne.
    """
    try:
        # Charger l'image
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        
        # Charger une police (utiliser la police par défaut si aucune police TTF n'est fournie)
        # Pour un vrai rendu comic, téléchargez 'Bangers.ttf' ou 'ComicSans.ttf' et remplacez ici :
        # font = ImageFont.truetype("Bangers.ttf", 40)
        try:
            # Essayer de charger une police Windows courante
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()

        # Préparer le texte (gérer les retours à la ligne)
        lines = textwrap.wrap(text, width=max_width)
        
        # ⚡ Bolt Optimization: Cache bounding box dimensions to eliminate redundant textbbox() calls
        line_dims = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_dims.append({
                'line': line,
                'width': bbox[2] - bbox[0],
                'height': bbox[3] - bbox[1]
            })

        max_text_width = max((dim['width'] for dim in line_dims), default=0)
        total_text_height = sum(dim['height'] for dim in line_dims) + 5 * (len(lines) - 1) # 5px d'espacement entre les lignes
        
        # Dimensions de la bulle (avec une marge/padding)
        padding_x = 30
        padding_y = 20
        bubble_width = max_text_width + padding_x * 2
        bubble_height = total_text_height + padding_y * 2
        
        bx, by = bubble_position
        # Coordonnées du rectangle englobant la bulle elliptique
        left = bx - bubble_width // 2
        top = by - bubble_height // 2
        right = bx + bubble_width // 2
        bottom = by + bubble_height // 2
        
        # Dessiner la queue de la bulle en premier (pour qu'elle soit sous le contour de la bulle)
        # On définit un triangle qui part du centre-bas de la bulle vers la cible
        tail_width = 30
        point1 = (bx - tail_width // 2, bottom - 10)
        point2 = (bx + tail_width // 2, bottom - 10)
        point3 = tail_target
        
        # Ombre ou contour noir
        draw.polygon([point1, point2, point3], fill="white", outline="black", width=3)
        draw.ellipse([left, top, right, bottom], fill="white", outline="black", width=3)
        
        # Redessiner le triangle sans bordure pour effacer la ligne de séparation avec l'ellipse
        draw.polygon([point1, point2, point3], fill="white")
        draw.ellipse([left+2, top+2, right-2, bottom-2], fill="white") # Effacer l'intérieur
        
        # Écrire le texte centré dans la bulle
        current_y = top + padding_y
        for dim in line_dims:
            line_x = bx - dim['width'] // 2
            draw.text((line_x, current_y), dim['line'], fill="black", font=font)
            current_y += dim['height'] + 5
            
        # Sauvegarder
        img = img.convert("RGB") # Enlever l'alpha pour jpg
        img.save(output_path)
        print(f"[SUCCES] Image avec bulle sauvegardee : {output_path}")

    except Exception as e:
        print(f"[ERREUR] Erreur : {e}")

if __name__ == "__main__":
    # Test du script
    import urllib.request
    
    # 1. On crée un dossier de test s'il n'existe pas
    os.makedirs("generated_images", exist_ok=True)
    
    # 2. Utilisons une de vos images de la banque comme base (assurez-vous du chemin)
    # Remplacer par le vrai chemin d'une image de la banque d'image
    test_input = "banque d'image/photo_5161364836896672656_y.jpg" 
    test_output = "generated_images/test_comic_bubble.jpg"
    
    if os.path.exists(test_input):
        print(f"Génération d'une bulle sur {test_input}...")
        create_comic_bubble(
            image_path=test_input,
            output_path=test_output,
            text="Salut ! Je suis Sollie et je suis maintenant dans un comic book !",
            bubble_position=(250, 150), # Position (x, y) de la bulle (à ajuster selon l'image)
            tail_target=(400, 400),     # Vers où pointe la queue (la bouche du perso)
            max_width=25                # Caractères max par ligne
        )
    else:
        print(f"Image {test_input} introuvable pour le test. Modifiez le chemin dans le script.")
