# -*- coding: utf-8 -*-
"""
Réécrit les 79 captions .txt du dataset en LANGAGE NATUREL pour un
entraînement LoRA sur **Stable Diffusion XL 1.0 Base** (XL Lora Trainer
Hollowstrawberry).

Pourquoi du langage naturel et pas des tags booru :
  SDXL 1.0 Base est entraîné sur des légendes en phrases (CLIP + OpenCLIP),
  pas sur des tags booru comme Illustrious. Il répond donc beaucoup mieux à
  des descriptions rédigées qu'à des listes de mots-clés.

Convention (choisie avec l'utilisateur) :
  - `sollechar` = mot-déclencheur, toujours en 1er, il porte l'IDENTITÉ.
  - On AJOUTE après lui une courte ancre d'identité CONSTANTE (ANCHOR), car
    SDXL base ne connaît pas du tout ce personnage inventé : la répéter aide
    CLIP à associer fermement le trigger à la créature.
  - Puis on décrit ce qui VARIE en langage naturel : style de rendu, cadrage,
    tenue/accessoires, pose/action, décor, expression. Le modèle apprend ainsi
    à dissocier le perso (fixe) de la scène (variable).

Usage :
    python training/write_captions_natural.py
"""

from pathlib import Path

DATASET = Path(__file__).resolve().parent / "dataset"

# Ancre d'identité constante (traits invariants de Solle).
ANCHOR = (
    "sollechar, purple furry monster"
)

# Description (style + scène) propre à chaque image. La caption finale =
# f"{ANCHOR}, {desc}".
SCENES = {
    "ChatGPT Image 24 juin 2026, 18_52_43":
        "comic style, full body standing and looking at viewer standing worried expression, set lavender background",
    "ChatGPT Image 24 juin 2026, 18_54_39":
        "comic style, walking in a sunny park with trees and a bench, three-quarter view",
    "ChatGPT Image 24 juin 2026, 18_55_57":
        "comic style, bust portrait three-quarter view purple background",
    "ChatGPT Image 24 juin 2026, 18_57_23":
        "comic style, sitting on beach hugging knees, sunset over ocean",
    "ChatGPT Image 24 juin 2026, 18_58_58":
        "comic style, sitting cross-legged and looking at viewer purple background",
    "Gemini_Generated_Image_13i6rx13i6rx13i6":
        "comic style, upper body gesturing, looking aside, grayish-purple background",
    "Gemini_Generated_Image_19s4b319s4b319s4":
        "comic style, seen from behind, walking in a forest",
    "Gemini_Generated_Image_29xw3r29xw3r29xw":
        "comic style, lounging inside a tent, glowing oil lantern",
    "Gemini_Generated_Image_351ygo351ygo351y":
        "comic style, walking three-quarter view, gray background ",
    "Gemini_Generated_Image_3cun403cun403cun":
        "comic style, bust portrait looking back over shoulder, light purple background",
    "Gemini_Generated_Image_48li2f48li2f48li":
        "comic style, close up portrait laughing, eyes closed, mouth open, purple background",
    "Gemini_Generated_Image_5p6p6g5p6p6g5p6p":
        "comic style, leaning against brick wall in gritty city alley, peeking nervously",
    "Gemini_Generated_Image_6s4pqo6s4pqo6s4p":
        "comic style, seen from behind on rooftop, looking at city skyline at sunset",
    "Gemini_Generated_Image_7diljh7diljh7dil":
        "comic style, close up portrait waving hand, purple background",
    "Gemini_Generated_Image_7pz3um7pz3um7pz3":
        "comic style, sitting on stool by fireplace in cozy living room",
    "Gemini_Generated_Image_7x8ldb7x8ldb7x8l":
        "comic style, a bust portrait arms crossed, looking at viewer grayish-purple background",
    "Gemini_Generated_Image_7ykm8i7ykm8i7ykm":
        "comic style, climbing ladder, tool belt, construction site",
    "Gemini_Generated_Image_84h9z584h9z584h9":
        "comic style, sitting in armchair, reading book in library",
    "Gemini_Generated_Image_a9gfwza9gfwza9gf":
        "comic style, lying in green meadow with wildflowers, blue sky",
    "Gemini_Generated_Image_ckmrm5ckmrm5ckmr":
        "comic style, full body standing and looking at viewer light gray background",
    "Gemini_Generated_Image_cq3zaicq3zaicq3z":
        "comic style, a moody close-up portrait with a grumpy, frowning expression, looking to the side dark background",
    "Gemini_Generated_Image_dd0izfdd0izfdd0i":
        "comic style, sitting on stone steps, quiet city street at dusk",
    "Gemini_Generated_Image_er8mqter8mqter8m":
        "comic style, close up portrait in office, pointing at itself",
    "Gemini_Generated_Image_fxfpuqfxfpuqfxfp":
        "comic style, leaping joyfully in a park, big smile",
    "Gemini_Generated_Image_isbooqisbooqisbo":
        "comic style, lying on bed reading, cozy bedroom",
    "Gemini_Generated_Image_j1hkfyj1hkfyj1hk":
        "comic style, seen from behind, fists raised, stadium",
    "Gemini_Generated_Image_jn6vqljn6vqljn6v":
        "comic style, reclining on lounge chair, holding cocktail, rooftop terrace at sunset",
    "Gemini_Generated_Image_kd1gfokd1gfokd1g":
        "comic style, holding umbrella in rainy city street at night, neon signs",
    "Gemini_Generated_Image_kps4fskps4fskps4":
        "comic style, typing on laptop at desk, modern office",
    "Gemini_Generated_Image_kwgfomkwgfomkwgf":
        "comic style, crouching in garden, reaching for seed",
    "Gemini_Generated_Image_l4yr3cl4yr3cl4yr":
        "comic style, sitting on park bench, autumn leaves, distant city skyline",
    "Gemini_Generated_Image_ljr2jdljr2jdljr2":
        "comic style, lounging on sofa, watching tv in dim living room",
    "Gemini_Generated_Image_m7xrrzm7xrrzm7xr":
        "comic style, extreme close up portrait toothy grin, against a softly blurred purple background",
    "Gemini_Generated_Image_nj84mtnj84mtnj84":
        "comic style, sitting on bar stool, holding drink, neon-lit bar",
    "Gemini_Generated_Image_nkto2ankto2ankto":
        "comic style, extreme close up portrait shocked expression, mouth open, light purple background",
    "Gemini_Generated_Image_oiv5ploiv5ploiv5":
        "comic style, sitting on park bench in autumn, pond",
    "Gemini_Generated_Image_ptriqgptriqgptri":
        "comic style, walking down snowy street, puffer jacket",
    "Gemini_Generated_Image_qt462cqt462cqt46":
        "comic style, dancing in nightclub, disco lights",
    "Gemini_Generated_Image_s4xcoys4xcoys4xc":
        "comic style, standing on city sidewalk, street background",
    "Gemini_Generated_Image_sqn2bcsqn2bcsqn2":
        "comic style, standing in grungy alley, arms crossed, graffiti wall",
    "Gemini_Generated_Image_uftmmcuftmmcuftm":
        "comic style, running down city street, motion blur",
    "Gemini_Generated_Image_x3zbdvx3zbdvx3zb":
        "comic style, close up portrait toothy grin, finger raised, pink background",
    "Gemini_Generated_Image_yyia2uyyia2uyyia":
        "illustration style, melancholic close up, crying, behind rain-streaked window",
    "Gemini_Generated_Image_z14j0sz14j0sz14j":
        "comic style, crouching on sidewalk, tying sneakers, graffiti wall",
    "photo_4949905197274696587_y":
        "comic style, a bust portrait wearing black cap and tactical vest, purple background",
    "photo_4954193176658905923_y":
        "comic style, a bust portrait wearing cap, yellow balaclava, blue jacket, purple background",
    "photo_5095909994867133241_y":
        "comic style, full body construction worker, hard hat, safety vest, blueprints, construction site",
    "photo_5098161794680818441_y":
        "drawn as a cozy warm illustration, sitting at table, holding mug, croissant, cozy sunlit room",
    "photo_5161364836896672656_y":
        "illustration style, standing on balcony, holding cocktail glass, tropical sunset",
    "photo_5167906866657430342_y":
        "spooky cartoon style, vampire costume, cape, fangs, haunted mansion",
    "photo_5167906866657430343_y":
        "spooky 3d render, mummy costume, bandages, graveyard at night, full moon",
    "photo_5167906866657430345_y":
        "cinematic 3d render, pirate captain costume, standing on raft, foggy sea, full moon",
    "photo_5183712737148734271_y":
        "comic style, witch costume, pointed hat, jack-o-lantern, halloween shop",
    "photo_5798712394207923005_y":
        "vintage pencil sketch, parchment, posed as a Vitruvian Man parody with multiple arms and legs spread out, surrounded by handwritten notes",
    "photo_5798901325524306741_y":
        "comic style, a bust portrait wearing a brown flat newsboy cap, a vest and a red bow tie while holding a billiard cue, in front of an 'ENGLISH PUB' sign and a pool hall",
    "photo_5800964194021608348_y":
        "comic style, a bust portrait steampunk king costume, crown, goggles, cape, purple background",
    "photo_5800964194021608351_y":
        "comic style, a bust portrait zeus costume, laurel wreath, toga, holding lightning bolt, on a cloud against a purple background",
    "photo_5832307868915403548_y":
        "comic style, full body wearing puffer vest and jeans, DeLorean, town hall at night",
    "photo_5837157097445985112_y":
        "comic style, extreme close up wearing gold suit, gold chains, holding champagne, golden room",
    "photo_5843583644126088277_y":
        "storybook style, painter, apron, palette, easel, artist studio",
    "photo_5872801645215812358_y":
        "drawn as a children's cartoon illustration, holding bouquet of flowers, celebrating crowd, balloons",
    "photo_5875132695762291392_y":
        "van gogh oil painting style, a bust portrait rendered in deep blue and golden-yellow tones",
    "photo_5920281409157712724_y":
        "comic style, jungle explorer, safari hat, backpack, ruins, waterfall",
    "photo_5920281409157712725_y":
        "drawn as a detailed comic illustration, storm chaser, goggles, rain jacket, camera on tripod, tornado",
    "photo_5922533208971397182_y":
        "3d render, wearing an orange hard hat, an orange safety vest and jeans, standing in a sunny wooden workshop with a potted plant",
    "photo_5922533208971397183_y":
        "photorealistic 3d render, sitting in a brown leather armchair reading a book, with a globe on a side table in a classic study library",
    "photo_5922533208971397370_y":
        "comic style, dressed as a Rio carnival dancer with large colorful feather wings, a feather headdress and a beaded necklace, dancing on a stage amid a festive street crowd, palm trees and confetti",
    "photo_5922533208971397482_y":
        "drawn in a vaporwave comic style, wearing a pastel pink jacket and pants while standing at a retro gas station, with vintage cars and a pastel swirling sky",
    "photo_5929108657876437966_y":
        "rendered as a 3d video-game poster, standing with its hands on its hips in a futuristic crypto game interface showing 'TO THE MOON', 'START GAME' and 'SCORE: 8946' text, with a rocket and neon trading charts in the background",
    "photo_5929108657876438113_y":
        "rendered as a photorealistic 3d image, a bust portrait wearing a red Santa hat and a festive red Christmas sweater, with a decorated Christmas tree and a fireplace behind it",
    "photo_5931360457690123379_y":
        "comic style, full body with its body covered in many colorful crypto token badge stickers, purple background",
    "photo_5931742555156892484_y":
        "drawn as a Banksy-style stencil graffiti on a grey concrete wall, reaching out toward a red heart-shaped balloon, the whole scene greyscale with a single red accent",
    "photo_5960605199242480804_y":
        "comic style, a close-up profile view holding a retro game controller while playing a Super Mario Bros style platformer on a vintage television, with an old game console below",
    "photo_5987623816299400981_y":
        "painted as a classical oil painting, a bust portrait dressed as a Victorian gentleman in a black top hat, black suit and bow tie, sitting inside an antique horse carriage",
    "photo_6021724370570956077_y":
        "drawn as a polished cartoon illustration, lounging on yacht, swim shorts, holding cocktail, ocean view",
    "photo_6023727221195263195_y":
        "storybook style, monk robe, holding fruit, temple pagoda steps",
    "photo_6034927503769538119_y":
        "vibrant comic style, skateboarding mid-air, tie-dye shirt, futuristic city skyline",
    "photo_6039454699947233270_y":
        "storybook style, sitting on ice, beanie, scarf, hugging penguin, snow",
    "photo_6048615697685876217_y":
        "cartoon style, full body knight costume, armor, holding sword, cape, purple background",
}


def main() -> None:
    if not DATASET.is_dir():
        raise SystemExit(f"Dataset introuvable : {DATASET}")

    written, missing = 0, []
    for stem, desc in SCENES.items():
        png = DATASET / f"{stem}.png"
        if not png.exists():
            missing.append(stem)
        caption = f"{ANCHOR}, {desc}"
        (DATASET / f"{stem}.txt").write_text(caption, encoding="utf-8")
        written += 1

    # Filets de sécurité : prévenir si une image n'a pas de caption.
    pngs = {p.stem for p in DATASET.glob("*.png")}
    orphans = sorted(pngs - set(SCENES))

    print(f"{written} captions ecrites dans {DATASET}")
    if missing:
        print(f"  [WARN] caption sans image : {missing}")
    if orphans:
        print(f"  [WARN] image SANS caption : {orphans}")


if __name__ == "__main__":
    main()
