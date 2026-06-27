"""
Réécrit les 35 fichiers caption .txt du dataset avec des descriptions détaillées,
rédigées à la main après examen de chaque image.

Convention (LoRA de personnage à styles variés) :
  - Le mot-déclencheur `sollechar` lie l'IDENTITÉ du perso (créature violette à
    fourrure, un grand œil + un petit, iris jaune/orange, grosses lèvres).
    On NE décrit donc PAS ces traits invariants -> ils sont absorbés par le trigger.
  - On décrit tout ce qui VARIE : style de rendu, cadrage, tenue/accessoires,
    action/pose, décor, expression. Le modèle apprend ainsi à dissocier le perso
    du style, ce qui permet ensuite de le générer dans n'importe quel style/scène.

Usage :
    python training/write_captions.py
"""

from pathlib import Path

DATASET = Path("training/dataset")

CAPTIONS = {
    1: "sollechar, comic illustration style, bust portrait, wearing a black Solana snapback cap and a black tactical vest, neutral expression, plain lavender background",
    2: "sollechar, comic illustration style, bust portrait, wearing a yellow balaclava hood and a blue jacket with radiation hazard symbols, plain purple background",
    3: "sollechar, comic illustration style, full body, wearing a yellow construction hard hat and an orange safety vest with jeans, holding rolled blueprints and pointing, construction site with a crane and workers, blue sky",
    4: "sollechar, cozy children's book illustration, half body sitting at a wooden table, holding a mug labeled GM with steam, croissant on a plate, smartphone, potted plants, warm sunlit cozy room",
    5: "sollechar, detailed illustration, full body standing on a wooden deck balcony, wearing an open white shirt and beige shorts, holding a cocktail glass, tropical mountains waterfall and sunset in the background, small side table with fruit",
    6: "sollechar, spooky cartoon illustration, full body in a vampire costume with a black cape and red lining and fangs, standing in a haunted mansion hallway with a chandelier mirror and cobwebs, autumn leaves on the floor",
    7: "sollechar, spooky 3d render, full body as a mummy wrapped in white bandages, dark graveyard at night with a chapel, full moon, bare trees and tombstones",
    8: "sollechar, cinematic 3d render, full body in a pirate captain costume with a black tricorn hat and coat, standing on a wooden raft, ghostly pirate ship and full moon over a foggy sea",
    9: "sollechar, comic illustration style, half body in a witch costume with a black pointed hat and cloak, holding a jack-o-lantern bucket, halloween candy shop interior with a happy halloween banner and a skeleton",
    10: "sollechar, vintage da vinci pencil sketch on aged parchment, vitruvian man parody with multiple arms and legs spread, handwritten notes around",
    11: "sollechar, comic illustration style, bust portrait, wearing a brown flat newsboy cap a vest and a red bow tie, holding a billiard cue, pool hall background",
    12: "sollechar, comic illustration style, bust portrait as a steampunk king, golden crown brass goggles and an ermine royal robe with a red cape, plain purple background",
    13: "sollechar, comic illustration style, bust portrait as the greek god zeus, golden laurel wreath halo and white toga, holding a glowing lightning bolt, sitting on a cloud, purple background",
    14: "sollechar, comic illustration style, full body wearing an orange puffer vest jeans and sneakers, parked DeLorean car and town hall at night in the background",
    15: "sollechar, bold comic illustration style, extreme close up portrait, wearing a gold suit and heavy gold chains and rings, holding a champagne flute, gold coins raining, luxurious golden interior",
    16: "sollechar, storybook illustration, full body as a painter wearing a paint stained apron, holding a palette and painting a candle on an easel canvas, messy artist studio",
    17: "sollechar, cartoon children's illustration, full body holding a colorful bouquet of flowers, surrounded by a cheering diverse crowd with balloons confetti and a party banner, a dog nearby",
    18: "sollechar, van gogh oil painting style with swirling starry brush strokes, bust portrait in blue and yellow tones",
    19: "sollechar, comic illustration style, full body as a jungle explorer wearing a safari hat backpack vest shorts and boots, holding a map, ancient overgrown ruins and a waterfall in the jungle",
    20: "sollechar, comic illustration style, full body as a storm chaser wearing yellow goggles and a black rain jacket, operating a camera on a tripod on a cliff, tornado and lightning storm with heavy rain",
    21: "sollechar, 3d render, full body wearing an orange hard hat an orange safety vest and jeans, standing in a sunny wooden workshop with a potted plant",
    22: "sollechar, photoreal cinematic 3d render, full body sitting in a brown leather armchair reading a book, a globe on a side table, classic study library room",
    23: "sollechar, comic illustration style, full body as a rio carnival dancer with colorful feather wings and headdress and a beaded necklace, dancing on a stage, festive street crowd palm trees and confetti",
    24: "sollechar, vaporwave comic illustration, full body wearing a pastel pink jacket and pants, standing at a retro gas station, vintage cars and a pastel swirling sky",
    25: "sollechar, 3d game poster render, full body with hands on hips, futuristic crypto game interface with to the moon and start game text, a rocket and neon trading charts in the background",
    26: "sollechar, photoreal 3d render, bust portrait wearing a red santa hat and a festive red christmas sweater, christmas tree and fireplace in the background",
    27: "sollechar, comic illustration style, full body covered with many colorful crypto token badge stickers, plain purple background",
    28: "sollechar, banksy stencil graffiti style, full body reaching toward a red heart shaped balloon, grey concrete wall, greyscale with a single red accent",
    29: "sollechar, comic illustration style, close up profile, holding a retro game controller, vintage tv showing a super mario bros style platformer, old game console",
    30: "sollechar, classical oil painting, bust portrait as a victorian gentleman wearing a black top hat black suit and bow tie, sitting inside an antique horse carriage",
    31: "sollechar, comic illustration style, full body lounging on a yacht deck sun lounger, wearing yellow floral swim shorts and a blue shirt, holding a cocktail, ocean view and an airplane towing a buy solle now banner",
    32: "sollechar, storybook illustration, full body reclining and lying down wearing an orange monk robe, holding an apple with a bowl of fruit, oriental temple pagoda background",
    33: "sollechar, vibrant dynamic comic illustration, full body skateboarding mid air, wearing a backwards cap a tie dye shirt knee pads and jeans, neon green futuristic city skyline with trading charts and laser lights",
    34: "sollechar, cozy children's illustration, full body sitting on ice with eyes closed, wearing a knit pom pom beanie a scarf and a puffer coat, hugging a baby penguin, arctic icebergs and falling snow",
    35: "sollechar, comic illustration style, full body as a medieval knight in silver plate armor with a red cape, holding a sword with a Solana logo on the chest, helmet visor up, plain purple background",
}


def main() -> None:
    if not DATASET.is_dir():
        raise SystemExit(f"Dataset introuvable : {DATASET.resolve()}")
    written = 0
    for idx, text in CAPTIONS.items():
        path = DATASET / f"sollechar_{idx:03d}.txt"
        if not path.with_suffix(".png").exists():
            print(f"  [WARN] image manquante pour {path.name}, j'écris quand même le .txt")
        path.write_text(text, encoding="utf-8")
        written += 1
    print(f"{written} captions réécrites dans {DATASET.resolve()}")


if __name__ == "__main__":
    main()
