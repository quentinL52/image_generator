from crewai import Crew, Process, Task, Agent
from prompt_from_tweet.config import crew_openai

LLM = crew_openai()

tweet_analyzer = Agent(
    role="Analyste de Tweet",
    goal="Analyser le contenu d’un tweet pour identifier le sujet, l’ambiance et les éléments visuels, en se concentrant uniquement sur la crypto-monnaie Solana et ses actifs liés.",
    backstory="Expert en analyse sémantique et compréhension contextuelle. Spécialiste de l'écosystème Solana.",
    verbose=True,
    llm=LLM
)

image_prompt_writer = Agent(
    role="Rédacteur de Prompt Visuel",
    goal="""Créer un prompt descriptif et artistique pour un générateur d'images, en intégrant un personnage de référence (avec vêtements et accessoires pertinents si nécessaire) à une scène variée (paysage, intérieur thématique, etc.) basée sur l'analyse d'un tweet. Le background doit être créatif, en lien direct avec le sujet et le contexte du tweet pour une interprétation visuelle fidèle.
    """,
    backstory="Spécialiste en prompt engineering pour IA générative visuelle. Expert dans la fusion d'un personnage avec de nouvelles situations.",
    verbose=True,
    llm=LLM
)

analyze_task = Task(
    description="Analyse le tweet : {tweet_text}. Identifie le sujet, l’ambiance et les éléments visuels liés à Solana.",
    expected_output="Analyse structurée du tweet, incluant le sujet (ex: 'événement communautaire', 'développement technique', 'annonce majeure'), l'ambiance (ex: 'joyeux', 'pressé', 'sérieux', 'optimiste'), et les éléments visuels clairs (ex: 'personnages en fête', 'paysage futuriste', 'graphiques financiers', 'logos Solana'). N'inclure aucune mention d'autres blockchains.",
    agent=tweet_analyzer
)

describe_character_task = Task(
    description="Décrire le personnage de référence, un monstre violet avec de grands yeux jaunes et une expression un peu perplexe.",
    expected_output="Description détaillée du personnage: 'Un monstre violet, duveteux, avec de grands yeux exorbités aux pupilles jaunes, une bouche charnue et une expression dubitative.'",
    agent=image_prompt_writer
)

include_logo_task = Task(
    description="Décide si le logo de Solana doit être inclus dans la scène. Si l'analyse du tweet le mentionne, réponds 'Oui'. Sinon, réponds 'Non'.",
    expected_output="Réponse 'Oui' ou 'Non'.",
    agent=tweet_analyzer,
    context=[analyze_task]
)

generate_prompt_task = Task(
    description="""À partir de l'analyse du tweet et de la description du personnage, rédige un prompt clair et détaillé pour un générateur d'image.

    Consignes :
    - Fais apparaître le personnage de référence dans le prompt.
    - Intègre-le dans la scène décrite par l'analyse du tweet.
    # MISE À JOUR DE LA CONSIGNE pour l'interaction et les accessoires
    - Le personnage doit être mis en scène de manière pertinente par rapport aux éléments visuels du tweet (ex: 'le monstre violet célèbre un jalon important', 'le monstre violet se promène dans un décor inattendu'). Il peut porter des vêtements ou des accessoires pour contextualiser la scène.
    - Assure-toi que les proportions corporelles sont réalistes et cohérentes, en particulier le cou qui doit être proportionnel à la tête et aux épaules.
    - Si le logo de Solana doit être inclus (contexte de 'include_logo_task'), fais en sorte qu'il apparaisse dans le prompt.
    - Mentionne uniquement Solana et évite toute autre crypto-monnaie.
    """,
    expected_output="""Prompt d'image optimisé, incluant le personnage de référence, le sujet lié à Solana, et le logo de Solana si applicable. Par exemple : 'Le monstre violet de l'image de référence, vêtu d'une chemise hawaïenne et de lunettes de soleil, est allongé sur un transat sur une plage paradisiaque au coucher du soleil, avec le logo Solana visible sur un panneau lumineux en arrière-plan.'""",
    agent=image_prompt_writer,
    context=[analyze_task, describe_character_task, include_logo_task]
)


class TweetToImageCrew:
    def __init__(self):
        self.crew = Crew(
            agents=[tweet_analyzer, image_prompt_writer],
            tasks=[analyze_task, describe_character_task, include_logo_task, generate_prompt_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )

    def run(self, tweet_text: str) -> str:
        result = self.crew.kickoff(inputs={'tweet_text': tweet_text})

        return result