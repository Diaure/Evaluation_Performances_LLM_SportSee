import os

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

from .models import PreparedDocumentModel


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY n'est pas définie."
    )


model = MistralModel(
    "mistral-small-latest",
    provider=MistralProvider(
        api_key=MISTRAL_API_KEY
    ),
)


document_preparation_agent = Agent(
    model=model,
    output_type=PreparedDocumentModel,
    instructions="""
Tu es un composant de préparation de données pour un système RAG NBA.

Ta tâche consiste à nettoyer le contenu fourni sans modifier
les informations factuelles.

Règles :
- supprimer les espaces inutiles ;
- supprimer les lignes vides inutiles ;
- normaliser les retours à la ligne ;
- conserver les noms de joueurs, équipes et statistiques ;
- conserver les valeurs numériques ;
- conserver les unités et pourcentages ;
- ne pas inventer d'information ;
- ne pas résumer le contenu ;
- ne pas supprimer une information utile à une question métier ;
- conserver les métadonnées fournies.
""",
)


def prepare_document(
    document: dict,
) -> PreparedDocumentModel:

    result = document_preparation_agent.run_sync(
        f"""
Voici le document à préparer :

{document["page_content"]}

Métadonnées :
{document.get("metadata", {})}
"""
    )

    return result.output