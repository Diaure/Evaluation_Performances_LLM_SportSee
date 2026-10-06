import os
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase

from mistralai.client import Mistral

# chargement clé api et model
load_dotenv()
from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,)

# connexion à posgresql
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
datababe = SQLDatabase(engine, include_tables=["teams", "players", "matches", "stats", "top_15_joueurs_points"]) # permettre à langchain de récupérer les bases sql pour fournir au llm les informations nécéssaires pour generer les réponses

# LLM Mistral
client = Mistral(api_key=MISTRAL_API_KEY)

# exemples few-shot pour améliorer la précision des templates SQL
exemples_few_shot = """
exemple 1
Question :
Quel joueur a marqué le plus de points?
SQL:
SELECT
    p.nom_du_joueur,
    t.nombre_points_total
FROM top_15_joueurs_points t
JOIN players p ON t.player_id = p.player_id
ORDER BY t.nombre_points_total DESC
LIMIT 1;

exemple 2
Question :
Quelle équipe a marqué le plus de points?
SQL:
SELECT
    nom_complet_equipe,
    nombre_points_total_par_equipe
FROM teams
ORDER BY nombre_points_total_par_equipe DESC
LIMIT 1;

exemple 3
Question :
Quels joueurs ont marqué plus de 20 points par match?
SQL:
SELECT
    p.nom_du_joueur,
    m.points_moyens
FROM players p
JOIN matches m ON p.player_id = m.player_id
WHERE m.points_moyens > 20
ORDER BY m.points_moyens DESC;

exemple 4
Quel est le pourcentage de réussite aux tirs de Jayson Tatum?
SQL:
SELECT
    p.nom_du_joueur,
    m.pourcentage_reussite
FROM players p
JOIN matches m ON p.player_id = m.player_id
WHERE p.nom_du_joueur like '%Tatum%';

exemple 5
Question:
Quel joueur a le meilleur pourcentage à 3 points?
SQL:
SELECT
    p.nom_du_joueur,
    s.pourcentage_3_points
FROM players p
JOIN stats s ON p.player_id = s.player_id
WHERE s.pourcentage_3_points = (
    SELECT MAX(pourcentage_3_points)
    FROM stats)
ORDER BY p.nom_du_joueur;

exemple 6
Question:
Quelles sont les cinq équipes qui ont réalisé le plus d'interceptions au total?
SQL:
SELECT
    t.nom_complet_equipe,
    sum(s.interceptions) as Total_interception
FROM teams t
JOIN players p ON p.team_id = t.team_id
JOIN stats s ON p.player_id = s.player_id
GROUP BY t.nom_complet_equipe
ORDER BY sum(s.interceptions) DESC
LIMIT 5;

exemple 7
Compare les performances de Jayson Tatum et Nikola Jokić en nombre de matchs, points, rebonds, passes décisives et possessions.
SQL:
SELECT
    p.nom_du_joueur,
    sum(m.nombre_matchs_joues) as Nb_de_matche,
	sum(m.points_moyens) as Nb_de_points,
	sum(m.tirs_reussis) as Nb_de_tirs_reussis,
	sum(s.rebonds_totaux) as N_de_rebonds_total,
	sum(s.passes_decisives) as Nb_de_asses_decisives,
	sum(s.possessions_totales) as Nb_de_possession
FROM players p
JOIN matches m ON p.player_id = m.player_id
JOIN stats s ON p.player_id = s.player_id
WHERE p.nom_du_joueur like '%Tatum%' or p.nom_du_joueur like '%Jokić%'
group by p.nom_du_joueur;

exemple 8
Quel est le joueur le plus jeune et le joueur le plus âgé?
SQL:
SELECT
    nom_du_joueur,
    age_du_joueur
FROM players
WHERE age_du_joueur = (SELECT MIN(age_du_joueur) FROM players)
   OR age_du_joueur = (SELECT MAX(age_du_joueur) FROM players)
ORDER BY age_du_joueur;
"""

# prompt pour générer les requêtes sql
def prompt_construction_sql(question: str) -> str:
    schema = datababe.get_table_info()
    return f"""
Tu es un assistant spécialisé dans la génération de requêtes SQL PostgreSQL pour une base de données contenant des données NBA.
Ta tâche consiste à transformer la question de l'utilisateur en une requête SQL valide.

Règles:
1. Génère uniquement une requête SQL SELECT
2. Ne génère jamais INSERT, UPDATE, DELETE, DROP, ALTER, CREATE ou TRUNCATE
3. Utilise uniquement les tables et colonnes présentes dans le schéma fourni
4. Utilise les clés étrangères pour effectuer les jointures
5. N'invente jamais de colonne
6. Lorsque tu demandes des informations sur un joueur, utilise players.nom_du_joueur
7. Les statistiques des joueurs sont dans players
7. Les statistiques de matchs sont dans matches
8. Les statistiques détaillées sont dans stats
9. Les informations d'équipe sont dans teams
10. Le classement des 15 meilleurs marqueurs est dans top_15_joueurs_points
11. Retourne uniquement le SQL, sans explication, sans markdown et sans ```sql.

SCHÉMA DE LA BASE :
{schema}

EXEMPLES FEW-SHOT :
{exemples_few_shot}

QUESTION UTILISATEUR :
{question}

REQUÊTE SQL :
"""

# récupérer la requete sql de la réponse
def nettoyage_sql (sql: str) -> str:
    sql = sql.strip()
    
    # suppression éventuelle des blocs Markdown
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)
    sql = sql.strip()

    return sql

# validation de la requete
def validation_requete(sql: str) -> bool:
    sql_clean = sql.strip()
    if not sql_clean:
        return False

    # La requête doit commencer par SELECT ou WITH
    if not re.match(
        r"^(SELECT|WITH)\b",
        sql_clean,
        flags=re.IGNORECASE):
        return False

    # Commandes interdites
    forbidden_commands = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE"]

    for command in forbidden_commands:
        if re.search(
            rf"\b{command}\b",
            sql_clean,
            flags=re.IGNORECASE):
            return False
    return True

# génération sql à partir de la question de l'utilisateur par le lllm
def generer_sql(question: str) -> str:
    prompt = prompt_construction_sql(question)
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[
            {"role": "system",
            "content": (
                "Tu es un expert SQL PostgreSQL. "
                "Tu génères uniquement des requêtes SELECT.")},

            {"role": "user",
            "content": prompt}],
        temperature=0)

    sql = response.choices[0].message.content
    sql = nettoyage_sql(sql)
    if not validation_requete(sql):
        raise ValueError("La requête générée n'est pas une requête SELECT valide.")

    return sql

# execution de la requete sql
def execute_sql(query: str):
    if not validation_requete(query):
        raise ValueError("Seules les requêtes SELECT sont autorisées.")

    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        columns = result.keys()

        return [dict(zip(columns, row)) for row in rows]

# transformation de la question en requete sql par langchain
@tool
def sql_tool(question: str) -> str:
    """
    Transforme une question utilisateur en requête SQL,
    exécute cette requête dans PostgreSQL et retourne
    les résultats.
    """

    try:
        # Génération du SQL
        sql_query = generer_sql(question)

        print("\n========== SQL GÉNÉRÉ ==========")
        print(sql_query)
        print("================================")

        # Exécution
        results = execute_sql(sql_query)

        # Aucun résultat
        if not results:
            return (
                f"Requête SQL exécutée :\n"
                f"{sql_query}\n\n"
                "Aucun résultat.")

        # Transformation des résultats
        formatted_results = "\n".join(str(row) for row in results)
        return (
            f"Requête SQL exécutée :\n"
            f"{sql_query}\n\n"
            f"Résultats :\n"
            f"{formatted_results}")

    except Exception as e:
        return (
            f"Erreur lors de l'utilisation du SQL Tool :\n"
            f"{str(e)}")

# main
if __name__ == "__main__":

    question = input(
        "\nPose ta question sur les données NBA : ")

    result = sql_tool.invoke({"question": question})

    print("\n========== RÉSULTAT ==========")
    print(result)
    print("==============================")