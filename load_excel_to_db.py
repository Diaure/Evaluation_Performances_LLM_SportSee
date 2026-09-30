import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pydantic import BaseModel, ValidationError

from utils.pydantic_validation import (
    Team,
    Player,
    Match,
    Stats,
    Top15Player)

# Chargement des variables d'environnement
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Lecture du fichier Excel
excel_path = os.path.join("inputs", "regular NBA.xlsx")

# Lecture du dictionnaire de données (en-têtes à la ligne 2)
dict_df = pd.read_excel(excel_path, sheet_name="Dictionnaire des données", header=None)
dict_df.columns = ["code", "definition"]

# Création du mapping code → nom français
mapping = dict(zip(dict_df["code"], dict_df["definition"]))

# Lecture du fichier principal Données NBA
df = pd.read_excel(excel_path, sheet_name="Données NBA", header=1)
df.columns = df.columns.map(str)

# Suppression des colonnes Unnamed (vides)
df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]

# Renommer des colonnes selon le dictionnaire
df.rename(columns=mapping, inplace=True)

# Uniformisation des noms de colonnes (remplace espaces par underscores)
df.columns = df.columns.map(str).str.strip().str.replace(" ", "_")
print(df.columns)


# Lecture de l’onglet Analyse (A6:D36) → table teams
teams_df = pd.read_excel(excel_path, sheet_name="Analyse", usecols="A:D", skiprows=5, nrows=30)
teams_df.columns = [
    "code_equipe",
    "nom_complet_equipe",
    "nombre_joueurs_par_equipe",
    "nombre_points_total_par_equipe"]

# Lecture de l’onglet Analyse (A89:H104) → table top_15_joueurs_points
top_15_joueurs_points_df = pd.read_excel(excel_path, sheet_name="Analyse", usecols="A:H", skiprows=88, nrows=15)
top_15_joueurs_points_df.columns = [
    "nom_du_joueur",
    "nombre_points_total",
    "tirs_reussis",
    "pourcentage_tirs_reussis",
    "pourcentage_tirs_3_points",
    "pourcentage_lancers_francs",
    "rebonds_offensifs",
    "Estimation de l’impact du joueur"]

# Construction table matches
matches_cols = [
    "Nom_du_joueur",
    "Nombre_de_matchs_joués_(Games_Played)",
    "Nombre_de_victoires_de_l'équipe_lors_des_matchs_joués",
    "Nombre_de_défaites",
    "Minutes_moyennes_jouées_par_match",
    "Points_marqués_en_moyenne_par_match",
    "Tirs_réussis_par_match_(Field_Goals_Made)",
    "Tirs_tentés_par_match_(Field_Goals_Attempted)",
    "Pourcentage_de_réussite_aux_tirs",
    "15:00:00",
    "Tirs_à_3_points_tentés_par_match"]

matches_df = df[matches_cols].copy()
matches_df.columns = [
    "nom_du_joueur",
    "nombre_matchs_joues",
    "victoires",
    "defaites",
    "minutes_moyennes",
    "points_moyens",
    "tirs_reussis",
    "tirs_tentes",
    "pourcentage_reussite",
    "minutes_apres_15",
    "tirs_3_points_tentes"]

# Construction table stats
stats_cols = [
    "Nom_du_joueur",
    "Pourcentage_de_réussite_à_3_points",
    "Lancers_francs_réussis_(Free_Throws_Made)",
    "Lancers_francs_tentés",
    "Pourcentage_de_réussite_aux_lancers_francs",
    "Rebonds_offensifs",
    "Rebonds_défensifs",
    "Rebonds_totaux",
    "Passes_décisives_(Assists)",
    "Balles_perdues_(Turnovers)",
    "Interceptions_(Steals)",
    "Contres_(Blocks)",
    "Fautes_personnelles",
    "Fantasy_Points",
    "Double-doubles_(≥10_dans_deux_catégories_principales)",
    "Triple-doubles_(≥10_dans_trois_catégories_principales)",
    "+/-",
    "Offensive_Rating_(points_marqués_par_100_possessions)",
    "Defensive_Rating_(points_encaissés_par_100_possessions)",
    "Net_Rating_=_OFFRTG_-_DEFRTG",
    "Pourcentage_d'assists_–_implication_dans_les_passes_décisives",
    "Ratio_passes_/_pertes_de_balle",
    "Ratio_d’assists_pour_100_possessions",
    "Pourcentage_de_rebonds_offensifs_parmi_ceux_disponibles",
    "Idem_en_défensif",
    "Pourcentage_de_rebonds_totaux_parmi_ceux_disponibles",
    "Turnover_Ratio_–_pertes_de_balle_par_100_possessions",
    "Effective_Field_Goal_%_(pondère_les_3_points)",
    "True_Shooting_%_(inclut_FG_et_FT_dans_l'efficacité)",
    "Usage_Rate_–_%_des_actions_utilisées_par_le_joueur",
    "Rythme_de_jeu_(possessions_par_48_minutes)",
    "Player_Impact_Estimate_–_évaluation_globale_de_l’impact",
    "Nombre_total_de_possessions_jouées"]

stats_df = df[stats_cols].copy()
stats_df.columns = [
    "nom_du_joueur",
    "pourcentage_3_points",
    "lancers_francs_reussis",
    "lancers_francs_tentes",
    "pourcentage_lancers_francs",
    "rebonds_offensifs",
    "rebonds_defensifs",
    "rebonds_totaux",
    "passes_decisives",
    "balles_perdues",
    "interceptions",
    "contres",
    "fautes_personnelles",
    "fantasy_points",
    "double_doubles",
    "triple_doubles",
    "plus_minus",
    "offensive_rating",
    "defensive_rating",
    "net_rating",
    "pourcentage_assists",
    "ratio_assists_pertes",
    "ratio_assists_100_possessions",
    "pourcentage_rebonds_offensifs",
    "pourcentage_rebonds_defensifs",
    "pourcentage_rebonds_totaux",
    "turnover_ratio",
    "efg_percent",
    "true_shooting_percent",
    "usage_rate",
    "rythme_de_jeu",
    "player_impact_estimate",
    "possessions_totales"]


# Validation Pydantic
def validate_dataframe(df, model_class, table_name):
    validated_data = []
    for index, row in df.iterrows():
        try:
            model = model_class(**row.to_dict())
            validated_data.append(model.model_dump())
        except ValidationError as e:
            print(
                f"\nERREUR DE VALIDATION dans {table_name}, "
                f"ligne {index} :")
            print(e)
            raise
    print(
        f"Validation Pydantic réussie pour {table_name} : "
        f"{len(validated_data)} ligne(s).")
    return pd.DataFrame(validated_data)

# préparation et validation pydantic joueurs
players_df = df[
    [
        "Nom_du_joueur",
        "Équipe_du_joueur_(code_à_3_lettres)",
        "Âge_du_joueur"
    ]
].copy()

players_df.rename(
    columns={
        "Nom_du_joueur": "nom_du_joueur",
        "Équipe_du_joueur_(code_à_3_lettres)": "equipe_du_joueur",
        "Âge_du_joueur": "age_du_joueur"
    },
    inplace=True
)

players_df = validate_dataframe(
    players_df,
    Player,
    "players"
)


# Création des tables
with engine.begin() as conn:
    conn.execute(text("""
        DROP TABLE IF EXISTS matches, stats, players, teams, top_15_joueurs_points, top_15_temp CASCADE;
    """))

    # Table teams
    conn.execute(text("""
        CREATE TABLE teams (
            team_id SERIAL PRIMARY KEY,
            code_equipe CHAR(3) UNIQUE NOT NULL,
            nom_complet_equipe TEXT,
            nombre_joueurs_par_equipe INT,
            nombre_points_total_par_equipe INT);"""))

    # Table players
    conn.execute(text("""
        CREATE TABLE players (
            player_id SERIAL PRIMARY KEY,
            team_id INT,
            nom_du_joueur TEXT,
            equipe_du_joueur CHAR(3),
            age_du_joueur INT,

            CONSTRAINT fk_team
                FOREIGN KEY (team_id)
                REFERENCES teams(team_id));"""))

    # Table matches
    conn.execute(text("""
        CREATE TABLE matches (
            match_id SERIAL PRIMARY KEY,
            player_id INT,
            nombre_matchs_joues INT,
            victoires INT,
            defaites INT,
            minutes_moyennes FLOAT,
            points_moyens FLOAT,
            tirs_reussis FLOAT,
            tirs_tentes FLOAT,
            pourcentage_reussite FLOAT,
            minutes_apres_15 FLOAT,
            tirs_3_points_tentes FLOAT,

            CONSTRAINT fk_player
                FOREIGN KEY (player_id)
                REFERENCES players(player_id));"""))

    # Table stats
    conn.execute(text("""
        CREATE TABLE stats (
            stat_id SERIAL PRIMARY KEY,
            player_id INT,
            pourcentage_3_points FLOAT,
            lancers_francs_reussis FLOAT,
            lancers_francs_tentes FLOAT,
            pourcentage_lancers_francs FLOAT,
            rebonds_offensifs FLOAT,
            rebonds_defensifs FLOAT,
            rebonds_totaux FLOAT,
            passes_decisives FLOAT,
            balles_perdues FLOAT,
            interceptions FLOAT,
            contres FLOAT,
            fautes_personnelles FLOAT,
            fantasy_points FLOAT,
            double_doubles INT,
            triple_doubles INT,
            plus_minus FLOAT,
            offensive_rating FLOAT,
            defensive_rating FLOAT,
            net_rating FLOAT,
            pourcentage_assists FLOAT,
            ratio_assists_pertes FLOAT,
            ratio_assists_100_possessions FLOAT,
            pourcentage_rebonds_offensifs FLOAT,
            pourcentage_rebonds_defensifs FLOAT,
            pourcentage_rebonds_totaux FLOAT,
            turnover_ratio FLOAT,
            efg_percent FLOAT,
            true_shooting_percent FLOAT,
            usage_rate FLOAT,
            rythme_de_jeu FLOAT,
            player_impact_estimate FLOAT,
            possessions_totales FLOAT,

            CONSTRAINT fk_player_stats
                FOREIGN KEY (player_id)
                REFERENCES players(player_id));"""))

    # Table top_15_joueurs_points
    conn.execute(text("""
        CREATE TABLE top_15_joueurs_points (
            player_id INT PRIMARY KEY,
            nom_du_joueur TEXT,
            nombre_points_total FLOAT,
            tirs_reussis FLOAT,
            pourcentage_tirs_reussis FLOAT,
            pourcentage_tirs_3_points FLOAT,
            pourcentage_lancers_francs FLOAT,
            rebonds_offensifs FLOAT,
            impact_estime FLOAT,

            CONSTRAINT fk_top15_player
                FOREIGN KEY (player_id)
                REFERENCES players(player_id));"""))

# validation pydantic teams
teams_df = validate_dataframe(
    teams_df,
    Team, 
    "teams"
)
teams_df.to_sql(
    "teams",
    engine,
    if_exists="append",
    index=False)

# validation pydantic players
players_df = validate_dataframe(
    players_df,
    Player,
    "players"
)
players_df.to_sql(
    "players",
    engine,
    if_exists="append",
    index=False)

# Liaison des joueurs aux équipes
with engine.begin() as conn:
    conn.execute(text("""
        UPDATE players
        SET team_id = teams.team_id
        FROM teams
        WHERE players.equipe_du_joueur = teams.code_equipe;"""))

# vérifier si tous les joueurs ont une équipe
with engine.connect() as conn:

    result = conn.execute(text("""
        SELECT COUNT(*)
        FROM players
        WHERE team_id IS NULL;"""))
    joueurs_sans_equipe = result.scalar()

    if joueurs_sans_equipe > 0:
        print(
            f"ATTENTION : {joueurs_sans_equipe} joueur(s) "
            "n'ont pas été associés à une équipe.")
    else:
        print("Tous les joueurs ont été associés à une équipe.")

# récupérer des id de joueurs
players_mapping = pd.read_sql(
    """
    SELECT
        player_id,
        nom_du_joueur
    FROM players;""", engine)
print(f"Nombre de joueurs dans PostgreSQL : {len(players_mapping)}")

matches_df = validate_dataframe(
    matches_df,
    Match,
    "matches"
)
matches_df = matches_df.merge(players_mapping, on="nom_du_joueur", how="left")
# Vérification
if matches_df["player_id"].isna().any():
    nb = matches_df["player_id"].isna().sum()
    print(
        f"ATTENTION : {nb} ligne(s) de matches "
        "n'ont pas trouvé de player_id.")

matches_df = matches_df[[
        "player_id",
        "nom_du_joueur",
        "nombre_matchs_joues",
        "victoires",
        "defaites",
        "minutes_moyennes",
        "points_moyens",
        "tirs_reussis",
        "tirs_tentes",
        "pourcentage_reussite",
        "minutes_apres_15",
        "tirs_3_points_tentes"]] 

matches_insert = matches_df.drop(columns=["nom_du_joueur"]) 
matches_insert.to_sql(
    "matches", 
    engine, 
    if_exists="append", 
    index=False)

stats_df = validate_dataframe(
    stats_df,
    Stats,
    "stats"
)
stats_df = stats_df.merge(
    players_mapping,
    on="nom_du_joueur",
    how="left")

# Vérification
if stats_df["player_id"].isna().any():
    nb = stats_df["player_id"].isna().sum()
    print(
        f"ATTENTION : {nb} ligne(s) de stats "
        "n'ont pas trouvé de player_id.")

# On place player_id en première colonne
stats_df = stats_df[[
        "player_id",
        "nom_du_joueur",
        "pourcentage_3_points",
        "lancers_francs_reussis",
        "lancers_francs_tentes",
        "pourcentage_lancers_francs",
        "rebonds_offensifs",
        "rebonds_defensifs",
        "rebonds_totaux",
        "passes_decisives",
        "balles_perdues",
        "interceptions",
        "contres",
        "fautes_personnelles",
        "fantasy_points",
        "double_doubles",
        "triple_doubles",
        "plus_minus",
        "offensive_rating",
        "defensive_rating",
        "net_rating",
        "pourcentage_assists",
        "ratio_assists_pertes",
        "ratio_assists_100_possessions",
        "pourcentage_rebonds_offensifs",
        "pourcentage_rebonds_defensifs",
        "pourcentage_rebonds_totaux",
        "turnover_ratio",
        "efg_percent",
        "true_shooting_percent",
        "usage_rate",
        "rythme_de_jeu",
        "player_impact_estimate",
        "possessions_totales"]]

# Suppression du nom avant insertion
stats_insert = stats_df.drop(columns=["nom_du_joueur"])
stats_insert.to_sql(
    "stats",
    engine,
    if_exists="append",
    index=False)

top_15_joueurs_points_df = top_15_joueurs_points_df.rename(columns={"Estimation de l’impact du joueur": "impact_estime"})
top_15_joueurs_points_df = validate_dataframe(
    top_15_joueurs_points_df,
    Top15Player,
    "top_15_joueurs_points"
)
top_15_joueurs_points_df = top_15_joueurs_points_df.merge(
    players_mapping,
    on="nom_du_joueur",
    how="left")

# Vérification
if top_15_joueurs_points_df["player_id"].isna().any():
    nb = top_15_joueurs_points_df["player_id"].isna().sum()
    print(
        f"ATTENTION : {nb} joueur(s) du top 15 "
        "n'ont pas trouvé de player_id.")
    
top_15_insert = top_15_joueurs_points_df[[
        "player_id",
        "nom_du_joueur",
        "nombre_points_total",
        "tirs_reussis",
        "pourcentage_tirs_reussis",
        "pourcentage_tirs_3_points",
        "pourcentage_lancers_francs",
        "rebonds_offensifs",
        "impact_estime"]].copy()

top_15_insert.to_sql(
    "top_15_joueurs_points",
    engine,
    if_exists="append",
    index=False)

# check final
with engine.connect() as conn:
    nb_players = conn.execute(text("""SELECT COUNT(*) FROM players;""")).scalar()
    nb_matches = conn.execute(text("""SELECT COUNT(*) FROM matches;""")).scalar()
    nb_stats = conn.execute(text("""SELECT COUNT(*) FROM stats;""")).scalar()
    nb_top15 = conn.execute(text("""SELECT COUNT(*) FROM top_15_joueurs_points;""")).scalar()

    print(f"Nombre de joueurs : {nb_players}")
    print(f"Nombre de lignes matches : {nb_matches}")
    print(f"Nombre de lignes stats : {nb_stats}")
    print(f"Nombre de joueurs dans le top 15 : {nb_top15}")

print("Toutes les données Excel ont été insérées correctement dans PostgreSQL.")
