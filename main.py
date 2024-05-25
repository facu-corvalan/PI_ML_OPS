from fastapi import FastAPI, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

@app.get("/developer", tags=["desarrollador"])
#games = pd.read_parquet('Dataset/games_clean.parquet')
async def developer(developer : str = Query(default='Valve')):
    # Cargamos el dataset.
    games = pd.read_parquet('Dataset/developer.parquet')
    # Filtramos por el desarrollador ingresado.
    games = games[games['developer'] == developer]
    # Creamos la columna año en base a la columna release_date.
    games['Año'] = games['release_date'].dt.year
    # Agrupamos por año y contamos la cantidad de items.
    games_year = games.groupby('Año').size().reset_index(name='Cantidad de Items')
    # Agrupamos por año y sumamos la cantidad de items free.
    games_free = games.groupby('Año')['free'].sum().reset_index(name='free')
    # Hacemos un merge de ambos dataframes mediante la columna Año.
    games = pd.merge(games_year, games_free, on='Año')
    # Calculamos el porcentaje de contenido free.
    games['Contenido Free'] = round(games['free'] / games['Cantidad de Items'] * 100, 2)
    # Eliminamos la columna free.
    games.drop(columns=['free'], inplace=True)
    # Convertimos el dataframe a un diccionario.
    games = games.to_dict('records')
    # Devolvemos el resultado.
    return games
