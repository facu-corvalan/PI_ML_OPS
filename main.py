from fastapi import FastAPI, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.get("/Developer", tags=["Desarrollador"])

async def developer(developer : str = Query(default='Kotoshiro')):
    """
    End point 1
    - def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free
      por año según empresa desarrolladora. Ejemplo de retorno:

    - Año	Cantidad de Items	Contenido Free
    - 2023	      50	              27%
    - 2022	      45	              25%
    - xxxx	      xx	              xx%

    """
    games = pd.read_parquet('Dataset/developer.parquet')

    # Filtramos por el desarrollador ingresado
    games = games[games['developer'] == developer]

    # Agrupamos por año y contamos la cantidad de items
    games['Año'] = games['release_date'].dt.year
    games_year = games.groupby('Año').size().reset_index(name='Cantidad de Items')

    # Agrupamos por año y sumamos la cantidad de items free
    games_free = games.groupby('Año')['free'].sum().reset_index(name='free')

    # Hacemos un merge de ambos dataframes mediante la columna Año
    games = pd.merge(games_year, games_free, on='Año')

    # Calculamos el porcentaje de contenido free
    games['Contenido Free'] = round(games['free'] / games['Cantidad de Items'] * 100, 2)

    # Eliminamos la columna free
    games.drop(columns=['free'], inplace=True)

    # Convertimos el dataframe a un diccionario
    games = games.to_dict('records')
    
    return games
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.get('/UserForGenre', tags=['Jugador por Género'])

def UserForGenre(genre: str = Query(default='Action')):
    """
    3) enpoint:

    - def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado 
      y una lista de la acumulación de horas jugadas por año de lanzamiento.

    - Ejemplo de retorno: 
    {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}
    , {Año: 2011, Horas: 23}]}
    """
    games = pd.read_parquet('Dataset/user_genre_games.parquet')
    items = pd.read_parquet('Dataset/user_genre_items.parquet')
    
    # Filtrando los juegos por el género dado
    games_genre = games[games['genres'].str.contains(genre, case=False, na=False)]
    
    # Combinando los DataFrames games e items
    filter_games = games_genre.merge(items, on='id', how='inner')
    
    # Añadiendo la columna del año de lanzamiento
    filter_games['release_year'] = filter_games['release_date'].dt.year
    
    # Agrupando por user_id y release_year para sumar el tiempo de juego
    user_year = filter_games.groupby(['id', 'release_year']).agg({'playtime_forever': 'sum'}).reset_index()
    
    # Encontrando el usuario con más horas jugadas
    total_playtime_per_user = user_year.groupby('user_id')['playtime_forever'].sum()
    max_user = total_playtime_per_user.idxmax()
    
    # Filtrando las horas jugadas por año para el usuario con más horas
    user_playtime_by_year = user_year[user_year['user_id'] == max_user][['release_year', 'playtime_forever']]
    
    # Renombrando las columnas para cumplir con el formato requerido
    user_playtime_by_year.rename(columns={'release_year': 'Año', 'playtime_forever': 'Horas jugadas'}, inplace=True)
    
    # Convirtiendo el resultado a un formato de lista de diccionarios
    year_playtime = user_playtime_by_year.to_dict('records')
    
    # Devolviendo el resultado en el formato especificado
    return {"Usuario con más horas jugadas para Género " + genre: max_user, "Horas jugadas": year_playtime}
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.get('/best_developer_year', tags=['Mejor Desarrollador'])
def best_developer_year( año : int= Query(default='2000' )):
    """
    4) enpoint:
    - def best_developer_year( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
    - Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
    """
    games = pd.read_parquet('Dataset/best_developer_games.parquet')
    reviews = pd.read_parquet('Dataset/best_developer_reviews.parquet')

    # Filtrar juegos por el año especificado
    games['release_year'] = games['release_date'].dt.year
    games_filtered = games[games['release_year'] == año]

    # Combinar juegos filtrados con reseñas para obtener solo los datos relevantes
    filtered_reviews = games_filtered.merge(reviews, left_on='id', right_on='item_id', how='inner')
    filtered_reviews = filtered_reviews[['developer', 'title', 'positivo']]

    # Agrupar por desarrollador y sumar los comentarios positivos
    filtered_reviews = filtered_reviews.groupby(['developer']).agg({'positivo': 'sum'}).reset_index()

    # Ordenar desarrolladores por la suma de comentarios positivos en orden descendente
    filtered_reviews = filtered_reviews.sort_values(by='positivo', ascending=False)

    # Obtener los tres principales desarrolladores
    top_developers = filtered_reviews.head(3)['developer'].tolist()

    return [
        {"Puesto 1": top_developers[0]},
        {"Puesto 2": top_developers[1]},
        {"Puesto 3": top_developers[2]}]
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
