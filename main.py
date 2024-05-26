from fastapi import FastAPI, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.get("/developer", tags=["Desarrollador"])

async def developer(developer : str = Query(default='Bohemia Interactive')):
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

    # Agrupamos por release_year y contamos la cantidad de items
    games['release_year'] = games['release_date'].dt.year
    games_year = games.groupby('release_year').size().reset_index(name='Cantidad de Items')

    # Agrupamos por release_year y sumamos la cantidad de items free
    games_free = games.groupby('release_year')['free'].sum().reset_index(name='free')

    # Hacemos un merge de ambos dataframes mediante la columna release_year
    games = pd.merge(games_year, games_free, on='release_year')

    # Calculamos el porcentaje de contenido free
    games['Contenido Free'] = round(games['free'] / games['Cantidad de Items'] * 100, 2)

    # Eliminamos la columna free
    games.drop(columns=['free'], inplace=True)

    # Convertimos el dataframe a un diccionario
    games = games.to_dict('records')
    
    return games
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.get('/userdata', tags=['Informacion del Jugador'])

def userdata( User_id : str ):
    """
    def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
    Ejemplo de retorno: {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}
    """

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.get('/User_For_Genre', tags=['Jugador por Género'])

def UserForGenre(genre: str = Query(default='Action')):
    """
    Endpoint para obtener el usuario con más horas jugadas y la acumulación de horas jugadas por año de lanzamiento para un género dado.

    Ejemplo de retorno:
    {
        "Usuario con más horas jugadas para Género X": "us213ndjss09sdf",
        "Horas jugadas": [
            {"Año": 2013, "Horas jugadas": 203},
            {"Año": 2012, "Horas jugadas": 100},
            {"Año": 2011, "Horas jugadas": 23}
        ]
    }
    """
    games = pd.read_parquet('Dataset/user_genre_games.parquet')
    items = pd.read_parquet('Dataset/user_genre_items.parquet')
    # Filtrar los juegos por el género dado
    games_genre = games[games['genres'].str.contains(genre, case=False, na=False)]
    
    # Combinar los DataFrames de juegos e items
    filter_games = games_genre.merge(items, on='id', how='inner')
    
    # Añadir la columna del año de lanzamiento
    filter_games['release_year'] = filter_games['release_date'].dt.year
    
    # Agrupar por user_id y release_year para sumar el tiempo de juego
    user_year = filter_games.groupby(['user_id', 'release_year'])['playtime_forever'].sum().reset_index()
    
    # Encontrar el usuario con más horas jugadas
    max_user = user_year.loc[user_year['playtime_forever'].idxmax(), 'user_id']
    
    # Filtrar las horas jugadas por año para el usuario con más horas
    user_playtime_by_year = user_year[user_year['user_id'] == max_user][['release_year', 'playtime_forever']]
    
    # Renombrar las columnas para cumplir con el formato requerido
    user_playtime_by_year.rename(columns={'release_year': 'Año', 'playtime_forever': 'Horas jugadas'}, inplace=True)
    
    # Convertir el resultado a un formato de lista de diccionarios
    year_playtime = user_playtime_by_year.to_dict('records')
    
    return {
        "Usuario con más horas jugadas para Género " + genre: max_user,
        "Horas jugadas": year_playtime
    }


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

@app.get('/developer_reviews_analysis', tags=['Reseña de Desarrolladores'])

def developer_reviews_analysis(desarrolladora: str = Query(default='Bohemia Interactive')):
    """
    Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave
    y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados
    con un análisis de sentimiento como valor positivo o negativo.

    Ejemplo de retorno: {'Valve': {'Negativas': 182, 'Positivas': 278}}
    """
    games = pd.read_parquet('Dataset/developer_review_games.parquet')  # Cargar datos de juegos
    reviews = pd.read_parquet('Dataset/developer_review_reviews.parquet')  # Cargar datos de reseñas

    # Filtrar los juegos por el desarrollador especificado
    developer_games = games[games['developer'] == desarrolladora]
    
    # Combinar datos de juegos y reseñas basado en el id del juego
    merged_data = developer_games.merge(reviews, left_on='id', right_on='item_id', how='inner')

    # Verificar si no hay datos disponibles para el desarrollador
    if merged_data.empty:
        return {desarrolladora: {'Negativas': 0, 'Positivas': 0}}

    # Calcular el total de reseñas negativas y positivas
    total_negativas = merged_data['negativo'].sum()
    total_positivas = merged_data['positivo'].sum()

    # Retornar el resultado en el formato esperado
    return {desarrolladora: {'Negativas': int(total_negativas), 'Positivas': int(total_positivas)}}

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

