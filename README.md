# PROGRAMA DE RECOMENDACION DE JUEGOS, DE LA PLATAFORMA STEAM

![Steam](imagenes/steam.jpg)

## Introducción : 
Este proyecto consiste en crear una API para Steam, construyendo un modelo de recomendacion basado en Machine Learning . 
Otorgando una interfaz intuitiva para el usuario donde podra consultar datos sobre generos , fechas , sentiment score ,etc. De manera puntual.

<img src="imagenes/mando.jpg" alt="Mando" width="900" height="500">


## Tecnologías utilizadas :
- Python
- FastAPI
- Matplotlib
- NLTK
- Numpy
- Pandas
- Render
- Scikit-Learn
- Seaborn
- Uvicorn
- Wordcloud
- textblob

## Resolución :
### 1. [ETL :](ETL)

![ETL](imagenes/ETL.jpg)

Realice el proceso de ETL (extracción, transformación y carga ) con los archivos obtenidos de distintas fuentes para su posterior análisis y utilización dentro del modelo de ML.

### 2. [EDA :](EDA)

![EDA](imagenes/EDA.jpg)

El analisis exploratoria (EDA) se realizo dentro de cada ETL pero a su vez se genero un archivo llamado [EDA](EDA), en el cual se repesenta graficamente el proceso de analisis de los datos.
Este analisis nos permite identificar relaciones, tendencias y/o patrones, tal que, nos sirva para la creacion y ejecucion del modelo de ML

### 3. Deployment de la API

![API](imagenes/API.jpg)

Creamos una API utilizando el módulo FastAPI de Python, creando 5 funciones para que puedan ser consultadas:
- developer(desarrollador: str):
  Proporciona el número de ítems y el porcentaje de contenido gratuito por año para un desarrollador específico.

- userdata(User_id: str):
  Devuelve la cantidad total de dinero gastado por el usuario, el porcentaje de recomendación basado en las reseñas y la cantidad de ítems adquiridos.

- UserForGenre(genero: str):
  Encuentra el usuario con más horas jugadas para un género específico y proporciona un historial de horas jugadas por año de lanzamiento.

- best_developer_year(año: int):
  Retorna los tres desarrolladores con los juegos más recomendados por los usuarios en un año determinado.

- developer_reviews_analysis(desarrolladora: str):
  Ofrece un análisis de las reseñas de usuarios para una empresa desarrolladora, mostrando la cantidad total de registros clasificados como positivos o negativos.

Luego realizamos el deployement de esta API utilizando Render.

## Enlaces : 
- [API](https://recomendacion-de-juegos-b6zg.onrender.com/docs#/desarrollador/developer_developer_get)

- [Video explicativo](https://youtu.be/aZKEAiAjtcs?si=7kUubKvb8dHXvtjn)
