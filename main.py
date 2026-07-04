import re
import unicodedata
import pandas as pd
import numpy as np
import spacy
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Cargar el modelo de spaCy para español
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("El modelo de spaCy no está instalado. Ejecuta en tu terminal:")
    print("python -m spacy download es_core_news_sm")
    exit()

# Carga del dataset corte 1
df_entrada = pd.read_csv("dataset/main_poetry_seleccionado.csv")

# Stop words pueden aumentar o decrementar
# depende de la evolucion del proyecto
STOPWORDS_ES = [
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "y", "o", "pero", "aunque", "a", "en", "con",
    "por", "para", "que", "se", "al", "lo", "es", "fue", "muy",
    "mas", "menos", "durante", "desde", "sobre", "sin", "su", "sus",
]

# Diccionarios de clasificación por lemas
DICCIONARIOS = {
    "Amoroso": {"amor", "deseo", "belleza", "corazon", "labio", "pasion", "querer", "amar", "hermoso"},
    "Religioso Mistico": {"dios", "cielo", "santo", "alma", "fe", "divino", "rezar", "cristo", "pecado", "sacro"},
    "Melancolico": {"dolor", "muerte", "llanto", "perdida", "tiempo", "fugaz", "triste", "llorar", "duelo","fatal","parca","temor"},
    "Heroico": {"espada", "rey", "victoria", "sangre", "patria", "guerra", "honor", "triunfo", "batalla","belico","gloria"},
    "Satira": {"burla", "risa", "necio", "dinero", "engaño", "hipocresia", "ironia", "comico", "torpe"}
}

def quitar_acentos(texto: str) -> str:
    # Se elminan acentos
    normalizado = unicodedata.normalize("NFD", texto)
    return "".join(caracter for caracter in normalizado if unicodedata.category(caracter) != "Mn")

def limpiar_texto(texto: str) -> str:
    # Limpieza de texto, se eliminan caracteres especiales
    # Se quitan los saltos de linea
    texto = str(texto).lower()
    texto = quitar_acentos(texto)
    
    # Se quitan saltos de linea para tener el formato estandar de CSV
    texto = texto.replace("\n", " ").replace("\r", " ")
    
    # Quitar todo lo que no sea letras, eñes o numeros
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    
    # Quitar espacios dobles o triples consecutivos
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def remover_stopwords(texto_limpio: str) -> str:
    # Se realiza la tokenizacion del texto, se agrega un espacio en lugar del stop words
    palabras = texto_limpio.split()
    palabras_filtradas = [w for w in palabras if w not in STOPWORDS_ES]
    return " ".join(palabras_filtradas)

def clasificar_poema_lexico(texto: str) -> str:
    # Validar que el texto sea procesable
    if not isinstance(texto, str) or not texto.strip():
        return "Sin Clasificar"

    # Procesar el texto con spaCy
    doc = nlp(texto)
    
    # Extraer los lemas (raíces de las palabras) ignorando puntuación/números
    lemas_texto = {token.lemma_ for token in doc if token.is_alpha}
    
    # Inicializar contador de coincidencias
    puntuaciones = {categoria: 0 for categoria in DICCIONARIOS}
    
    for categoria, palabras_clave in DICCIONARIOS.items():
        # Contar cuántas palabras del diccionario aparecen en los lemas del texto
        coincidencias = lemas_texto.intersection(palabras_clave)
        puntuaciones[categoria] = len(coincidencias)
    
    # Encontrar la categoría con más coincidencias
    categoria_ganadora = max(puntuaciones, key=puntuaciones.get)
    
    # Si ninguna palabra del diccionario coincide, se marca para revisión
    if puntuaciones[categoria_ganadora] == 0:
        return "Requiere Revision Manual"
        
    return categoria_ganadora

def representar_texto(ruta_entrada):
    
    # Cargar el dataset etiquetado
    df = pd.read_csv(ruta_entrada)
    
    # Se eliminan valores nulos
    df["tokens_utiles"] = df["tokens_utiles"].fillna("")

    # Crear un directorio para guardar los CSV generados
    if not os.path.exists("matrices_csv"):
        os.makedirs("matrices_csv")

    # BoW
    print("Generando Bolsa de Palabras\n")
    vectorizador_bow = CountVectorizer()
    matriz_bow = vectorizador_bow.fit_transform(df["tokens_utiles"])
    
    # El BoW se convierte a DF y se guarda en CSV
    vocabulario_bow = vectorizador_bow.get_feature_names_out()
    df_bow = pd.DataFrame(matriz_bow.toarray(), columns=vocabulario_bow)
    df_bow.to_csv("matrices_csv/matriz_bow.csv", index=False, encoding="utf-8")
    print("Archivo 'matriz_bow.csv' guardado.\n\n")

    # TF-IDF
    print("\nGenerando TF-IDF")
    vectorizador_tfidf = TfidfVectorizer()
    matriz_tfidf = vectorizador_tfidf.fit_transform(df["tokens_utiles"])
    
    # TF-IDF se convierte en DF y se guarda en CSV
    vocabulario_tfidf = vectorizador_tfidf.get_feature_names_out()
    df_tfidf = pd.DataFrame(matriz_tfidf.toarray(), columns=vocabulario_tfidf)
    df_tfidf.to_csv("matrices_csv/matriz_tfidf.csv", index=False, encoding="utf-8")
    print("Archivo 'matriz_tfidf.csv' guardado.\n\n")
    
    return matriz_tfidf, df["etiqueta"]

def entrenar_svm(X, y, df):
    
    mascara_validos = y != "Requiere Revision Manual"
    mascara_numpy = mascara_validos.to_numpy()
    
    X_filtrado = X[mascara_numpy]
    y_filtrado = y[mascara_validos]
    
    # Filtramos el DataFrame original para mantener alineados los textos
    # asi se evita guardar una prediccion junto a la oracion que no le corresponde
    df_filtrado = df[mascara_numpy].reset_index(drop=True)
    
    print(f"Total de registros originales: {X.shape[0]}")
    print(f"Registros útiles para entrenamiento (con etiqueta real): {X_filtrado.shape[0]}")

    
    indices = np.arange(len(df_filtrado))
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_filtrado, y_filtrado, indices, test_size=0.2, random_state=42, stratify=y_filtrado
    )
    
    # Definimos la grid search
    param_grid = {
        'C': [0.1, 1, 10, 50],             
        'kernel': ['linear', 'rbf'],       
        'class_weight': [None, 'balanced'] 
    }
    
    
    print("\nINICIA GRID SEARCH\n")
    
    svm_base = SVC(random_state=42)
    grid_search = GridSearchCV(
        estimator=svm_base,
        param_grid=param_grid,
        cv=3,                  
        scoring='f1_weighted', 
        verbose=1,             
        n_jobs=-1              
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n GRID SEARCH FINALIZADO")
    
    print(f"\nMEJORES HIPERPARÁMETROS: {grid_search.best_params_}")
    
    
    # Se evalua el mejor modelo
    mejor_svm = grid_search.best_estimator_
    y_pred = mejor_svm.predict(X_test)
    
    print("\nREPORTE MEJOR MODELO:")
    print(classification_report(y_test, y_pred))

    
    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred, labels=mejor_svm.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mejor_svm.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Matriz de Confusión - SVM Optimizado')
    plt.tight_layout() 
    
    ruta_matriz = "matrices_csv/matriz_confusion_svm.png"
    plt.savefig(ruta_matriz)
    print("MATRIZ DE CONFUSION CREADA.")

    # se genera CSV con comparacion de predicciones
    print("Generando CSV de predicciones")
    
    # Usamos los índices guardados (idx_test) para recuperar los textos exactos
    df_resultados = pd.DataFrame({
        'id_original': df_filtrado.loc[idx_test, 'id'],
        'texto_limpio': df_filtrado.loc[idx_test, 'texto_limpio'],
        'etiqueta_original': y_test.values,
        'prediccion_svm': y_pred
    })
    
    ruta_predicciones = "matrices_csv/predicciones_svm.csv"
    df_resultados.to_csv(ruta_predicciones, index=False, encoding='utf-8')
    

def main():
    print("INICIA PRE PROCESAMIENTO \n\n")
    print("1/4 Limpiando texto")
    df_entrada["texto_limpio"] = df_entrada["text"].apply(limpiar_texto)

    print("2/4 Etiquetando dataset con spaCy")
    # Aplicamos el clasificador sobre el texto limpio ANTES de quitar stopwords 
    # para no perder contexto gramatical durante la lematización
    df_entrada["etiqueta"] = df_entrada["texto_limpio"].apply(clasificar_poema_lexico)

    print("3/4 Removiendo stopwords y generando tokens")
    df_entrada["tokens_utiles"] = df_entrada["texto_limpio"].apply(remover_stopwords)

    print("4/4 Formateando y exportando CSV")
    # Se formatea el nuevo CSV ya limpio y pre procesado
    df_final = pd.DataFrame()
    df_final["id"] = range(1, len(df_entrada) + 1)
    df_final["titulo"] = df_entrada["title"]
    df_final["autor"] = df_entrada["author"]
    df_final["siglo"] = df_entrada["century"]
    df_final["texto_limpio"] = df_entrada["texto_limpio"]
    df_final["tokens_utiles"] = df_entrada["tokens_utiles"]

    # primer csv sin etiquetar
    ruta_salida = "dataset/main_poetry_limpio.csv"
    df_final.to_csv(ruta_salida, index=False, encoding="utf-8")

    df_final["etiqueta"] = df_entrada["etiqueta"]

    # Guardar el archivo estandarizado final
    ruta_salida = "dataset/main_poetry_limpio_etiquetado.csv"
    df_final.to_csv(ruta_salida, index=False, encoding="utf-8")

    print("\n\nFINALIZA PRE PROCESAMIENTO\n\n")

    print("\nResumen de Etiquetado:")
    print(df_final["etiqueta"].value_counts())

    print("\n\nCOMIENZA REPRESENTACION")

    # Generar representaciones matemáticas
    X_tfidf, y_etiquetas = representar_texto(ruta_salida)
    
    print("\n\nFINALIZA REPRESENTACION")

    # Entrenar algoritmo y evaluar 
    print("\n\nCOMIENZA ENTRENAMIENTO")
    entrenar_svm(X_tfidf, y_etiquetas, df_final)
    print("\n\nFINALIZA ENTRENAMIENTO")

if __name__ == "__main__":
    main()