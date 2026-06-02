import re
import unicodedata
import pandas as pd

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

def main():
    # se llaman las funciones de limpieza
    df_entrada["text"] = df_entrada["text"].apply(limpiar_texto)
    df_entrada["texto_limpio"] = df_entrada["text"]

    # Tokenizar el texto, eliminacion de stop words
    df_entrada["text"] = df_entrada["text"].apply(remover_stopwords)

    # Se formatea el nuevo CSV ya limpio y pre procesado
    df_final = pd.DataFrame()
    df_final["id"] = range(1, len(df_entrada) + 1)
    df_final["titulo"] = df_entrada["title"]
    df_final["autor"] = df_entrada["author"]
    df_final["siglo"] = df_entrada["century"]
    df_final["texto_limpio"] = df_entrada["texto_limpio"]
    df_final["tokens_utiles"] = df_entrada["text"]

    # Guardar el archivo estandarizado final
    ruta_salida = "dataset/main_poetry_limpio.csv"
    df_final.to_csv(ruta_salida, index=False, encoding="utf-8")

    print(f"CSV Limpio y Preprocesado: {ruta_salida}")

if __name__ == "__main__":
    main()