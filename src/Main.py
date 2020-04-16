#!/usr/bin/env python3
from entrada import leer_entrada, checkear_entrada
from PRIM import PRIM

# Entrada del usuario:
# fichero : fichero con los datos
# salida : nombre de la columna que es la salida
# clase_positiva : nombre de la clase positiva

# Valores para el ejemplo
fichero = "entrada.csv"
columna_salida = "Survived"
clase_positiva = "Yes"


def ejecutar_algoritmo(fichero, columna_salida, clase_positiva):
    """
    Función que ejecuta los algoritmos PRIM / C50-SD de búsqueda de subgrupos
    sobre un dataset de entrada.
    
    :param fichero: Nombre del fichero con los datos
    :param columna_salida: Nombre de la columna que es la salida
    :param clase_positiva: Nombre de la clase positiva 
    :return: 
    """
    entrada = leer_entrada(fichero)
    ordinal_cols = {
        "Pclass": [1, 2, 3]
    }
    checkear_entrada(df=entrada, col=columna_salida, positiva=clase_positiva, ordinal_columns=ordinal_cols)
    prim = PRIM(entrada,columna_salida,clase_positiva,alpha=0.2,threshold_box=0.15, threshold_global=0.1,
                ordinal_columns=ordinal_cols)
    prim.execute()


ejecutar_algoritmo(fichero, columna_salida, clase_positiva)

