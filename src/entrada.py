import pandas as pd
import errors


def leer_entrada(fichero):
    """
    Lee el fichero csv de entrada.

    :param fichero: Nombre del fichero csv
    :return: dataframe con los datos leídos
    """
    entrada = pd.read_csv(fichero)
    return entrada


def checkear_entrada(df, col, positiva):
    """
    Comprueba que la columna existe en la entrada (si no salta un KeyError) y que
    la clase positiva está en la columna de salida (si no salta un UserInputError)

    :param df: Dataframe de entrada
    :param col: Nombre de la columna de salida
    :param positiva: Nombre de la clase positiva
    :return:
    """
    # Si no existe la columna, salta un "KeyError"
    columna = df[col]
    # Convertimos la columna a categórica
    columna = columna.astype("category")
    niveles = columna.unique()
    if positiva not in niveles:
        raise errors.UserInputError("La clase positiva indicada no está en los niveles de la columna de salida.")
