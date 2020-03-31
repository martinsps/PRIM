from entrada import leer_entrada, checkear_entrada

# Valores para el ejemplo
fichero = "entrada.csv"
columna_salida = "Survived"
clase_positiva = "Yes"

entrada = leer_entrada(fichero)
checkear_entrada(df=entrada, col=columna_salida, positiva=clase_positiva)

print(len(entrada[entrada["Fare"]>=13]["Fare"].sort_values()[0:1000]))
