# =============================================================================
# Tarea 3 - Big Data UNAD
# Procesamiento en Batch con PySpark
# Dataset: Café Colombiano - Producción y Comercialización
# Autores: Grupo Colaborativo
# Universidad Nacional Abierta y a Distancia - UNAD
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, to_date, month, year, avg, sum, count, 
                                    max as spark_max, min as spark_min, 
                                    round as spark_round, desc, when)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =============================================================================
# 1. CREAR SESIÓN DE SPARK
# =============================================================================
print("=" * 70)
print(" ANÁLISIS EN BATCH - CAFÉ COLOMBIANO")
print(" Procesamiento de Datos con Apache Spark - UNAD")
print("=" * 70)

spark = SparkSession.builder \
    .appName("CafeColombia_BatchAnalysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# =============================================================================
# 2. CARGAR EL DATASET
# =============================================================================
print("\n[1/6] Cargando dataset de café colombiano...")
# Ajusta la ruta según donde tengas el archivo CSV
df = spark.read.csv("cafe_colombia.csv", 
                    header=True, 
                    inferSchema=True)

print(f"✓ Dataset cargado exitosamente")
print(f"  - Registros: {df.count():,}")
print(f"  - Columnas: {len(df.columns)}")
print(f"  - Departamentos: {df.select('departamento').distinct().count()}")

# =============================================================================
# 3. LIMPIEZA DE DATOS
# =============================================================================
print("\n[2/6] Realizando limpieza de datos...")

# Eliminar filas con valores nulos en columnas críticas
df_clean = df.dropna(subset=["registro_id", "departamento", "cantidad_kg", 
                              "precio_carga", "total_venta_cop"])

# Convertir la columna 'fecha_cosecha' a formato fecha
df_clean = df_clean.withColumn("fecha_cosecha", 
                                to_date(col("fecha_cosecha"), "yyyy-MM-dd")) \
                   .withColumn("anio", year(col("fecha_cosecha"))) \
                   .withColumn("mes", month(col("fecha_cosecha")))

# Crear columna de región cafetera
df_clean = df_clean.withColumn("region", 
    when(col("departamento").isin(["Caldas", "Quindío", "Risaralda"]), "Eje Cafetero")
    .when(col("departamento").isin(["Antioquia", "Santander", "Norte de Santander"]), "Norte")
    .when(col("departamento").isin(["Huila", "Tolima", "Cauca", "Nariño"]), "Sur")
    .otherwise("Otros")
)

print(f"✓ Datos limpios: {df_clean.count():,} registros válidos")

# =============================================================================
# 4. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
print("\n[3/6] Realizando Análisis Exploratorio de Datos (EDA)...")

# 4.1 Estadísticas generales
print("\n--- Estadísticas Descriptivas ---")
df_clean.describe(["cantidad_kg", "precio_carga", "total_venta_cop"]).show()

# 4.2 Producción total por departamento
print("\n--- Producción Total por Departamento (Top 10) ---")
produccion_dept = df_clean.groupBy("departamento") \
    .agg(
        sum("cantidad_kg").alias("total_kg"),
        sum("total_venta_cop").alias("ventas_totales_cop"),
        count("registro_id").alias("num_transacciones"),
        avg("precio_carga").alias("precio_promedio_carga")
    ).orderBy(desc("total_kg"))
produccion_dept.show(10)

# 4.3 Producción por variedad de café
print("\n--- Producción por Variedad de Café ---")
produccion_variedad = df_clean.groupBy("variedad") \
    .agg(
        sum("cantidad_kg").alias("total_kg"),
        sum("total_venta_cop").alias("ventas_totales_cop"),
        avg("precio_carga").alias("precio_promedio_carga")
    ).orderBy(desc("total_kg"))
produccion_variedad.show()

# 4.4 Producción por región cafetera
print("\n--- Producción por Región Cafetera ---")
produccion_region = df_clean.groupBy("region") \
    .agg(
        sum("cantidad_kg").alias("total_kg"),
        sum("total_venta_cop").alias("ventas_totales_cop"),
        count("registro_id").alias("num_transacciones")
    ).orderBy(desc("total_kg"))
produccion_region.show()

# 4.5 Distribución por tipo de producto
print("\n--- Distribución por Tipo de Producto ---")
tipo_producto = df_clean.groupBy("tipo_producto") \
    .agg(
        sum("cantidad_kg").alias("total_kg"),
        sum("total_venta_cop").alias("ventas_totales_cop"),
        count("registro_id").alias("num_transacciones")
    ).orderBy(desc("ventas_totales_cop"))
tipo_producto.show()

# 4.6 Análisis por calidad del café
print("\n--- Análisis por Calidad del Café ---")
calidad_stats = df_clean.groupBy("calidad") \
    .agg(
        sum("cantidad_kg").alias("total_kg"),
        avg("precio_carga").alias("precio_promedio_carga"),
        sum("total_venta_cop").alias("ventas_totales_cop")
    ).orderBy(desc("precio_promedio_carga"))
calidad_stats.show()

# 4.7 Distribución por tipo de comprador
print("\n--- Distribución por Tipo de Comprador ---")
comprador_stats = df_clean.groupBy("comprador") \
    .agg(
        sum("cantidad_kg").alias("total_kg"),
        sum("total_venta_cop").alias("ventas_totales_cop"),
        count("registro_id").alias("num_transacciones")
    ).orderBy(desc("ventas_totales_cop"))
comprador_stats.show()

# 4.8 Producción por año
print("\n--- Producción Anual ---")
produccion_anio = df_clean.groupBy("anio") \
    .agg(
        sum("cantidad_kg").alias("total_kg"),
        sum("total_venta_cop").alias("ventas_totales_cop"),
        count("registro_id").alias("num_transacciones")
    ).orderBy("anio")
produccion_anio.show()

# =============================================================================
# 5. VISUALIZACIONES
# =============================================================================
print("\n[4/6] Generando visualizaciones...")

# Convertir a Pandas para visualización
pd_dept = produccion_dept.limit(10).toPandas()
pd_variedad = produccion_variedad.toPandas()
pd_region = produccion_region.toPandas()
pd_tipo = tipo_producto.toPandas()
pd_calidad = calidad_stats.toPandas()
pd_comprador = comprador_stats.toPandas()

# Configurar estilo de gráficos con colores inspirados en Colombia
sns.set_theme(style="whitegrid")
colores_colombia = ["#FCD116", "#003893", "#CE1126", "#2E8B57", "#FF8C00"]

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Análisis Exploratorio - Café Colombiano 🇨🇴", 
             fontsize=18, fontweight="bold", y=0.98)

# Gráfico 1: Top 10 Departamentos Productores
ax1 = plt.subplot(2, 3, 1)
sns.barplot(data=pd_dept, x="total_kg", y="departamento", 
            palette=colores_colombia, ax=ax1)
ax1.set_title("Top 10 Departamentos Productores", fontweight="bold")
ax1.set_xlabel("Producción Total (kg)")
ax1.set_ylabel("Departamento")

# Gráfico 2: Producción por Variedad
ax2 = plt.subplot(2, 3, 2)
ax2.pie(pd_variedad["total_kg"], labels=pd_variedad["variedad"], 
        autopct="%1.1f%%", colors=colores_colombia)
ax2.set_title("Distribución por Variedad de Café", fontweight="bold")

# Gráfico 3: Producción por Región
ax3 = plt.subplot(2, 3, 3)
sns.barplot(data=pd_region, x="region", y="total_kg", 
            palette="YlOrRd", ax=ax3)
ax3.set_title("Producción por Región Cafetera", fontweight="bold")
ax3.set_xlabel("Región")
ax3.set_ylabel("Producción Total (kg)")
ax3.tick_params(axis='x', rotation=15)

# Gráfico 4: Precio Promedio por Calidad
ax4 = plt.subplot(2, 3, 4)
sns.barplot(data=pd_calidad, x="calidad", y="precio_promedio_carga", 
            palette="viridis", ax=ax4)
ax4.set_title("Precio Promedio por Calidad (COP/carga)", fontweight="bold")
ax4.set_xlabel("Calidad")
ax4.set_ylabel("Precio Promedio (COP)")

# Gráfico 5: Tipo de Producto más Comercializado
ax5 = plt.subplot(2, 3, 5)
sns.barplot(data=pd_tipo, x="tipo_producto", y="ventas_totales_cop", 
            palette="magma", ax=ax5)
ax5.set_title("Ventas por Tipo de Producto", fontweight="bold")
ax5.set_xlabel("Tipo de Producto")
ax5.set_ylabel("Ventas Totales (COP)")
ax5.tick_params(axis='x', rotation=20)

# Gráfico 6: Distribución por Tipo de Comprador
ax6 = plt.subplot(2, 3, 6)
ax6.pie(pd_comprador["ventas_totales_cop"], 
        labels=pd_comprador["comprador"], 
        autopct="%1.1f%%", colors=sns.color_palette("Set2"))
ax6.set_title("Distribución por Tipo de Comprador", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("cafe_colombia_eda_results.png", dpi=150, bbox_inches="tight")
print("✓ Visualizaciones guardadas en: cafe_colombia_eda_results.png")
plt.show()

# =============================================================================
# 6. HALLAZGOS PRINCIPALES
# =============================================================================
print("\n[5/6] Hallazgos Principales del Análisis:")
print("-" * 70)

# Hallazgo 1: Departamento líder
top_dept = pd_dept.iloc[0]
print(f"1. 🏆 El departamento de {top_dept['departamento']} lidera la producción "
      f"con {top_dept['total_kg']:,.0f} kg de café")

# Hallazgo 2: Variedad más producida
top_variedad = pd_variedad.iloc[0]
print(f"2. 🌱 La variedad '{top_variedad['variedad']}' es la más producida "
      f"con {top_variedad['total_kg']:,.0f} kg")

# Hallazgo 3: Región con mayor producción
top_region = pd_region.iloc[0]
print(f"3. 🗺️ La región '{top_region['region']}' concentra la mayor producción "
      f"con {top_region['total_kg']:,.0f} kg")

# Hallazgo 4: Calidad con mejor precio
top_calidad = pd_calidad.iloc[0]
print(f"4. 💰 El café de calidad '{top_calidad['calidad']}' obtiene el mejor "
      f"precio promedio con ${top_calidad['precio_promedio_carga']:,.0f} COP/carga")

# Hallazgo 5: Comprador principal
top_comprador = pd_comprador.iloc[0]
print(f"5. 🤝 Los '{top_comprador['comprador']}' representan el mayor volumen "
      f"de compras con ${top_comprador['ventas_totales_cop']:,.0f} COP")

# =============================================================================
# 7. CERRAR SESIÓN
# =============================================================================
print("\n[6/6] Finalizando análisis...")
spark.stop()
print("✓ Análisis completado exitosamente")
print("=" * 70)
print("🇨🇴 ¡Viva el Café Colombiano!")
print("=" * 70)
