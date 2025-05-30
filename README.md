# Clustering con Python
El término clustering hace referencia a un amplio abanico de técnicas cuya finalidad es encontrar patrones o grupos (clusters) dentro de un conjunto de observaciones. Las particiones se establecen de forma que, las observaciones que están dentro de un mismo grupo, son similares entre ellas y distintas a las observaciones de otros grupos. Se trata de un método de aprendizaje no supervisado (unsupervised), ya que el proceso no tiene en cuenta a qué grupo pertenece realmente cada observación (si es que existe tal información). Esta característica es la que diferencia al clustering de las métodos de clasificación en el que sí emplea la verdadera clasificación durante su entrenamiento.

## Medidas de distancia
Todos los métodos de clustering tienen una cosa en común, para llevar a cabo las agrupaciones necesitan definir y cuantificar la similitud entre las observaciones. El término distancia se emplea dentro del contexto del clustering como cuantificación de la similitud o diferencia entre observaciones. Si se representan las observaciones en un espacio 𝑝 dimensional, siendo 𝑝 el número de variables asociadas a cada observación, cuando más se asemejen dos observaciones más próximas estarán, de ahí que se emplee el término distancia. La característica que hace del clustering un método adaptable a escenarios muy diversos es que puede emplear casi cualquier tipo de distancia, lo que permite al investigador escoger la más adecuada para el estudio en cuestión. A continuación, se describen algunas de las más utilizadas.

### 1. Distancia euclídea
La distancia euclídea entre dos puntos 𝑝 y 𝑞 se define como la longitud del segmento que une ambos puntos. En coordenadas cartesianas, la distancia euclídea se calcula empleando el teorema de Pitágoras. Por ejemplo, en un espacio de dos dimensiones en el que cada punto está definido por las coordenadas (𝑥,𝑦), la distancia euclídea entre 𝑝 y 𝑞 es:
<img src="https://e7.pngegg.com/pngimages/801/436/png-clipart-euclidean-distance-k-nearest-neighbors-algorithm-euclidean-space-line-blue-angle.png" alt="Euclidean distance image" height="100px">

Esta ecuación puede generalizarse para un espacio euclídeo n-dimensional donde cada punto está definido por un vector de n coordenadas:  𝑝=(𝑝1,𝑝2,𝑝3,...,𝑝𝑛) y  𝑞=(𝑞1,𝑞2,𝑞3,...,𝑞𝑛).

Una forma de dar mayor peso a aquellas observaciones que están más alejadas es emplear la distancia euclídea al cuadrado. En el caso del clustering, donde se busca agrupar observaciones que minimicen la distancia, esto se traduce en una mayor influencia de aquellas observaciones que están más distantes.

### 2. Distancia de Manhattan
La distancia de Manhattan, también conocida como taxicab metric, rectilinear distance o L1 distance, define la distancia entre dos puntos 𝑝 y 𝑞 como el sumatorio de las diferencias absolutas entre cada dimensión. Esta medida se ve menos afectada por outliers (es más robusta) que la distancia euclídea debido a que no eleva al cuadrado las diferencias.

<img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjhpnVELOqVWvtnprL4GqjGe7th-ToBmKfxZ49layLePHhQYYk-a_nRN4jKC0QSN1ieDTyYJG_DiSWisJ1BD6_2bFY4sOlVWQK-yUb9L8ZHnt1FqcKmsGYkW7gHo5DXNcvJxgHtT6DE7Id5/s1600/f%C3%B3rmula2.png" alt="Manhattan distance" height="100px">

La siguiente imagen muestra una comparación entre la distancia euclídea (segmento verde) y la distancia de manhattan (segmento rojo, amarillo y azul) en un espacio bidimensional. Existen múltiples caminos para unir dos puntos con una misma distancia de manhattan, pero solo uno con la distancia euclídea.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Manhattan_distance.svg/1200px-Manhattan_distance.svg.png" alt="Distancia Manhattan" height="150px">

### 3. Correlación
La correlación es una medida de distancia muy útil cuando la definición de similitud se hace en términos de patrón o forma y no de desplazamiento o magnitud.
𝑑𝑐𝑜𝑟(𝑝,𝑞)=1−correlacion(𝑝,𝑞) donde la correlación puede ser de distintos tipos (Pearson, Spearman, Kendall...)

En la siguiente imagen se muestra el perfil de 3 observaciones. a y b tienen exactamente el mismo patrón, pero b está desplazada 4 unidades por debajo de a. La observación c tiene un patrón distinto a las demás, pero su rango de valores es similar al de b. Acorde a la distancia euclídea, las observaciones b y c son las más similares ya que la suma de la distancia al cuadrado entre sus valores es menor. Sin embargo, acorde a la correlación de Pearson, las observaciones más similares son a y b ya que siguen el mismo patrón.

![Correlación](https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRWz4mUNDkrJWfRuEDmsUJQuvzGA8BbYhQ20o6nmOLYua4yyvn9)

Este ejemplo pone de manifiesto que no existe una única medida de distancia que sea mejor que las demás, sino que depende del contexto en el que se utilice.

### 4. Correlación Jackknife
El coeficiente de correlación de Pearson resulta efectivo como medida de similitud en ámbitos muy diversos. Sin embargo, tiene la desventaja de no ser robusto frente a outliers, a pesar de que se cumpla la condición de normalidad. Si dos variables tienen un pico o un valle común en una única observación, por ejemplo, por un error de lectura, la correlación va a estar dominada por este registro a pesar de que entre las dos variables no exista correlación real. Lo mismo puede ocurrir en la dirección opuesta. Si dos variables están altamente correlacionadas excepto para una observación, en la que los valores son muy dispares, entonces la correlación existente quedará enmascarada. Una forma de evitarlo es recurrir a la Jackknife correlation, que consiste en calcular todos los posibles coeficientes de correlación entre dos variables si se excluye cada vez una de las observaciones. El promedio de todas las Jackknife correlations atenúa en cierta medida el efecto del outlier.

### 5. Simple matching coefficient
Cuando las variables con las que se pretende determinar la similitud entre observaciones son de tipo binario, a pesar de que es posible codificarlas de forma numérica como 1 o 0, no tiene sentido aplicar operaciones aritméticas sobre ellas (media, suma...). Por ejemplo, si la variable sexo se codifica como 1 para mujer y 0 para hombre, carece de significado decir que la media de la variable sexo en un determinado set de datos es 0.5. En situaciones como esta, no se pueden emplear medidas de similitud basadas en distancia euclídea, manhattan, correlación...

Dadas dos observaciones A y B, cada uno con 𝑛 atributos binarios, el simple matching coefficient (SMC) define la similitud entre ellos como:

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSsR4xCny5EqrFV7Puy9_GqFMlgkJ0a6rKPsw&s" alt="Descripción de la imagen" height="100px">

donde 𝑀00 y 𝑀11 son el número de variables para las que ambas observaciones tienen el mismo valor (ambas 0 o ambas 1), y 𝑀01 y 𝑀10 el número de variables que no coinciden. El valor de distancia simple matching distance (SMD) se corresponde con 1−𝑆𝑀𝐶.

### 6. Índice Jaccard
El índice Jaccard o coeficiente de correlación Jaccard es similar al simple matching coefficient (SMC). La diferencia radica en que, el SMC, tiene el término 𝑀00 en el numerador y denominador, mientras que el índice de Jaccard no. Esto significa que SMC considera como coincidencias tanto si el atributo está presente en ambos sets como si el atributo no está en ninguno de los sets, mientras que Jaccard solo cuenta como coincidencias cuando el atributo está presente en ambos sets.
𝐽= (𝑀11) / (𝑀01+𝑀10+𝑀11)

o en términos matemáticos de sets:

<img src="https://miro.medium.com/v2/resize:fit:648/0*ZjJjzvmmdD9QhXG2" alt="Descripción de la imagen" height="100px">

La distancia de Jaccard (1−𝐽) supera a la simple matching distance en aquellas situaciones en las que la coincidencia de ausencia no aporta información. Para ilustrar este hecho, supóngase que se quiere cuantificar la similitud entre dos clientes de un supermercado en base a los artículos comprados. Es de esperar que cada cliente solo adquiera unos pocos artículos de los muchos disponibles, por lo que el número de artículos no comprados por ninguno (𝑀00) será muy alto. Como la distancia de Jaccard ignora las coincidencias de tipo 𝑀00, el grado de similitud dependerá únicamente de las coincidencias entre los artículos comprados.

### 7. Distancia Coseno
El coseno del ángulo que forman dos vectores puede interpretarse como una medida de similitud de sus orientaciones, independientemente de sus magnitudes. Si dos vectores tienen exactamente la misma orientación (el ángulo que forman es 0º) su coseno toma el valor de 1, si son perpendiculares (forman un ángulo de 90º) su coseno es 0 y si tienen orientaciones opuestas (ángulo de 180º) su coseno es de -1.

<img src="https://www.formulasexplicadas.com/wp-content/uploads/2025/01/angulo-entre-dos-vectores-formula.png" alt="Fórmula ángulo entre dos vectores" height="100px">

Si los vectores se normalizan restandoles la media (𝑋−𝑋¯), la medida recibe el nombre de coseno centrado y es equivalente a la correlación de Pearson.

## Escalado de las variables

La escala en la que se miden las variables y la magnitud de su varianza pueden afectar a los resultados obtenidos por clustering. Si una variable tiene una escala mucho mayor que el resto, determinará en gran medida el valor de distancia/similitud obtenido al comparar las observaciones, dirigiendo así la agrupación final. Escalar y centrar las variables antes de calcular la matriz de distancias para que tengan media 0 y desviación estándar 1, asegura que todas las variables tengan el mismo peso cuando se realice el clustering. Sebastian Raschka hace un análisis muy explicativo de los distintos tipos de escalado y normalización.

La forma más utilizada para escalar y centrar las variables es:

<img src="https://proyectodescartes.org/iCartesiLibri/materiales_didacticos/IntroduccionEstadisticaProbabilidad/4ESO/Estadistica/6_5PuntuacionesNormalizadas/img/FormulaTipificacion.png" alt="Fórmula tipificación" height="100px">

Aplicando esta transformación, existe una relación entre la distancia euclídea y la correlación de Pearson que hace que los resultados obtenidos por clustering sean equivalentes.

