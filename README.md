# 🍷 Wine Quality Dashboard

## 📌 Descripción

Este proyecto se centró en el análisis de la calidad del vino usando variables fisicoquímicas del dataset **WineQT**.
El objetivo fue cumplir con los lineamientos de la actividad: aplicar métodos de incertidumbre, contraste de hipótesis y modelos estadísticos dentro de un dashboard interactivo.

---

## 🎯 Enfoque

El desarrollo se realizó en función directa de los requisitos del curso:

* Análisis exploratorio de datos (EDA)
* Contraste de hipótesis (prueba t de Welch)
* Modelos:

  * Regresión lineal
  * Regresión logística
* Visualización en dashboard interactivo

Todo el proceso se orientó a **explicar el problema y no solo predecirlo**.

---

## ⚙️ Variables utilizadas

Se trabajó únicamente con:

* `alcohol`
* `volatile acidity`
* `sulphates`

La selección se hizo por su relación con la variable objetivo (`quality`), pero también porque la actividad requería un enfoque controlado y explicable, evitando complejidad innecesaria.

---

## 📊 Resultados

* Alcohol → relación positiva con la calidad
* Acidez volátil → relación negativa significativa
* Diferencias entre grupos → estadísticamente significativas
* Regresión lineal → capacidad explicativa moderada
* Regresión logística → clasificación aceptable

Los resultados son consistentes con el carácter parcialmente subjetivo de la calidad del vino.

---

## ⚠️ Limitaciones

* Uso de pocas variables (en parte por lineamientos de la actividad)
* No se aplicó escalamiento
* No se utilizó validación cruzada
* Variable objetivo subjetiva

---

## 🔁 Recomendaciones

* Incluir más variables
* Aplicar feature engineering
* Usar escalamiento (`StandardScaler`)
* Implementar validación cruzada
* Evaluar modelos más complejos

---

## 🖥️ Dashboard

Desarrollado con:

* Dash
* Plotly
* Pandas

Incluye análisis, modelos y un predictor interactivo en tiempo real.

---

## ▶️ Ejecución local

```bash
pip install -r requirements.txt
python app.py
```

Abrir en:

```
http://localhost:8050
```

---

## 🌐 Ejecución en Binder

Puedes ejecutarlo directamente aquí:

https://hub.gesis.mybinder.org/user/pabdus-proyecto-encia-de-datos--3bpmi3ih/lab

Si no abre automáticamente el dashboard:

```bash
python app.py
```

Luego ir a:

```
http://localhost:8050
```

---

## 🧠 Conclusión

El proyecto cumple con los requerimientos de la actividad y demuestra cómo aplicar métodos estadísticos para analizar un problema real.
Más que buscar máxima precisión, se priorizó la interpretación y la coherencia con los conceptos trabajados en el curso.

