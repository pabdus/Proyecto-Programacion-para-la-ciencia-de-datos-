# 🍷 Wine Quality Dashboard

## 📌 Descripción

Este proyecto se desarrolló con el objetivo de aplicar métodos estadísticos computacionales para analizar la calidad del vino a partir de variables fisicoquímicas del dataset **WineQT**.

El trabajo se realizó siguiendo los lineamientos de la actividad, enfocándose en:

* métodos de incertidumbre
* contraste de hipótesis
* modelado estadístico

---

## 🎯 Enfoque del desarrollo

Se trabajó en función de los requerimientos del curso:

* Análisis exploratorio de datos (EDA)
* Contraste de hipótesis:

  * Prueba de Levene (igualdad de varianzas)
  * Prueba t de Welch (diferencia de medias)
* Modelado:

  * Regresión lineal
  * Regresión logística
* Dashboard interactivo en Dash

Se priorizó la interpretabilidad sobre la complejidad.

---

## ⚙️ Variables utilizadas

* `alcohol`
* `volatile acidity`
* `sulphates`

La selección responde tanto a su relación con la calidad como a los lineamientos de la actividad, que requerían un modelo controlado y explicable.

---

## 📊 Resultados principales

* Alcohol → relación positiva con la calidad
* Acidez volátil → relación negativa significativa
* Prueba de Levene → evidencia de varianzas no homogéneas
* Prueba t de Welch → diferencias significativas entre grupos
* Regresión lineal → capacidad explicativa moderada
* Regresión logística → clasificación aceptable

---

## ⚠️ Limitaciones

* Uso de pocas variables (acorde a la actividad)
* No se aplicó escalamiento
* No se utilizó validación cruzada
* Variable objetivo subjetiva

---

## 🔁 Evaluación y mejoras

El modelo cumple con los objetivos del análisis, pero puede mejorarse mediante:

* inclusión de más variables
* feature engineering
* validación cruzada
* ajuste de hiperparámetros
* regularización

---

## 🖥️ Dashboard

Desarrollado con:

* Dash
* Plotly
* Pandas

Incluye análisis, contraste de hipótesis, modelos y predicción interactiva.

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

https://hub.gesis.mybinder.org/user/pabdus-proyecto-encia-de-datos--3bpmi3ih/lab

Ejecutar:

```bash
python app.py
```

Ir a:

```
/proxy/8050/
```

---

## 🧠 Conclusión

El proyecto cumple con los requerimientos de la actividad y aplica correctamente técnicas de contraste de hipótesis y modelado estadístico, priorizando la comprensión del problema sobre la complejidad del modelo.

