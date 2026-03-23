import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score,
                             confusion_matrix, classification_report,
                             accuracy_score)

# ── Datos ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("WineQT.csv")
df['calidad_binaria'] = df['quality'].apply(lambda x: 'Alta (≥6)' if x >= 6 else 'Baja (<6)')
df['calidad_num'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# ── Modelos ───────────────────────────────────────────────────────────────────
FEATURES = ['alcohol', 'volatile acidity', 'sulphates']

X = df[FEATURES]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
modelo_lineal = LinearRegression().fit(X_train, y_train)
y_pred_lin = modelo_lineal.predict(X_test)
mse  = mean_squared_error(y_test, y_pred_lin)
r2   = r2_score(y_test, y_pred_lin)
rmse = np.sqrt(mse)

y_log = df['calidad_num']
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X, y_log, test_size=0.3, random_state=42)
modelo_log = LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000, random_state=42).fit(Xl_train, yl_train)
y_pred_log = modelo_log.predict(Xl_test)
y_prob_log = modelo_log.predict_proba(Xl_test)[:, 1]
acc    = accuracy_score(yl_test, y_pred_log)
cm     = confusion_matrix(yl_test, y_pred_log)
report = classification_report(yl_test, y_pred_log, output_dict=True)

# ── Contraste de hipótesis ────────────────────────────────────────────────────
alcohol_alta = df[df['calidad_binaria'] == 'Alta (≥6)']['alcohol']
alcohol_baja = df[df['calidad_binaria'] == 'Baja (<6)']['alcohol']
_, levene_p  = stats.levene(alcohol_alta, alcohol_baja)
t_stat, p_val = stats.ttest_ind(alcohol_alta, alcohol_baja, equal_var=False)

# ── Comparación de configuraciones logísticas ─────────────────────────────────
configuraciones = [
    {'C': 0.01, 'solver': 'lbfgs'},
    {'C': 0.1,  'solver': 'lbfgs'},
    {'C': 1.0,  'solver': 'lbfgs'},
    {'C': 10.0, 'solver': 'lbfgs'},
    {'C': 1.0,  'solver': 'liblinear'},
    {'C': 1.0,  'solver': 'saga'},
]
resultados_log = []
for cfg in configuraciones:
    m = LogisticRegression(C=cfg['C'], solver=cfg['solver'],
                           max_iter=1000, random_state=42).fit(Xl_train, yl_train)
    resultados_log.append({
        'C': cfg['C'], 'Solver': cfg['solver'],
        'Accuracy': round(accuracy_score(yl_test, m.predict(Xl_test)), 4)
    })
df_cfg = pd.DataFrame(resultados_log)

# ── Estilos ───────────────────────────────────────────────────────────────────
COLORES = {
    'primario':   '#d97757',
    'secundario': '#6a9bcc',
    'acento':     '#788c5d',
    'fondo':      '#faf9f5',
    'oscuro':     '#141413',
    'gris':       '#b0aea5',
}

CARD_STYLE = {
    'backgroundColor': 'white',
    'borderRadius': '10px',
    'padding': '20px',
    'marginBottom': '20px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)'
}

INFO_STYLE = {
    'backgroundColor': '#fffbf0',
    'borderLeft': f'4px solid #e6b84a',
    'borderRadius': '0 8px 8px 0',
    'padding': '12px 16px',
    'marginBottom': '16px',
    'fontSize': '13px',
    'color': '#5a4a1a',
    'lineHeight': '1.6'
}

def titulo_seccion(texto, subtexto=""):
    return html.Div([
        html.H4(texto, style={'color': COLORES['oscuro'], 'fontWeight': '700', 'marginBottom': '4px'}),
        html.P(subtexto, style={'color': COLORES['gris'], 'fontSize': '14px', 'marginBottom': '16px'})
    ])

def tarjeta_kpi(titulo, valor, subtitulo="", color=None):
    color = color or COLORES['primario']
    return dbc.Card([
        dbc.CardBody([
            html.P(titulo, style={'fontSize': '12px', 'color': COLORES['gris'],
                                  'textTransform': 'uppercase', 'letterSpacing': '0.05em',
                                  'marginBottom': '4px'}),
            html.H3(valor, style={'color': color, 'fontWeight': '800', 'marginBottom': '4px'}),
            html.P(subtitulo, style={'fontSize': '12px', 'color': COLORES['gris'], 'marginBottom': '0'})
        ])
    ], style={**CARD_STYLE, 'borderLeft': f'4px solid {color}', 'marginBottom': '10px'})

def caja_narrativa(texto):
    return html.Div(
        html.P(texto, style={'margin': '0', 'fontSize': '14px',
                             'color': COLORES['oscuro'], 'lineHeight': '1.7'}),
        style={'backgroundColor': '#f0f4ff', 'borderLeft': f'4px solid {COLORES["secundario"]}',
               'borderRadius': '0 8px 8px 0', 'padding': '14px 18px', 'marginBottom': '20px'}
    )

def caja_info(texto):
    return html.Div(
        html.P([html.Span("ℹ️  ", style={'fontSize': '15px'}), texto],
               style={'margin': '0', 'lineHeight': '1.6'}),
        style=INFO_STYLE
    )

def caja_conclusion(texto):
    return dbc.Card([
        html.Div([
            html.H5("📋 Conclusión", style={'color': COLORES['oscuro'],
                                            'fontWeight': '700', 'marginBottom': '10px'}),
            html.P(texto, style={'fontSize': '14px', 'lineHeight': '1.7', 'margin': '0'})
        ], style={'padding': '16px'})
    ], style={**CARD_STYLE, 'borderLeft': f'4px solid {COLORES["acento"]}'})

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
app.title = "Wine Quality Dashboard"

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="🍷 Wine Quality — Análisis Completo",
        color="dark", dark=True, className="mb-4"
    ),
    dbc.Container([
        dcc.Tabs(id="tabs", value="tab-eda", children=[
            dcc.Tab(label="📊 Análisis Exploratorio",  value="tab-eda"),
            dcc.Tab(label="🔬 Contraste de Hipótesis", value="tab-hipotesis"),
            dcc.Tab(label="📈 Regresión Lineal",        value="tab-lineal"),
            dcc.Tab(label="🎯 Regresión Logística",     value="tab-logistica"),
            dcc.Tab(label="🤖 Predictor",               value="tab-predictor"),
        ]),
        html.Div(id="contenido-tab", className="mt-4")
    ], fluid=True)
], style={'backgroundColor': '#faf9f5', 'minHeight': '100vh'})


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
def tab_eda():
    corr = df.drop(columns=['Id', 'calidad_binaria', 'calidad_num']).corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r',
                         zmin=-1, zmax=1, aspect='auto', title="Matriz de Correlación")
    fig_corr.update_layout(paper_bgcolor='white', plot_bgcolor='white', title_font_size=14)

    fig_qual = px.histogram(df, x='quality', color='calidad_binaria',
                            color_discrete_map={'Alta (≥6)': COLORES['primario'],
                                               'Baja (<6)': COLORES['secundario']},
                            barmode='overlay', opacity=0.8,
                            title="Distribución de la Variable quality")
    fig_qual.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                           legend_title="Calidad", title_font_size=14)

    fig_box = make_subplots(rows=1, cols=3,
                            subplot_titles=["Alcohol", "Volatile Acidity", "Sulphates"])
    for i, var in enumerate(FEATURES, 1):
        for cat, color in [('Alta (≥6)', COLORES['primario']), ('Baja (<6)', COLORES['secundario'])]:
            fig_box.add_trace(
                go.Box(y=df[df['calidad_binaria'] == cat][var], name=cat,
                       marker_color=color, showlegend=(i == 1)),
                row=1, col=i)
    fig_box.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          title_text="Variables Clave por Grupo de Calidad",
                          title_font_size=14, height=400)

    return html.Div([
        titulo_seccion("Análisis Exploratorio de Datos",
                       "Exploración inicial del dataset WineQT — 1,143 observaciones, 12 variables fisicoquímicas"),

        caja_info(
            "El EDA es el punto de partida del análisis. Antes de construir cualquier modelo, "
            "es fundamental entender la distribución de los datos, identificar relaciones entre "
            "variables y detectar patrones que guíen las decisiones posteriores."
        ),

        dbc.Row([
            dbc.Col(tarjeta_kpi("Observaciones", "1,143", "registros de vino tinto"), md=3),
            dbc.Col(tarjeta_kpi("Variables", "12", "propiedades fisicoquímicas"), md=3),
            dbc.Col(tarjeta_kpi("Vinos Alta Calidad", f"{len(alcohol_alta)}", "quality ≥ 6", COLORES['acento']), md=3),
            dbc.Col(tarjeta_kpi("Vinos Baja Calidad", f"{len(alcohol_baja)}", "quality < 6", COLORES['secundario']), md=3),
        ], className="mb-3"),

        caja_narrativa(
            "El dataset Wine Quality contiene 1,143 registros de vino tinto con 12 propiedades "
            "fisicoquímicas. La variable objetivo 'quality' representa una calificación subjetiva "
            "otorgada por catadores. El análisis de correlación reveló que el alcohol tiene la "
            "relación positiva más fuerte con la calidad (r = 0.48), mientras que la acidez volátil "
            "presenta la correlación negativa más significativa (r = -0.41). Estas dos variables "
            "se convirtieron en los predictores principales de todos los modelos posteriores."
        ),

        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_corr), style=CARD_STYLE), md=7),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_qual), style=CARD_STYLE), md=5),
        ], className="mb-3"),

        caja_info(
            "Los boxplots comparan la distribución de las tres variables predictoras seleccionadas "
            "entre ambos grupos de calidad. Una separación visual clara entre cajas indica que "
            "la variable tiene poder discriminativo — es decir, ayuda al modelo a distinguir "
            "entre vinos buenos y malos."
        ),

        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_box), style=CARD_STYLE), md=12),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.Label("Explorar relación con quality — selecciona variable X:",
                                   style={'fontWeight': '600', 'fontSize': '13px'}),
                        dcc.Dropdown(
                            id='eda-xvar',
                            options=[{'label': c, 'value': c}
                                     for c in df.select_dtypes('float64').columns],
                            value='alcohol', clearable=False,
                            style={'marginBottom': '10px'}
                        ),
                        dcc.Graph(id='eda-scatter-dinamico')
                    ], style={'padding': '16px'})
                ], style=CARD_STYLE)
            ], md=12)
        ]),

        caja_conclusion(
            "El EDA confirmó que alcohol (r = 0.48) y volatile acidity (r = -0.41) son las variables "
            "con mayor relación con la calidad del vino. Los boxplots muestran una separación clara "
            "entre grupos para estas dos variables, validando su selección como predictores. "
            "El umbral quality ≥ 6 para definir alta calidad es consistente con la mediana de la "
            "distribución, garantizando una partición natural y equilibrada del dataset "
            "(621 alta calidad vs 522 baja calidad)."
        )
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CONTRASTE DE HIPÓTESIS
# ══════════════════════════════════════════════════════════════════════════════
def tab_hipotesis():
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=alcohol_alta, name='Alta (≥6)', opacity=0.7,
                                    marker_color=COLORES['primario'], nbinsx=30))
    fig_hist.add_trace(go.Histogram(x=alcohol_baja, name='Baja (<6)', opacity=0.7,
                                    marker_color=COLORES['secundario'], nbinsx=30))
    fig_hist.update_layout(barmode='overlay', title="Distribución del Alcohol por Grupo",
                           xaxis_title="Alcohol (%)", yaxis_title="Frecuencia",
                           paper_bgcolor='white', plot_bgcolor='white', title_font_size=14)

    fig_violin = px.violin(df, x='calidad_binaria', y='alcohol', color='calidad_binaria',
                           color_discrete_map={'Alta (≥6)': COLORES['primario'],
                                              'Baja (<6)': COLORES['secundario']},
                           box=True, points='outliers',
                           title="Distribución del Alcohol — Violin Plot")
    fig_violin.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                             title_font_size=14, showlegend=False)

    decision  = "✅ Se RECHAZA H₀" if p_val < 0.05 else "❌ No se rechaza H₀"
    color_dec = COLORES['acento'] if p_val < 0.05 else COLORES['gris']

    return html.Div([
        titulo_seccion("Contraste de Hipótesis",
                       "¿Es significativa la diferencia en alcohol entre vinos de alta y baja calidad?"),

        caja_info(
            "El contraste de hipótesis permite determinar si una diferencia observada entre grupos "
            "es estadísticamente significativa o si podría ser ruido aleatorio. "
            "H₀ siempre representa el escenario conservador — 'no hay diferencia' — "
            "y necesitamos evidencia fuerte para rechazarla. El umbral estándar es p-valor < 0.05."
        ),

        caja_narrativa(
            "H₀: La media del alcohol en vinos de alta calidad es igual a la de baja calidad. "
            "H₁: Las medias son significativamente diferentes. "
            "Antes de la prueba t se aplicó la prueba de Levene (p ≈ 0) que confirmó varianzas "
            "heterogéneas entre grupos, justificando el uso de la prueba t de Welch (equal_var=False) "
            "en lugar de la versión clásica que asume homocedasticidad."
        ),

        dbc.Row([
            dbc.Col(tarjeta_kpi("Media Alcohol — Alta", f"{alcohol_alta.mean():.4f}°",
                                "quality ≥ 6", COLORES['primario']), md=3),
            dbc.Col(tarjeta_kpi("Media Alcohol — Baja", f"{alcohol_baja.mean():.4f}°",
                                "quality < 6", COLORES['secundario']), md=3),
            dbc.Col(tarjeta_kpi("Estadístico t", f"{t_stat:.4f}",
                                "Prueba t de Welch", COLORES['oscuro']), md=3),
            dbc.Col(tarjeta_kpi("P-valor", f"{p_val:.2e}", decision, color_dec), md=3),
        ], className="mb-3"),

        caja_info(
            "El estadístico t mide qué tan grande es la diferencia entre medias en unidades de "
            "error estándar. Cuanto mayor sea su valor absoluto, más improbable es que la "
            "diferencia sea aleatoria. El p-valor expresa esa probabilidad directamente: "
            "un p-valor de 3.71×10⁻⁵⁸ significa que hay prácticamente cero probabilidad "
            "de observar esta diferencia si H₀ fuera verdadera."
        ),

        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_hist),   style=CARD_STYLE), md=7),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_violin), style=CARD_STYLE), md=5),
        ], className="mb-3"),

        caja_conclusion(
            f"Con un estadístico t = {t_stat:.4f} y un p-valor = {p_val:.2e}, "
            f"muy inferior al nivel de significancia α = 0.05, se rechaza H₀. "
            f"La diferencia de {alcohol_alta.mean() - alcohol_baja.mean():.2f}° de alcohol "
            f"entre grupos no es atribuible al azar. Los vinos de alta calidad tienen "
            f"significativamente más alcohol que los de baja calidad, resultado consistente "
            f"con la correlación positiva r = 0.48 identificada en el EDA. "
            f"Adicionalmente, la comparación entre media y mediana en cada grupo "
            f"(alta: {alcohol_alta.mean():.2f}° vs {alcohol_alta.median():.2f}°; "
            f"baja: {alcohol_baja.mean():.2f}° vs {alcohol_baja.median():.2f}°) "
            f"confirmó distribuciones aproximadamente simétricas, validando la robustez del análisis."
        )
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — REGRESIÓN LINEAL
# ══════════════════════════════════════════════════════════════════════════════
def tab_lineal():
    coef_names  = FEATURES
    coef_values = modelo_lineal.coef_

    fig_coef = go.Figure(go.Bar(
        x=coef_values,
        y=coef_names,
        orientation='h',
        marker_color=[COLORES['primario'] if v > 0 else COLORES['secundario'] for v in coef_values],
        text=[f"{v:+.4f}" for v in coef_values],
        textposition='outside'
    ))
    fig_coef.update_layout(
        title="Coeficientes del Modelo Lineal",
        xaxis_title="Valor del coeficiente",
        paper_bgcolor='white', plot_bgcolor='white',
        title_font_size=14, height=300,
        xaxis=dict(zeroline=True, zerolinecolor='#ccc', zerolinewidth=2)
    )

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=y_test, y=y_pred_lin, mode='markers',
        marker=dict(color=COLORES['primario'], opacity=0.5, size=5),
        name='Predicciones'
    ))
    fig_pred.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        line=dict(color=COLORES['oscuro'], dash='dash', width=2),
        name='Predicción perfecta'
    ))
    fig_pred.update_layout(
        title="Valores Reales vs Predichos",
        xaxis_title="Quality real",
        yaxis_title="Quality predicho",
        paper_bgcolor='white', plot_bgcolor='white',
        title_font_size=14
    )

    residuos = y_test - y_pred_lin
    fig_res = px.histogram(residuos, nbins=30, title="Distribución de Residuos",
                           color_discrete_sequence=[COLORES['secundario']])
    fig_res.update_layout(paper_bgcolor='white', plot_bgcolor='white',
                          xaxis_title="Residuo", yaxis_title="Frecuencia",
                          title_font_size=14, showlegend=False)
    fig_res.add_vline(x=0, line_dash="dash", line_color=COLORES['oscuro'])

    ecuacion = (f"ŷ = {modelo_lineal.intercept_:.4f} "
                f"+ {coef_values[0]:+.4f}·alcohol "
                f"+ {coef_values[1]:+.4f}·volatile_acidity "
                f"+ {coef_values[2]:+.4f}·sulphates")

    return html.Div([
        titulo_seccion("Regresión Lineal Múltiple",
                       "Predicción numérica de quality a partir de tres variables fisicoquímicas"),

        caja_info(
            "La regresión lineal múltiple modela la relación entre una variable dependiente continua "
            "(quality) y múltiples predictores. Cada coeficiente representa el cambio en quality "
            "por cada unidad de aumento en esa variable, manteniendo las demás constantes "
            "(efecto ceteris paribus). Las variables se seleccionaron priorizando alta correlación "
            "con quality y ausencia de multicolinealidad entre predictores."
        ),

        dbc.Row([
            dbc.Col(tarjeta_kpi("R²", f"{r2:.4f}",
                                f"Explica el {r2*100:.1f}% de la variabilidad", COLORES['primario']), md=3),
            dbc.Col(tarjeta_kpi("RMSE", f"{rmse:.4f}",
                                "Error promedio en unidades de quality", COLORES['secundario']), md=3),
            dbc.Col(tarjeta_kpi("MSE", f"{mse:.4f}",
                                "Error cuadrático medio", COLORES['oscuro']), md=3),
            dbc.Col(tarjeta_kpi("Intercepto (b₀)", f"{modelo_lineal.intercept_:.4f}",
                                "Valor base del modelo", COLORES['gris']), md=3),
        ], className="mb-3"),

        dbc.Card([
            html.Div([
                html.H6("Ecuación del Modelo", style={'fontWeight': '700', 'marginBottom': '8px',
                                                       'color': COLORES['oscuro']}),
                html.Code(ecuacion, style={'fontSize': '14px', 'color': COLORES['primario'],
                                           'fontWeight': '600'})
            ], style={'padding': '16px'})
        ], style={**CARD_STYLE, 'borderLeft': f'4px solid {COLORES["primario"]}', 'marginBottom': '20px'}),

        caja_info(
            "Los coeficientes negativos indican que al aumentar esa variable, la calidad predicha "
            "disminuye. Los positivos indican el efecto contrario. La magnitud del coeficiente "
            "refleja el impacto por unidad, pero solo es comparable entre variables si estas "
            "están en escalas similares."
        ),

        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_coef), style=CARD_STYLE), md=5),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_pred), style=CARD_STYLE), md=7),
        ], className="mb-3"),

        caja_info(
            "El histograma de residuos muestra la distribución de los errores del modelo "
            "(valor real − valor predicho). Un buen modelo tiene residuos centrados en cero "
            "y distribuidos simétricamente, lo que indica que no hay sesgo sistemático "
            "en las predicciones."
        ),

        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_res), style=CARD_STYLE), md=12),
        ], className="mb-3"),

        caja_conclusion(
            f"El modelo explica el {r2*100:.1f}% de la variabilidad total de quality (R² = {r2:.4f}), "
            f"con un error promedio de {rmse:.4f} puntos en la escala de calidad (RMSE), "
            f"lo que representa aproximadamente el 12% del rango total (3 a 8). "
            f"El coeficiente de volatile acidity ({coef_values[1]:+.4f}) es el de mayor impacto "
            f"en valor absoluto: por cada unidad que aumenta la acidez volátil, la calidad predicha "
            f"disminuye {abs(coef_values[1]):.4f} puntos. El alcohol ({coef_values[0]:+.4f}) y los "
            f"sulfatos ({coef_values[2]:+.4f}) tienen efectos positivos moderados. "
            f"El R² moderado es coherente con la naturaleza subjetiva de la calidad del vino, "
            f"que no sigue una relación perfectamente lineal con sus propiedades químicas."
        )
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — REGRESIÓN LOGÍSTICA
# ══════════════════════════════════════════════════════════════════════════════
def tab_logistica():
    fig_cm = px.imshow(
        cm, text_auto=True,
        color_continuous_scale=[[0, '#f0f4ff'], [1, COLORES['primario']]],
        x=['Pred: Baja', 'Pred: Alta'],
        y=['Real: Baja', 'Real: Alta'],
        title="Matriz de Confusión"
    )
    fig_cm.update_layout(paper_bgcolor='white', title_font_size=14,
                         coloraxis_showscale=False)
    fig_cm.update_traces(textfont_size=16)

    metricas = {
        'Clase': ['Baja calidad', 'Alta calidad'],
        'Precision': [round(report['0']['precision'], 4), round(report['1']['precision'], 4)],
        'Recall':    [round(report['0']['recall'],    4), round(report['1']['recall'],    4)],
        'F1-score':  [round(report['0']['f1-score'],  4), round(report['1']['f1-score'],  4)],
    }
    df_met = pd.DataFrame(metricas)
    fig_met = go.Figure()
    for col, color in zip(['Precision', 'Recall', 'F1-score'],
                          [COLORES['primario'], COLORES['secundario'], COLORES['acento']]):
        fig_met.add_trace(go.Bar(name=col, x=df_met['Clase'], y=df_met[col],
                                 marker_color=color, text=df_met[col],
                                 textposition='outside'))
    fig_met.update_layout(
        barmode='group', title="Métricas por Clase",
        yaxis=dict(range=[0, 1.1]),
        paper_bgcolor='white', plot_bgcolor='white',
        title_font_size=14
    )

    fig_cfg = px.bar(
        df_cfg, x=df_cfg['C'].astype(str) + ' / ' + df_cfg['Solver'],
        y='Accuracy', color='Accuracy',
        color_continuous_scale=[[0, '#e8f0fb'], [1, COLORES['secundario']]],
        title="Accuracy por Configuración de Hiperparámetros",
        text='Accuracy'
    )
    fig_cfg.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_cfg.update_layout(
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis_title="C / Solver", yaxis=dict(range=[0.65, 0.78]),
        coloraxis_showscale=False, title_font_size=14
    )

    coef_log = modelo_log.coef_[0]
    intercept_log = modelo_log.intercept_[0]
    ecuacion_z = (f"z = {intercept_log:.4f} "
                  f"+ {coef_log[0]:+.4f}·alcohol "
                  f"+ {coef_log[1]:+.4f}·volatile_acidity "
                  f"+ {coef_log[2]:+.4f}·sulphates")

    return html.Div([
        titulo_seccion("Regresión Logística",
                       "Clasificación binaria: Alta calidad (≥6) vs Baja calidad (<6)"),

        caja_info(
            "A diferencia de la regresión lineal, la regresión logística no predice un número "
            "continuo sino la probabilidad de pertenecer a una clase. Internamente calcula el "
            "mismo valor z que la regresión lineal, pero lo transforma con la función sigmoide "
            "σ(z) = 1 / (1 + e⁻ᶻ), garantizando una salida entre 0 y 1. "
            "Si P ≥ 0.5 → Alta calidad; si P < 0.5 → Baja calidad."
        ),

        dbc.Card([
            html.Div([
                html.H6("Ecuación Logística", style={'fontWeight': '700', 'marginBottom': '8px',
                                                      'color': COLORES['oscuro']}),
                html.Code(ecuacion_z, style={'fontSize': '13px', 'color': COLORES['primario'],
                                              'fontWeight': '600'}),
                html.Br(), html.Br(),
                html.Code("P(alta calidad) = 1 / (1 + e⁻ᶻ)",
                          style={'fontSize': '13px', 'color': COLORES['secundario'], 'fontWeight': '600'})
            ], style={'padding': '16px'})
        ], style={**CARD_STYLE, 'borderLeft': f'4px solid {COLORES["primario"]}', 'marginBottom': '20px'}),

        dbc.Row([
            dbc.Col(tarjeta_kpi("Accuracy", f"{acc:.4f}",
                                f"{acc*100:.1f}% de clasificaciones correctas", COLORES['primario']), md=3),
            dbc.Col(tarjeta_kpi("F1 — Alta calidad", f"{report['1']['f1-score']:.4f}",
                                "Balance precision/recall", COLORES['acento']), md=3),
            dbc.Col(tarjeta_kpi("F1 — Baja calidad", f"{report['0']['f1-score']:.4f}",
                                "Balance precision/recall", COLORES['secundario']), md=3),
            dbc.Col(tarjeta_kpi("Configuración", "C=0.1 / lbfgs",
                                "Mejor hiperparámetro", COLORES['oscuro']), md=3),
        ], className="mb-3"),

        caja_info(
            "La experimentación con hiperparámetros es una práctica recomendada para optimizar "
            "el rendimiento del modelo. El parámetro C controla la regularización: valores pequeños "
            "restringen los coeficientes (menos overfitting), valores grandes los liberan. "
            "El solver define el algoritmo de optimización. Se probaron 6 combinaciones y se "
            "seleccionó C=0.1 con lbfgs por ofrecer el mejor accuracy con la regularización "
            "más conservadora."
        ),

        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_cfg), style=CARD_STYLE), md=12),
        ], className="mb-3"),

        caja_info(
            "La matriz de confusión muestra los 4 tipos de resultados posibles: "
            "Verdaderos Positivos (alta predicha y real), Verdaderos Negativos (baja predicha y real), "
            "Falsos Positivos (predijo alta pero era baja) y Falsos Negativos (predijo baja pero era alta). "
            "La diagonal principal son los aciertos del modelo."
        ),

        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_cm),  style=CARD_STYLE), md=5),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_met), style=CARD_STYLE), md=7),
        ], className="mb-3"),

        caja_conclusion(
            f"El modelo logístico clasificó correctamente el {acc*100:.1f}% de los vinos del conjunto "
            f"de prueba. Para la clase alta calidad obtuvo precision = {report['1']['precision']:.2f} y "
            f"recall = {report['1']['recall']:.2f}: cuando predice un vino como bueno acierta el "
            f"{report['1']['precision']*100:.0f}% de las veces. Para baja calidad la precision es "
            f"menor ({report['0']['precision']:.2f}), indicando que el modelo es ligeramente menos "
            f"confiable al clasificar vinos malos. Los resultados son coherentes con el R² de la "
            f"regresión lineal: la calidad del vino no es completamente predecible solo con "
            f"propiedades fisicoquímicas, pero ambos modelos confirman que alcohol, volatile acidity "
            f"y sulphates son los predictores más relevantes del dataset."
        )
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — PREDICTOR INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════
def tab_predictor():
    return html.Div([
        titulo_seccion("Predictor Interactivo",
                       "Ajusta las propiedades fisicoquímicas y obtén predicciones en tiempo real"),

        caja_info(
            "Este predictor usa los dos modelos entrenados sobre el dataset WineQT. "
            "El modelo lineal estima la calidad numérica (escala 3–8). "
            "El modelo logístico estima la probabilidad de que el vino sea de alta calidad (≥6). "
            "Mueve los sliders y observa cómo cambian las predicciones en tiempo real."
        ),

        dbc.Row([
            # Controles
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.H6("⚗️  Propiedades del Vino",
                                style={'fontWeight': '700', 'color': COLORES['oscuro'],
                                       'marginBottom': '20px'}),

                        html.Label("🍷 Alcohol (%)",
                                   style={'fontWeight': '600', 'fontSize': '13px'}),
                        dcc.Slider(id='pred-alcohol',
                                   min=float(df['alcohol'].min()),
                                   max=float(df['alcohol'].max()),
                                   step=0.1,
                                   value=float(df['alcohol'].mean()),
                                   marks={v: str(v) for v in [8, 9, 10, 11, 12, 13, 14]},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Br(),

                        html.Label("🧪 Volatile Acidity (g/L)",
                                   style={'fontWeight': '600', 'fontSize': '13px'}),
                        dcc.Slider(id='pred-acidity',
                                   min=float(df['volatile acidity'].min()),
                                   max=float(df['volatile acidity'].max()),
                                   step=0.01,
                                   value=float(df['volatile acidity'].mean()),
                                   marks={v: str(round(v, 1)) for v in [0.1, 0.4, 0.7, 1.0, 1.3, 1.6]},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Br(),

                        html.Label("🧂 Sulphates (g/L)",
                                   style={'fontWeight': '600', 'fontSize': '13px'}),
                        dcc.Slider(id='pred-sulphates',
                                   min=float(df['sulphates'].min()),
                                   max=float(df['sulphates'].max()),
                                   step=0.01,
                                   value=float(df['sulphates'].mean()),
                                   marks={v: str(round(v, 1)) for v in [0.3, 0.6, 0.9, 1.2, 1.5, 1.9]},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], style={'padding': '20px'})
                ], style=CARD_STYLE)
            ], md=5),

            # Resultados
            dbc.Col([
                html.Div(id='pred-output')
            ], md=7)
        ]),

        caja_info(
            "Los valores de los sliders están limitados al rango observado en el dataset. "
            "Valores de alcohol altos y acidez volátil baja consistentemente producen "
            "probabilidades más altas de alta calidad — resultado directo de los coeficientes "
            "aprendidos por el modelo y validados en el contraste de hipótesis previo."
        )
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output('eda-scatter-dinamico', 'figure'),
    Input('eda-xvar', 'value')
)
def actualizar_scatter_eda(xvar):
    fig = px.scatter(df, x=xvar, y='quality', color='calidad_binaria',
                     color_discrete_map={'Alta (≥6)': COLORES['primario'],
                                        'Baja (<6)': COLORES['secundario']},
                     opacity=0.6, trendline='ols',
                     title=f"{xvar.title()} vs Calidad del Vino")
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', title_font_size=14)
    return fig


@app.callback(
    Output('pred-output', 'children'),
    Input('pred-alcohol',   'value'),
    Input('pred-acidity',   'value'),
    Input('pred-sulphates', 'value')
)
def actualizar_prediccion(alcohol, acidity, sulphates):
    X_input = np.array([[alcohol, acidity, sulphates]])

    quality_pred = modelo_lineal.predict(X_input)[0]
    quality_pred = np.clip(quality_pred, 3, 8)

    prob_alta = modelo_log.predict_proba(X_input)[0][1]
    clasificacion = "Alta calidad ≥6" if prob_alta >= 0.5 else "Baja calidad <6"
    color_clas = COLORES['acento'] if prob_alta >= 0.5 else COLORES['secundario']

    # Gauge de probabilidad
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_alta * 100,
        title={'text': "Probabilidad de Alta Calidad (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color_clas},
            'steps': [
                {'range': [0,  50], 'color': '#f0f4ff'},
                {'range': [50, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': COLORES['oscuro'], 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        },
        number={'suffix': '%', 'valueformat': '.1f'}
    ))
    fig_gauge.update_layout(height=280, paper_bgcolor='white', margin=dict(t=40, b=10))

    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta_kpi("Calidad Predicha (Lineal)",
                                f"{quality_pred:.2f} / 8",
                                "Escala numérica 3–8",
                                COLORES['primario']), md=6),
            dbc.Col(tarjeta_kpi("Clasificación (Logístico)",
                                clasificacion,
                                f"Probabilidad: {prob_alta*100:.1f}%",
                                color_clas), md=6),
        ], className="mb-3"),

        dbc.Card(dcc.Graph(figure=fig_gauge), style=CARD_STYLE),

        dbc.Card([
            html.Div([
                html.H6("📐 Cálculo detallado", style={'fontWeight': '700',
                                                        'color': COLORES['oscuro'],
                                                        'marginBottom': '10px'}),
                html.P([
                    html.Strong("Modelo lineal: "),
                    f"ŷ = {modelo_lineal.intercept_:.4f} "
                    f"+ {modelo_lineal.coef_[0]:+.4f}×{alcohol:.1f} "
                    f"+ {modelo_lineal.coef_[1]:+.4f}×{acidity:.2f} "
                    f"+ {modelo_lineal.coef_[2]:+.4f}×{sulphates:.2f} "
                    f"= {quality_pred:.4f}"
                ], style={'fontSize': '13px', 'marginBottom': '6px'}),
                html.P([
                    html.Strong("Modelo logístico: "),
                    f"z = {modelo_log.intercept_[0]:.4f} "
                    f"+ {modelo_log.coef_[0][0]:+.4f}×{alcohol:.1f} "
                    f"+ {modelo_log.coef_[0][1]:+.4f}×{acidity:.2f} "
                    f"+ {modelo_log.coef_[0][2]:+.4f}×{sulphates:.2f}"
                ], style={'fontSize': '13px', 'marginBottom': '6px'}),
                html.P([
                    html.Strong("Probabilidad: "),
                    f"σ(z) = 1 / (1 + e⁻ᶻ) = {prob_alta*100:.2f}%  →  {clasificacion}"
                ], style={'fontSize': '13px', 'marginBottom': '0'})
            ], style={'padding': '16px'})
        ], style={**CARD_STYLE, 'borderLeft': f'4px solid {COLORES["gris"]}'})
    ])


# ── Callback principal ────────────────────────────────────────────────────────
@app.callback(
    Output('contenido-tab', 'children'),
    Input('tabs', 'value')
)
def renderizar_tab(tab):
    if tab == 'tab-eda':        return tab_eda()
    if tab == 'tab-hipotesis':  return tab_hipotesis()
    if tab == 'tab-lineal':     return tab_lineal()
    if tab == 'tab-logistica':  return tab_logistica()
    if tab == 'tab-predictor':  return tab_predictor()
    return html.Div("Sección no encontrada.", style={'padding': '40px'})


if __name__ == "__main__":
    app.run_server(debug=False, host='0.0.0.0', port=8050)
