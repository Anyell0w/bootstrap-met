import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Bootstrap Estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con Tailwind-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .method-card {
        background: emerald;
        color: black;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: emerald;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: emerald;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: emerald;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Funciones Bootstrap (adaptadas del código original)


def bootstrap_media_alturas(datos, n_bootstrap=1000):
    datos = np.array(datos)
    n = len(datos)

    medias_bootstrap = []
    for i in range(n_bootstrap):
        muestra_bootstrap = np.random.choice(datos, size=n, replace=True)
        media_bootstrap = np.mean(muestra_bootstrap)
        medias_bootstrap.append(media_bootstrap)

    medias_bootstrap = np.array(medias_bootstrap)

    media_original = np.mean(datos)
    media_bootstrap_promedio = np.mean(medias_bootstrap)
    error_estandar = np.std(medias_bootstrap)

    ic_95 = np.percentile(medias_bootstrap, [2.5, 97.5])
    ic_90 = np.percentile(medias_bootstrap, [5, 95])

    return {
        'datos_originales': datos,
        'media_original': media_original,
        'medias_bootstrap': medias_bootstrap,
        'media_bootstrap_promedio': media_bootstrap_promedio,
        'error_estandar': error_estandar,
        'ic_95': ic_95,
        'ic_90': ic_90,
        'n_bootstrap': n_bootstrap
    }


def bootstrap_mediana_salarios(datos, n_bootstrap=1000):
    datos = np.array(datos)
    n = len(datos)

    medianas_bootstrap = []
    medias_bootstrap = []

    for i in range(n_bootstrap):
        muestra_bootstrap = np.random.choice(datos, size=n, replace=True)
        mediana_bootstrap = np.median(muestra_bootstrap)
        media_bootstrap = np.mean(muestra_bootstrap)
        medianas_bootstrap.append(mediana_bootstrap)
        medias_bootstrap.append(media_bootstrap)

    medianas_bootstrap = np.array(medianas_bootstrap)
    medias_bootstrap = np.array(medias_bootstrap)

    mediana_original = np.median(datos)
    media_original = np.mean(datos)

    mediana_bootstrap_promedio = np.mean(medianas_bootstrap)
    error_estandar_mediana = np.std(medianas_bootstrap)

    ic_95_mediana = np.percentile(medianas_bootstrap, [2.5, 97.5])

    return {
        'datos_originales': datos,
        'mediana_original': mediana_original,
        'media_original': media_original,
        'medianas_bootstrap': medianas_bootstrap,
        'medias_bootstrap': medias_bootstrap,
        'mediana_bootstrap_promedio': mediana_bootstrap_promedio,
        'error_estandar_mediana': error_estandar_mediana,
        'ic_95_mediana': ic_95_mediana
    }


def bootstrap_desviacion_estandar(datos, n_bootstrap=1000):
    datos = np.array(datos)
    n = len(datos)

    desviaciones_bootstrap = []

    for i in range(n_bootstrap):
        muestra_bootstrap = np.random.choice(datos, size=n, replace=True)
        desviacion_bootstrap = np.std(muestra_bootstrap, ddof=1)
        desviaciones_bootstrap.append(desviacion_bootstrap)

    desviaciones_bootstrap = np.array(desviaciones_bootstrap)

    desviacion_original = np.std(datos, ddof=1)
    desviacion_bootstrap_promedio = np.mean(desviaciones_bootstrap)
    error_estandar_desviacion = np.std(desviaciones_bootstrap)

    ic_95_desviacion = np.percentile(desviaciones_bootstrap, [2.5, 97.5])

    return {
        'datos_originales': datos,
        'desviacion_original': desviacion_original,
        'desviaciones_bootstrap': desviaciones_bootstrap,
        'desviacion_bootstrap_promedio': desviacion_bootstrap_promedio,
        'error_estandar_desviacion': error_estandar_desviacion,
        'ic_95_desviacion': ic_95_desviacion
    }


def bootstrap_proporcion_calidad(datos, n_bootstrap=1000):
    datos = np.array(datos)
    n = len(datos)

    proporciones_bootstrap = []

    for i in range(n_bootstrap):
        muestra_bootstrap = np.random.choice(datos, size=n, replace=True)
        proporcion_bootstrap = np.mean(muestra_bootstrap)
        proporciones_bootstrap.append(proporcion_bootstrap)

    proporciones_bootstrap = np.array(proporciones_bootstrap)

    proporcion_original = np.mean(datos)
    proporcion_bootstrap_promedio = np.mean(proporciones_bootstrap)
    error_estandar_proporcion = np.std(proporciones_bootstrap)

    ic_95 = np.percentile(proporciones_bootstrap, [2.5, 97.5])

    return {
        'datos_originales': datos,
        'proporcion_original': proporcion_original,
        'proporciones_bootstrap': proporciones_bootstrap,
        'proporcion_bootstrap_promedio': proporcion_bootstrap_promedio,
        'error_estandar_proporcion': error_estandar_proporcion,
        'ic_95_percentil': ic_95,
        'n_total': n,
        'n_exitos': int(np.sum(datos))
    }


def bootstrap_percentil_90(datos, percentil=90, n_bootstrap=1000):
    datos = np.array(datos)
    n = len(datos)

    percentiles_bootstrap = []

    for i in range(n_bootstrap):
        muestra_bootstrap = np.random.choice(datos, size=n, replace=True)
        percentil_bootstrap = np.percentile(muestra_bootstrap, percentil)
        percentiles_bootstrap.append(percentil_bootstrap)

    percentiles_bootstrap = np.array(percentiles_bootstrap)

    percentil_original = np.percentile(datos, percentil)
    percentil_bootstrap_promedio = np.mean(percentiles_bootstrap)
    error_estandar_percentil = np.std(percentiles_bootstrap)

    ic_95 = np.percentile(percentiles_bootstrap, [2.5, 97.5])

    return {
        'datos_originales': datos,
        'percentil_objetivo': percentil,
        'percentil_original': percentil_original,
        'percentiles_bootstrap': percentiles_bootstrap,
        'percentil_bootstrap_promedio': percentil_bootstrap_promedio,
        'error_estandar_percentil': error_estandar_percentil,
        'ic_95': ic_95
    }

# Función para crear gráficos con Plotly


def crear_grafico_bootstrap(datos_originales, datos_bootstrap, titulo, valor_original, ic_95, xlabel="Valor", color="#3b82f6"):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Datos Originales", "Distribución Bootstrap"),
        horizontal_spacing=0.1
    )

    # Gráfico de datos originales
    fig.add_trace(
        go.Histogram(x=datos_originales, name="Datos Originales",
                     marker_color="skyblue", opacity=0.7),
        row=1, col=1
    )

    fig.add_vline(x=valor_original, line_dash="dash", line_color="red",
                  annotation_text=f"Original: {valor_original:.2f}", row=1, col=1)

    # Gráfico de bootstrap
    fig.add_trace(
        go.Histogram(x=datos_bootstrap, name="Bootstrap",
                     marker_color=color, opacity=0.7),
        row=1, col=2
    )

    fig.add_vline(x=np.mean(datos_bootstrap), line_dash="dash", line_color="red",
                  annotation_text=f"Bootstrap: {np.mean(datos_bootstrap):.2f}", row=1, col=2)

    fig.add_vline(x=ic_95[0], line_dash="dot", line_color="orange",
                  annotation_text=f"IC 95%: [{ic_95[0]:.2f}, {ic_95[1]:.2f}]", row=1, col=2)
    fig.add_vline(x=ic_95[1], line_dash="dot",
                  line_color="orange", row=1, col=2)

    fig.update_layout(
        title_text=titulo,
        title_x=0.5,
        showlegend=False,
        height=400,
        template="plotly_white"
    )

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="Frecuencia")

    return fig

# Interfaz principal


def main():
    st.markdown('<h1 class="main-header">🎯 Métodos Bootstrap Interactivos</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #6b7280;">
        Angello Marcelo Zamora Valencia
    </div>
    """, unsafe_allow_html=True)

    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        n_bootstrap = st.slider(
            "Número de muestras Bootstrap", 100, 5000, 1000, 100)
        st.divider()

        st.header("📋 Métodos Disponibles")
        st.markdown("""
        1. **Media** - Alturas de estudiantes
        2. **Mediana** - Salarios (datos asimétricos)
        3. **Desviación Estándar** - Tiempos
        4. **Proporción** - Control de calidad
        5. **Percentil 90** - Puntuaciones
        """)

    # Tabs para diferentes métodos
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📏 Media", "💰 Mediana", "📊 Desv. Estándar",
        "✅ Proporción", "📈 Percentil"
    ])

    with tab1:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.subheader("🎯 Bootstrap para la Media - Alturas de Estudiantes")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Configuración del ejemplo:**")

            # Opción para usar datos predeterminados o personalizados
            usar_datos_default = st.checkbox(
                "Usar datos de ejemplo", value=True, key="media_default")

            if usar_datos_default:
                alturas = [165, 170, 168, 172, 175, 169, 171, 167, 174, 173,
                           166, 168, 170, 172, 169, 171, 174, 176, 168, 170]
                st.info(f"Usando {len(alturas)} alturas de ejemplo")
            else:
                alturas_input = st.text_area(
                    "Ingresa las alturas (separadas por comas):",
                    value="165, 170, 168, 172, 175, 169, 171, 167, 174, 173",
                    key="alturas_input"
                )
                try:
                    alturas = [float(x.strip())
                               for x in alturas_input.split(',')]
                    st.success(f"✅ {len(alturas)} valores cargados")
                except:
                    st.error(
                        "❌ Formato inválido. Usa números separados por comas.")
                    alturas = []

            if st.button("🚀 Ejecutar Bootstrap", key="btn_media"):
                if len(alturas) > 0:
                    resultado = bootstrap_media_alturas(alturas, n_bootstrap)

                    # Métricas
                    st.markdown("**Resultados:**")
                    st.markdown(
                        f'<div class="metric-card"><strong>Media Original:</strong> {resultado["media_original"]:.2f} cm</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Media Bootstrap:</strong> {resultado["media_bootstrap_promedio"]:.2f} cm</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Error Estándar:</strong> {resultado["error_estandar"]:.2f} cm</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>IC 95%:</strong> [{resultado["ic_95"][0]:.2f}, {resultado["ic_95"][1]:.2f}] cm</div>', unsafe_allow_html=True)

                    # Guardar en session state
                    st.session_state['resultado_media'] = resultado

        with col2:
            if 'resultado_media' in st.session_state:
                resultado = st.session_state['resultado_media']
                fig = crear_grafico_bootstrap(
                    resultado['datos_originales'],
                    resultado['medias_bootstrap'],
                    "Bootstrap para la Media - Alturas",
                    resultado['media_original'],
                    resultado['ic_95'],
                    "Altura (cm)",
                    "#3b82f6"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.subheader("💰 Bootstrap para la Mediana - Salarios")

        col1, col2 = st.columns([1, 2])

        with col1:
            usar_datos_default = st.checkbox(
                "Usar datos de ejemplo", value=True, key="mediana_default")

            if usar_datos_default:
                salarios = [25000, 28000, 30000, 32000, 35000, 38000, 40000, 45000,
                            50000, 55000, 60000, 70000, 85000, 120000, 150000]
                st.info(
                    f"Usando {len(salarios)} salarios de ejemplo (datos asimétricos)")
            else:
                salarios_input = st.text_area(
                    "Ingresa los salarios (separados por comas):",
                    value="25000, 28000, 30000, 32000, 35000, 38000, 40000",
                    key="salarios_input"
                )
                try:
                    salarios = [float(x.strip())
                                for x in salarios_input.split(',')]
                    st.success(f"✅ {len(salarios)} valores cargados")
                except:
                    st.error("❌ Formato inválido")
                    salarios = []

            if st.button("🚀 Ejecutar Bootstrap", key="btn_mediana"):
                if len(salarios) > 0:
                    resultado = bootstrap_mediana_salarios(
                        salarios, n_bootstrap)

                    st.markdown("**Resultados:**")
                    st.markdown(
                        f'<div class="metric-card"><strong>Mediana Original:</strong> ${resultado["mediana_original"]:,.0f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Media Original:</strong> ${resultado["media_original"]:,.0f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Mediana Bootstrap:</strong> ${resultado["mediana_bootstrap_promedio"]:,.0f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>IC 95%:</strong> [${resultado["ic_95_mediana"][0]:,.0f}, ${resultado["ic_95_mediana"][1]:,.0f}]</div>', unsafe_allow_html=True)

                    st.session_state['resultado_mediana'] = resultado

        with col2:
            if 'resultado_mediana' in st.session_state:
                resultado = st.session_state['resultado_mediana']
                fig = crear_grafico_bootstrap(
                    resultado['datos_originales'],
                    resultado['medianas_bootstrap'],
                    "Bootstrap para la Mediana - Salarios",
                    resultado['mediana_original'],
                    resultado['ic_95_mediana'],
                    "Salario ($)",
                    "#10b981"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.subheader("📊 Bootstrap para Desviación Estándar - Tiempos")

        col1, col2 = st.columns([1, 2])

        with col1:
            usar_datos_default = st.checkbox(
                "Usar datos de ejemplo", value=True, key="desvest_default")

            if usar_datos_default:
                tiempos = [12.5, 13.2, 11.8, 14.1, 12.9, 13.5, 12.1, 13.8, 12.7, 13.3,
                           11.9, 14.2, 12.4, 13.1, 12.8, 13.6, 12.3, 13.9, 12.6, 13.4]
                st.info(f"Usando {len(tiempos)} tiempos de producción")
            else:
                tiempos_input = st.text_area(
                    "Ingresa los tiempos (separados por comas):",
                    value="12.5, 13.2, 11.8, 14.1, 12.9, 13.5, 12.1",
                    key="tiempos_input"
                )
                try:
                    tiempos = [float(x.strip())
                               for x in tiempos_input.split(',')]
                    st.success(f"✅ {len(tiempos)} valores cargados")
                except:
                    st.error("❌ Formato inválido")
                    tiempos = []

            if st.button("🚀 Ejecutar Bootstrap", key="btn_desvest"):
                if len(tiempos) > 0:
                    resultado = bootstrap_desviacion_estandar(
                        tiempos, n_bootstrap)

                    st.markdown("**Resultados:**")
                    st.markdown(
                        f'<div class="metric-card"><strong>Desv. Est. Original:</strong> {resultado["desviacion_original"]:.3f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Desv. Est. Bootstrap:</strong> {resultado["desviacion_bootstrap_promedio"]:.3f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Error Estándar:</strong> {resultado["error_estandar_desviacion"]:.3f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>IC 95%:</strong> [{resultado["ic_95_desviacion"][0]:.3f}, {resultado["ic_95_desviacion"][1]:.3f}]</div>', unsafe_allow_html=True)

                    st.session_state['resultado_desvest'] = resultado

        with col2:
            if 'resultado_desvest' in st.session_state:
                resultado = st.session_state['resultado_desvest']
                fig = crear_grafico_bootstrap(
                    resultado['datos_originales'],
                    resultado['desviaciones_bootstrap'],
                    "Bootstrap para Desviación Estándar - Tiempos",
                    resultado['desviacion_original'],
                    resultado['ic_95_desviacion'],
                    "Desviación Estándar",
                    "#f59e0b"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.subheader("✅ Bootstrap para Proporción - Control de Calidad")

        col1, col2 = st.columns([1, 2])

        with col1:
            usar_datos_default = st.checkbox(
                "Usar datos de ejemplo", value=True, key="proporcion_default")

            if usar_datos_default:
                control_calidad = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                st.info(
                    f"Usando {len(control_calidad)} productos (0=bueno, 1=defectuoso)")
                st.write(
                    f"Productos defectuosos: {sum(control_calidad)}/{len(control_calidad)}")
            else:
                n_productos = st.number_input(
                    "Número total de productos:", min_value=10, max_value=1000, value=40)
                n_defectuosos = st.number_input(
                    "Productos defectuosos:", min_value=0, max_value=n_productos, value=5)

                control_calidad = [1] * n_defectuosos + \
                    [0] * (n_productos - n_defectuosos)
                np.random.shuffle(control_calidad)
                st.success(
                    f"✅ Datos generados: {n_defectuosos}/{n_productos} defectuosos")

            if st.button("🚀 Ejecutar Bootstrap", key="btn_proporcion"):
                if len(control_calidad) > 0:
                    resultado = bootstrap_proporcion_calidad(
                        control_calidad, n_bootstrap)

                    st.markdown("**Resultados:**")
                    st.markdown(
                        f'<div class="metric-card"><strong>Proporción Original:</strong> {resultado["proporcion_original"]:.3f} ({resultado["proporcion_original"]*100:.1f}%)</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Proporción Bootstrap:</strong> {resultado["proporcion_bootstrap_promedio"]:.3f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Error Estándar:</strong> {resultado["error_estandar_proporcion"]:.3f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>IC 95%:</strong> [{resultado["ic_95_percentil"][0]:.3f}, {resultado["ic_95_percentil"][1]:.3f}]</div>', unsafe_allow_html=True)

                    st.session_state['resultado_proporcion'] = resultado

        with col2:
            if 'resultado_proporcion' in st.session_state:
                resultado = st.session_state['resultado_proporcion']
                fig = crear_grafico_bootstrap(
                    resultado['datos_originales'],
                    resultado['proporciones_bootstrap'],
                    "Bootstrap para Proporción - Control de Calidad",
                    resultado['proporcion_original'],
                    resultado['ic_95_percentil'],
                    "Proporción de Defectos",
                    "#ef4444"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="method-card">', unsafe_allow_html=True)
        st.subheader("📈 Bootstrap para Percentil 90 - Puntuaciones")

        col1, col2 = st.columns([1, 2])

        with col1:
            percentil_objetivo = st.slider(
                "Percentil objetivo:", 10, 99, 90, 5)

            usar_datos_default = st.checkbox(
                "Usar datos de ejemplo", value=True, key="percentil_default")

            if usar_datos_default:
                puntuaciones = [65, 70, 72, 75, 78, 80, 82, 84, 85, 87, 88, 89, 90, 91, 92,
                                93, 94, 95, 96, 97, 98, 85, 88, 91, 93, 76, 79, 81, 83, 86]
                st.info(f"Usando {len(puntuaciones)} puntuaciones de ejemplo")
            else:
                puntuaciones_input = st.text_area(
                    "Ingresa las puntuaciones (separadas por comas):",
                    value="65, 70, 72, 75, 78, 80, 82, 84, 85, 87",
                    key="puntuaciones_input"
                )
                try:
                    puntuaciones = [float(x.strip())
                                    for x in puntuaciones_input.split(',')]
                    st.success(f"✅ {len(puntuaciones)} valores cargados")
                except:
                    st.error("❌ Formato inválido")
                    puntuaciones = []

            if st.button("🚀 Ejecutar Bootstrap", key="btn_percentil"):
                if len(puntuaciones) > 0:
                    resultado = bootstrap_percentil_90(
                        puntuaciones, percentil_objetivo, n_bootstrap)

                    st.markdown("**Resultados:**")
                    st.markdown(
                        f'<div class="metric-card"><strong>Percentil {percentil_objetivo} Original:</strong> {resultado["percentil_original"]:.1f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Percentil {percentil_objetivo} Bootstrap:</strong> {resultado["percentil_bootstrap_promedio"]:.1f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>Error Estándar:</strong> {resultado["error_estandar_percentil"]:.2f}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="metric-card"><strong>IC 95%:</strong> [{resultado["ic_95"][0]:.1f}, {resultado["ic_95"][1]:.1f}]</div>', unsafe_allow_html=True)

                    st.session_state['resultado_percentil'] = resultado

        with col2:
            if 'resultado_percentil' in st.session_state:
                resultado = st.session_state['resultado_percentil']
                fig = crear_grafico_bootstrap(
                    resultado['datos_originales'],
                    resultado['percentiles_bootstrap'],
                    f"Bootstrap para Percentil {resultado['percentil_objetivo']} - Puntuaciones",
                    resultado['percentil_original'],
                    resultado['ic_95'],
                    f"Percentil {resultado['percentil_objetivo']}",
                    "#8b5cf6"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        🎯 Bootstrap Estadístico - Herramienta interactiva para métodos de remuestreo<br>
        Desarrollado con Streamlit y Plotly
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
