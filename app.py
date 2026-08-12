import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import networkx as nx
import hashlib

# Set page configuration
st.set_page_config(
    page_title="Causal Inference Lab - Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Academic Theme (combines dark-card colors & light text colors)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #38bdf8;
        font-weight: 600;
        margin-top: 0px;
    }
    
    /* Custom Card Style */
    .custom-card {
        background-color: #1e293b;
        color: #f1f5f9 !important;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .custom-card ul, .custom-card ol, .custom-card p, .custom-card li, .custom-card span {
        color: #f1f5f9 !important;
    }
    
    .custom-card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 12px;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }
    
    .highlight-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    
    .math-expr {
        font-family: 'Outfit', monospace;
        color: #fb7185;
        background-color: #311b22;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA GENERATION FUNCTIONS -----------------
def generate_spurious_icecream(N):
    np.random.seed(42)
    temp = np.random.normal(24, 6, N)
    ice_cream = 120 + 8.5 * temp + np.random.normal(0, 15, N)
    shark_attacks = 1.2 + 0.35 * temp + np.random.normal(0, 1.2, N)
    ice_cream = np.clip(ice_cream, 10, None)
    shark_attacks = np.clip(shark_attacks, 0, None)
    
    return pd.DataFrame({
        'Temperatura (°C)': np.round(temp, 1),
        'Vendite Gelati (€)': np.round(ice_cream, 2),
        'Attacchi di Squali': np.round(shark_attacks).astype(int)
    })

def generate_school_context(N):
    np.random.seed(42)
    lse = np.random.normal(0, 1, N)
    study_hours = 8 + 2.4 * lse + np.random.normal(0, 2, N)
    study_hours = np.clip(study_hours, 1, 24)
    grade = 18 + 0.35 * study_hours + 1.8 * lse + np.random.normal(0, 1.2, N)
    grade = np.clip(grade, 18, 30)
    
    return pd.DataFrame({
        'Stato Socio-Economico': np.round(lse, 2),
        'Ore di Studio': np.round(study_hours, 1),
        'Voto Esame': np.round(grade, 1)
    })

def generate_ai_intervention(N):
    np.random.seed(42)
    motivation = np.random.normal(6, 1.5, N)
    motivation = np.clip(motivation, 1, 10)
    ai_usage = 14 - 1.6 * motivation + np.random.normal(0, 1.5, N)
    ai_usage = np.clip(ai_usage, 0, 18)
    grade = 40 + 4.2 * motivation + 1.1 * ai_usage + np.random.normal(0, 3, N)
    grade = np.clip(grade, 30, 100)
    
    return pd.DataFrame({
        'Motivazione': np.round(motivation, 2),
        'Uso AI (ore/sett)': np.round(ai_usage, 1),
        'Voto Finale (0-100)': np.round(grade, 1)
    })

# Helper to automatically load nodes/edges based on dataset selection
def set_preset_dag(dataset_name):
    if "Gelati" in dataset_name or "Squali" in dataset_name or "Esempio 1" in dataset_name:
        st.session_state.dag_nodes = ['Temperatura (°C)', 'Vendite Gelati (€)', 'Attacchi di Squali']
        st.session_state.dag_edges = [
            ('Temperatura (°C)', 'Vendite Gelati (€)'),
            ('Temperatura (°C)', 'Attacchi di Squali')
        ]
        st.session_state.treatment = 'Vendite Gelati (€)'
        st.session_state.outcome = 'Attacchi di Squali'
        st.session_state.controls = ['Temperatura (°C)']
    elif "Studio" in dataset_name or "Scolastico" in dataset_name or "Esempio 2" in dataset_name:
        st.session_state.dag_nodes = ['Stato Socio-Economico', 'Ore di Studio', 'Voto Esame']
        st.session_state.dag_edges = [
            ('Stato Socio-Economico', 'Ore di Studio'),
            ('Stato Socio-Economico', 'Voto Esame'),
            ('Ore di Studio', 'Voto Esame')
        ]
        st.session_state.treatment = 'Ore di Studio'
        st.session_state.outcome = 'Voto Esame'
        st.session_state.controls = ['Stato Socio-Economico']
    elif "AI" in dataset_name or "Uso dell'AI" in dataset_name or "Esempio 3" in dataset_name or "Performance" in dataset_name:
        st.session_state.dag_nodes = ['Motivazione', 'Uso AI (ore/sett)', 'Voto Finale (0-100)']
        st.session_state.dag_edges = [
            ('Motivazione', 'Uso AI (ore/sett)'),
            ('Motivazione', 'Voto Finale (0-100)'),
            ('Uso AI (ore/sett)', 'Voto Finale (0-100)')
        ]
        st.session_state.treatment = 'Uso AI (ore/sett)'
        st.session_state.outcome = 'Voto Finale (0-100)'
        st.session_state.controls = ['Motivazione']

    else:
        # Reset custom dataset DAG to initial state (empty edges)
        if 'df' in st.session_state and st.session_state.df is not None:
            all_cols = st.session_state.df.columns.tolist()
            st.session_state.dag_nodes = all_cols
            st.session_state.dag_edges = []
            if len(all_cols) >= 2:
                st.session_state.treatment = all_cols[0]
                st.session_state.outcome = all_cols[1]
            st.session_state.controls = []

    # Sync with widget keys to force UI update
    st.session_state.sel_treatment = st.session_state.treatment
    st.session_state.sel_outcome = st.session_state.outcome
    st.session_state.sel_controls = st.session_state.controls

# ----------------- SESSION STATE INIT -----------------
# We set spurious icecream as default on startup so the dashboard is immediately loaded and interesting
if 'df' not in st.session_state or st.session_state.df is None:
    st.session_state.df = generate_spurious_icecream(400)
    st.session_state.dataset_name = "Esempio 1: Gelati vs Squali"
    set_preset_dag(st.session_state.dataset_name)

if 'dag_edges' not in st.session_state:
    st.session_state.dag_edges = []
if 'dag_nodes' not in st.session_state:
    st.session_state.dag_nodes = []
if 'treatment' not in st.session_state:
    st.session_state.treatment = ""
if 'outcome' not in st.session_state:
    st.session_state.outcome = ""
if 'controls' not in st.session_state:
    st.session_state.controls = []

# ----------------- SIDEBAR GLOBAL CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=70)
    st.title("Causal Lab")
    st.markdown("🧬 *Pannello di Controllo Globale*")
    st.markdown("---")
    
    # Data source selection
    data_source = st.radio(
        "Seleziona Origine Dati:",
        ["Playground Dati Demo", "Carica Dataset Personalizzato"]
    )
    
    if data_source == "Playground Dati Demo":
        demo_choice = st.selectbox(
            "Scegli uno Scenario:",
            [
                "Esempio 1: Gelati vs Squali (Spuria)",
                "Esempio 2: Ore di Studio vs Voto (Contesto)",
                "Esempio 3: Uso dell'AI vs Performance (Simpson)"
            ]
        )
        
        sample_size = st.slider(
            "Dimensione Campione (N):",
            min_value=100,
            max_value=1000,
            value=400,
            step=50
        )
        
        if st.button("Genera & Ripristina Presets", use_container_width=True):
            if "Esempio 1" in demo_choice:
                st.session_state.df = generate_spurious_icecream(sample_size)
                st.session_state.dataset_name = "Esempio 1: Gelati vs Squali"
            elif "Esempio 2" in demo_choice:
                st.session_state.df = generate_school_context(sample_size)
                st.session_state.dataset_name = "Esempio 2: Ore di Studio vs Voto"
            else:
                st.session_state.df = generate_ai_intervention(sample_size)
                st.session_state.dataset_name = "Esempio 3: Uso dell'AI vs Performance"
            set_preset_dag(st.session_state.dataset_name)
            st.session_state.last_uploaded_file_hash = None
            st.success("Dati demo generati e DAG configurato!")
            st.rerun()
            
    else:
        uploaded_file = st.file_uploader("Carica CSV o Excel:", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            # Calculate file content hash to detect changes even if the filename is identical
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
            # Check if this is a newly uploaded file (content hash is different) to avoid resetting state on every rerun
            if 'last_uploaded_file_hash' not in st.session_state or st.session_state.last_uploaded_file_hash != file_hash:
                try:
                    uploaded_file.seek(0)  # Reset file pointer to the beginning of the stream
                    if uploaded_file.name.endswith(".csv"):
                        uploaded_df = pd.read_csv(uploaded_file)
                    else:
                        uploaded_df = pd.read_excel(uploaded_file)
                    
                    numeric_cols = uploaded_df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) < 2:
                        st.error("Il dataset deve contenere almeno 2 variabili numeriche!")
                    else:
                        st.session_state.df = uploaded_df[numeric_cols].dropna()
                        st.session_state.dataset_name = f"Custom: {uploaded_file.name}"
                        st.session_state.dag_nodes = numeric_cols
                        st.session_state.dag_edges = []
                        st.session_state.treatment = numeric_cols[0]
                        st.session_state.outcome = numeric_cols[1]
                        st.session_state.controls = []
                        st.session_state.sel_treatment = numeric_cols[0]
                        st.session_state.sel_outcome = numeric_cols[1]
                        st.session_state.sel_controls = []
                        st.session_state.last_uploaded_file_hash = file_hash
                        st.success("Caricamento completato!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Errore caricamento: {e}")
                
    st.markdown("---")
    # Quick reset button for current DAG
    if st.button("Ripristina DAG predefinito", use_container_width=True):
        set_preset_dag(st.session_state.dataset_name)
        st.success("DAG ripristinato!")
        st.rerun()
        
    st.markdown("<br><br><span style='font-size:0.75rem; color:#64748b;'>Sviluppato per laboratori di formazione docenti. Versione 1.4 (Failsafe DAG)</span>", unsafe_allow_html=True)

# ----------------- MAIN APP DASHBOARD LAYOUT -----------------

st.title("🧬 Laboratorio di Inferenza Causale - Dashboard Unificata")
st.markdown("### Esplora visivamente come distinguere la *Correlazione* statistica dal *Nesso di Causalità*")

# Educational overview box
st.markdown("## 📖 Fondamenti di Inferenza Causale")
with st.expander("📚 Approfondimento Didattico: La differenza fondamentale tra Correlazione e Causalità (Clicca per espandere)", expanded=True):
    col_ed1, col_ed2 = st.columns(2)
    with col_ed1:
        st.markdown(r"""
        ### 📈 La Correlazione (Associazione)
        La correlazione misura il grado di **associazione lineare** tra due variabili. Si limita ad osservare passivamente i dati così come sono:
        * **Formula probabilistica**: $P(Y \mid X)$ (Qual è la probabilità di osservare $Y$ dato che abbiamo osservato il valore $X$?)
        * **Proprietà**: È **simmetrica**. Se le vendite di gelati sono correlate agli attacchi di squali, anche gli attacchi di squali sono correlati alle vendite di gelati.
        * **Limiti**: Non descrive cosa accade se modifichiamo attivamente il sistema. Rileva pattern causati sia da nessi diretti sia da **cause comuni non osservate** (*confondenti*).
        """)
    with col_ed2:
        st.markdown(r"""
        ### 🎯 La Causalità (Intervento)
        La causalità descrive il nesso di **causa-effetto**. Risponde a domande controfattuali e simula un intervento attivo che modifica il sistema:
        * **Formula probabilistica**: $P(Y \mid do(X))$ (Qual è la probabilità di $Y$ se costringiamo attivamente $X$ ad assumere un determinato valore?)
        * **Proprietà**: È **asimmetrica**. Se lo studio causa un buon voto ($Studio \rightarrow Voto$), costringere uno studente a studiare migliorerà il voto, ma regalargli un buon voto non aumenterà le sue ore di studio passate.
        * **Do-calculus di Pearl**: Il formalismo matematico per calcolare l'effetto di un intervento ($do(X)$) eliminando l'influenza dei confondenti tramite l'aggiustamento statistico.
        """)

# ----------------- ROW 1: EXPLORATORY DATA ANALYSIS (Preview & Heatmap) -----------------
st.markdown("## 📊 Riga 1: Analisi Esplorativa dei Dati Observazionali")

col_data, col_corr = st.columns([1.1, 0.9])

with col_data:
    with st.container(height=520, border=True):
        st.subheader(f"📋 Dataset Attivo: {st.session_state.dataset_name}")
        
        tab_tbl, tab_stat = st.tabs(["Anteprima Righe", "Statistiche Descrittive"])
        with tab_tbl:
            st.dataframe(st.session_state.df, height=350, use_container_width=True)
            st.caption(f"Campione totale di {len(st.session_state.df)} righe senza valori mancanti.")
        with tab_stat:
            st.dataframe(st.session_state.df.describe(), height=350, use_container_width=True)

with col_corr:
    with st.container(height=520, border=True):
        st.subheader("🔗 Matrice di Correlazione di Pearson")
        st.markdown("Misura la forza dell'associazione lineare tra coppie di variabili (scala da -1 a 1).")
        
        corr_matrix = st.session_state.df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".3f",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            height=320,
            margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ----------------- ROW 2: CAUSAL REASONING & INFERENCE (DAG & Regression) -----------------
st.markdown("## 📐 Riga 2: Modellazione Causale & Stima degli Effetti")

col_dag, col_analysis = st.columns([1.0, 1.2])

# Build Nodes and Edges list
all_cols = st.session_state.df.columns.tolist()
if not st.session_state.dag_nodes:
    st.session_state.dag_nodes = all_cols

with col_dag:
    with st.container(height=780, border=True):
        st.subheader("📐 Editor del Grafo Causale (DAG)")
        st.markdown("Aggiungi o elimina le relazioni causali dirette ($A \\rightarrow B$) ipotizzate dal tuo modello teorico.")
        
        # 1. Display current Graphviz DAG
        if not st.session_state.dag_edges:
            st.info("Grafo vuoto. I nodi sono isolati. Aggiungi relazioni causali qui sotto o carica il DAG predefinito.")
            
        dot_str = "digraph G {\n"
        dot_str += '    bgcolor="transparent";\n'
        dot_str += '    rankdir=LR;\n'
        dot_str += '    node [shape=box, style="filled,rounded", color="#38bdf8", fontcolor="#0f172a", fillcolor="#38bdf8", fontname="Helvetica-Bold", height=0.4];\n'
        dot_str += '    edge [color="#fb7185", penwidth=2.5, arrowsize=1.2];\n'
        
        for node in st.session_state.dag_nodes:
            dot_str += f'    "{node}" [label="{node}"];\n'
        for edge in st.session_state.dag_edges:
            dot_str += f'    "{edge[0]}" -> "{edge[1]}";\n'
        dot_str += "}"
        
        st.graphviz_chart(dot_str)
        
        # 2. Controls to Edit Edges
        st.markdown("---")
        st.markdown("**Modifica Relazioni:**")
        c_src, c_tgt = st.columns(2)
        with c_src:
            src = st.selectbox("Variabile Causa:", all_cols, key="dag_src")
        with c_tgt:
            tgt = st.selectbox("Variabile Effetto:", [c for c in all_cols if c != src], key="dag_tgt")
            
        if st.button("Aggiungi Relazione (Causa ➔ Effetto)", use_container_width=True):
            new_edge = (src, tgt)
            if new_edge in st.session_state.dag_edges:
                st.warning("Relazione già esistente!")
            else:
                # Check for cycles
                temp_g = nx.DiGraph(st.session_state.dag_edges)
                temp_g.add_edge(src, tgt)
                if not nx.is_directed_acyclic_graph(temp_g):
                    st.error("Errore: Impossibile aggiungere l'arco! Creerebbe un ciclo chiuso, vietato nei DAG.")
                else:
                    st.session_state.dag_edges.append(new_edge)
                    st.success("Relazione aggiunta!")
                    st.rerun()
                    
        # List and remove edges
        if st.session_state.dag_edges:
            st.markdown("**Gestisci archi esistenti:**")
            for edge in st.session_state.dag_edges:
                col_txt, col_del = st.columns([3, 1])
                with col_txt:
                    st.markdown(f"⚙️ `{edge[0]}` ➔ `{edge[1]}`")
                with col_del:
                    if st.button("Elimina", key=f"del_{edge[0]}_{edge[1]}"):
                        st.session_state.dag_edges.remove(edge)
                        st.rerun()

with col_analysis:
    with st.container(height=780, border=True):
        st.subheader("⚖️ Analizzatore: Correlazione vs Causalità")
        st.markdown("Confronta la regressione semplice (correlazione) con l'effetto causale stimato condizionando per i confondenti teorizzati.")
        
        # Dropdowns for variable configuration
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            try:
                x_idx = all_cols.index(st.session_state.treatment)
            except ValueError:
                x_idx = 0
            treatment_var = st.selectbox("Variabile Trattamento (X):", all_cols, index=x_idx, key="sel_treatment")
            st.session_state.treatment = treatment_var
        with col_y:
            try:
                y_idx = all_cols.index(st.session_state.outcome)
            except ValueError:
                y_idx = min(1, len(all_cols)-1)
            outcome_var = st.selectbox("Variabile Risultato (Y):", [c for c in all_cols if c != treatment_var], index=min(y_idx, len(all_cols)-2), key="sel_outcome")
            st.session_state.outcome = outcome_var
        with col_z:
            possible_controls = [c for c in all_cols if c != treatment_var and c != outcome_var]
            
            # Suggested controls: parents of X that also have a path/effect on Y
            suggested_controls = []
            for node in possible_controls:
                has_to_t = (node, treatment_var) in st.session_state.dag_edges
                has_to_o = (node, outcome_var) in st.session_state.dag_edges
                if has_to_t and has_to_o:
                    suggested_controls.append(node)
                    
            # Clean defaults of session state controls if they don't fit current selections
            st.session_state.controls = [c for c in st.session_state.controls if c in possible_controls]
            if not st.session_state.controls and suggested_controls:
                st.session_state.controls = suggested_controls
                
            controls_var = st.multiselect(
                "Controllo Confondenti (Z):",
                possible_controls,
                default=st.session_state.controls,
                key="sel_controls"
            )
            st.session_state.controls = controls_var

        st.markdown("---")
        
        # Regression Calculations
        df = st.session_state.df
        
        # Model 1: Simple
        X_simple = sm.add_constant(df[treatment_var])
        model_simple = sm.OLS(df[outcome_var], X_simple).fit()
        coef_simple = model_simple.params[treatment_var]
        p_simple = model_simple.pvalues[treatment_var]
        r2_simple = model_simple.rsquared
        
        # Model 2: Controlled
        if controls_var:
            X_controlled = sm.add_constant(df[[treatment_var] + list(controls_var)])
            model_controlled = sm.OLS(df[outcome_var], X_controlled).fit()
            coef_ctrl = model_controlled.params[treatment_var]
            p_ctrl = model_controlled.pvalues[treatment_var]
            r2_ctrl = model_controlled.rsquared
        else:
            model_controlled = None
            coef_ctrl = coef_simple
            p_ctrl = p_simple
            r2_ctrl = r2_simple

        # Visual side-by-side tabs for plots
        tab_plot_simple, tab_plot_ctrl = st.tabs(["📉 Plot 1: Correlazione Semplice", "⚖️ Plot 2: Effetto Causale Netto (Controllato)"])
        
        with tab_plot_simple:
            fig_s = px.scatter(
                df, x=treatment_var, y=outcome_var,
                color=controls_var[0] if controls_var else None,
                color_continuous_scale="Viridis",
                opacity=0.7, trendline="ols",
                trendline_color_override="#fb7185"
            )
            fig_s.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                height=250, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_s, use_container_width=True)
            st.markdown(f"**Coeff. Correlazionale (pendenza):** `{coef_simple:.4f}` | **P-value:** `{p_simple:.4e}` | **R²:** `{r2_simple:.4f}`")
            
        with tab_plot_ctrl:
            if controls_var:
                mean_controls = df[controls_var].mean()
                test_df = pd.DataFrame({treatment_var: np.linspace(df[treatment_var].min(), df[treatment_var].max(), 100)})
                for ctrl in controls_var:
                    test_df[ctrl] = mean_controls[ctrl]
                
                test_df_const = sm.add_constant(test_df[[treatment_var] + list(controls_var)], has_constant='add')
                test_df_const = test_df_const[model_controlled.model.exog_names]
                test_df['predicted_y'] = model_controlled.predict(test_df_const)
                
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=df[treatment_var], y=df[outcome_var], mode='markers',
                    marker=dict(
                        color=df[controls_var[0]] if controls_var else "#38bdf8",
                        colorscale='Viridis', showscale=True if controls_var else False,
                        colorbar=dict(title=controls_var[0]) if controls_var else None,
                        opacity=0.5
                    ),
                    name='Osservazioni'
                ))
                fig_c.add_trace(go.Scatter(
                    x=test_df[treatment_var], y=test_df['predicted_y'], mode='lines',
                    line=dict(color='#10b981', width=3),
                    name='Effetto Causale'
                ))
                fig_c.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                    height=250, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title=treatment_var, yaxis_title=outcome_var
                )
                st.plotly_chart(fig_c, use_container_width=True)
                st.markdown(f"**Coeff. Causale (pendenza netta):** `{coef_ctrl:.4f}` | **P-value:** `{p_ctrl:.4e}` | **R²:** `{r2_ctrl:.4f}`")
            else:
                st.info("Seleziona almeno una variabile di controllo (Z) sopra per calcolare il modello causale controllato.")

        # Educational natural language feedback
        st.markdown("---")
        if "Gelati" in st.session_state.dataset_name:
            st.markdown(f"""
            ### 🍦 Analisi Didattica: Scenario Gelati vs Attacchi di Squali
            * **Il Fenomeno**: Nei dati storici, osserviamo che nei giorni in cui aumentano le vendite di gelati, aumentano anche gli attacchi di squali. La pendenza semplice del modello non controllato è pari a **`{coef_simple:.4f}`** (altamente significativa).
            * **L'Inganno (Correlazione Spuria)**: Un'interpretazione causale ingenua suggerirebbe di vietare la vendita di gelati per salvare bagnanti. Questo è un errore grossolano dovuto alla presenza di un **confondente comune (Confounder)**: la **Temperatura esterna**.
            * **La Soluzione (Backdoor Adjustment)**: Nel DAG, vediamo che la Temperatura causa sia le vendite di gelati (le persone hanno caldo e comprano gelati) sia gli attacchi di squali (le persone hanno caldo ed entrano in acqua, aumentando la probabilità di incontrare squali). La Temperatura costituisce un *cammino backdoor* aperto: 
              $$Vendite\\ Gelati \\leftarrow Temperatura \\rightarrow Attacchi\\ di\\ Squali$$
              Quando controlliamo per la Temperatura nel modello OLS, stiamo confrontando giorni che hanno la **stessa identica temperatura**. All'interno di questi giorni omogenei, la pendenza statistica delle vendite di gelati crolla a **`{coef_ctrl:.4f}`** (statisticamente non diversa da 0).
            * **Messaggio Chiave**: L'effetto causale stimato $P(Attacchi \\mid do(Gelati))$ è nullo. Controllare per il confondente ha bloccato il flusso spurio di associazione, rivelando la verità scientifica.
            """)
        elif "Scolastico" in st.session_state.dataset_name or "Studio" in st.session_state.dataset_name:
            st.markdown(f"""
            ### 🏫 Analisi Didattica: Scenario Ore di Studio vs Voto Esame
            * **Il Fenomeno**: Il modello non controllato suggerisce che per ogni ora di studio in più, il voto esame aumenta di **`{coef_simple:.4f}`** punti.
            * **L'Inganno (Omitted Variable Bias)**: Pur essendoci un effetto causale reale dello studio sul voto, il coefficiente semplice **sovrastima** questo impatto. Perché? Perché lo **Stato Socio-Economico (LSE)** delle famiglie agisce da confondente.
            * **La Soluzione (Backdoor Adjustment)**: Gli studenti provenienti da famiglie con LSE più elevato hanno spesso accesso a camere di studio silenziose, tutor privati e libri extra (che aumentano il voto esame indipendentemente dallo studio). Al contempo, hanno più tempo libero e risorse stabili che consentono loro di studiare più ore. Si ha quindi il cammino backdoor:
              $$Ore\\ di\\ Studio \\leftarrow LSE \\rightarrow Voto\\ Esame$$
              Se non controlliamo per LSE, attribuiamo erroneamente alle ore di studio parte del merito che in realtà appartiene al background familiare dello studente. Controllando per LSE nel modello OLS, la pendenza scende a **`{coef_ctrl:.4f}`**. Questo coefficiente rappresenta il vero effetto causale dello studio, depurato dal vantaggio socio-economico di partenza.
            * **Messaggio Chiave**: Ignorare i confondenti nel sistema scolastico può portare i decisori politici a sovrastimare l'efficacia di singole azioni (come le ore di studio extra) se non si interviene contemporaneamente sulle disuguaglianze di partenza.
            """)
        elif "AI" in st.session_state.dataset_name or "Simpson" in st.session_state.dataset_name or "Performance" in st.session_state.dataset_name:
            st.markdown(f"""
            ### 🧠 Analisi Didattica: Scenario Uso dell'AI vs Voto Finale (Paradosso di Simpson)
            * **Il Fenomeno**: Il modello semplice mostra una pendenza **negativa** pari a **`{coef_simple:.4f}`**. Sembra che chi usa l'AI prenda voti peggiori!
            * **L'Inganno (Paradosso di Simpson & Confondimento Negativo)**: Questo risultato è paradossale ed estremamente fuorviante. La causa di questo ribaltamento di segno è la **Motivazione dello studente**, che agisce come confondente con un forte effetto negativo sulla scelta di usare l'AI ma positivo sul voto.
            * **La Soluzione (Backdoor Adjustment)**: Gli studenti meno motivati tendono ad usare gli strumenti di AI generativa per molte ore alla settimana (spesso come scorciatoia per fare i compiti all'ultimo minuto). Gli studenti altamente motivati si affidano a metodi di studio tradizionali e usano meno ore l'AI. Poiché la Motivazione ha un impatto enorme e positivo sul voto finale, chi usa molto l'AI (gli studenti meno motivati) finisce per prendere voti mediamente inferiori. Il cammino backdoor è:
              $$Uso\\ AI \\leftarrow Motivazione \\rightarrow Voto\\ Finale$$
              Se controlliamo per la Motivazione (Plot 2), confrontiamo studenti che hanno lo **stesso identico livello di motivazione**. In questo confronto equo, scopriamo che tra due studenti ugualmente motivati, quello che usa l'AI ottiene un voto superiore! Il vero effetto causale dell'AI è infatti **positivo** e pari a **`{coef_ctrl:.4f}`**.
            * **Messaggio Chiave**: Questo scenario mostra che la correlazione semplice può non solo sovrastimare o sottostimare un effetto, ma addirittura **invertirne il segno**. Senza inferenza causale e DAG, trarremmo la conclusione errata che l'AI fa male allo studio, mentre a parità di motivazione lo studio ne beneficia.
            """)
        else:
            if controls_var:
                st.markdown(f"""
                Nel tuo dataset personalizzato:
                - **Differenza ($\\\\beta_{{corr}} - \\\\beta_{{causale}}$)**: `{coef_simple - coef_ctrl:.4f}`.
                - Il controllo per `{', '.join(controls_var)}` ha modificato la stima dell'impatto di `{treatment_var}` su `{outcome_var}`. Se la differenza è consistente, significa che esisteva un bias di confondimento rilevante.
                """)
            else:
                st.markdown("Configura le variabili di trattamento, risultato e controllo per visualizzare l'interpretazione statistica dettagliata.")
