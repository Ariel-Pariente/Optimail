import re
import unicodedata
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types


MODEL_NAME = "gemini-3-flash-preview"
REQUIRED_COLUMNS = ["Email", "Prenom", "Entreprise", "Commentaire"]


def init_session_state() -> None:
    """Initialise toutes les clés de session nécessaires."""
    defaults = {
        "drafts_by_row": {},
        "uploaded_df": None,
        "last_uploaded_signature": None,
        "current_uploaded_signature": None,
        "csv_editor_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def is_valid_email(email: str) -> bool:
    """Validation simple d'un format email."""
    if not isinstance(email, str):
        return False
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(pattern, email) is not None


def dataframe_signature(df: pd.DataFrame) -> str:
    """
    Crée une signature stable pour détecter un nouveau CSV.
    Utilisée pour éviter d'écraser des modifications utilisateur inutilement.
    """
    return f"{len(df)}::{','.join(df.columns)}::{hash(tuple(df.fillna('').astype(str).to_numpy().flatten()))}"


def uploaded_file_signature(uploaded_file) -> str:
    """Crée une signature du fichier uploadé pour détecter un nouveau CSV."""
    # getvalue() permet une signature stable (nom seul n'est pas fiable).
    content = uploaded_file.getvalue()
    return f"{uploaded_file.name}::{len(content)}::{hash(content)}"


def validate_dataframe(df: pd.DataFrame) -> Optional[str]:
    """Vérifie la présence des colonnes attendues."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return f"Colonnes manquantes dans le CSV : {', '.join(missing)}"
    return None


def normalize_column_name(name: str) -> str:
    """Normalise un nom de colonne (accents, espaces, casse, ponctuation)."""
    if name is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(name))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.replace("\ufeff", "").strip().lower()
    normalized = re.sub(r"[\s\-_]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]", "", normalized)
    return normalized


def normalize_leads_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepte plusieurs variantes de colonnes et renomme vers le format cible:
    Email, Entreprise, Commentaire.
    """
    aliases = {
        "Email": {"email", "mail", "courriel", "e-mail", "adresseemail", "contactemail"},
        "Prenom": {"prenom", "firstname", "first_name", "name", "nomcontact", "contact"},
        "Entreprise": {"entreprise", "societe", "company", "organisation", "nomentreprise", "raisonsociale"},
        "Commentaire": {"commentaire", "commentaires", "besoin", "message", "notes", "note", "remarque", "remarques"},
    }

    normalized_to_original = {normalize_column_name(col): col for col in df.columns}
    rename_map = {}

    for target, options in aliases.items():
        normalized_options = {normalize_column_name(opt) for opt in options}
        # Si la colonne cible existe déjà exactement, on la garde.
        if target in df.columns:
            continue
        # Sinon, on cherche une variante.
        for normalized_col, original_col in normalized_to_original.items():
            if normalized_col in normalized_options:
                rename_map[original_col] = target
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def build_prompt(template: str, prenom: str, entreprise: str, commentaire: str) -> str:
    """Construit le prompt envoyé à Gemini."""
    return (
        "Tu es un assistant commercial expert en prospection B2B post-salon CSE pour l'entreprise pousses Et Plantes.\n"
        "Rédige uniquement le corps d'email en français, ton professionnel, naturel et personnalisé.\n"
        "N'inclus pas d'objet, de signature automatique, ni de markdown.\n\n"
        f"Template de base:\n{template}\n\n"
        f"Prénom du contact: {prenom}\n"
        f"Entreprise destinataire du mail: {entreprise}\n"
        f"Commentaire/Besoin: {commentaire}\n"
    )


def generate_email_draft(
    api_key: str, template: str, prenom: str, entreprise: str, commentaire: str
) -> str:
    """Appelle Gemini et renvoie un brouillon de mail."""
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(
        template=template,
        prenom=prenom,
        entreprise=entreprise,
        commentaire=commentaire,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return (response.text or "").strip()


def render_sidebar() -> Dict[str, str]:
    """Affiche la configuration API dans la sidebar."""
    st.sidebar.header("Configuration")

    api_key = st.sidebar.text_input("Clé API Gemini", type="password")

    return {
        "api_key": api_key.strip(),
    }


def render_copy_button(text: str, key: str) -> None:
    """Affiche un bouton HTML qui copie le texte dans le presse-papiers du navigateur."""
    safe_text = (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )
    button_id = f"copy_btn_{key}"
    html = f"""
    <button id="{button_id}" onclick="navigator.clipboard.writeText(`{safe_text}`)">
      Copier
    </button>
    """
    components.html(html, height=36)


def render_inputs() -> Optional[pd.DataFrame]:
    """Affiche la zone d'upload et le template."""
    st.subheader("1) Inputs")
    uploaded_file = st.file_uploader("Uploader un CSV de leads", type=["csv"])
    template = st.text_area(
        "Template de l'email",
        height=180,
        placeholder=(
            "Exemple: Rédige un email de relance post-salon CSE, "
            "mets en avant notre solution RH et termine avec un CTA pour un RDV de 20 minutes."
        ),
        key="email_template",
    )

    if uploaded_file is None:
        st.session_state.current_uploaded_signature = None
        st.info("Chargez un fichier CSV pour commencer.")
        return None

    file_sig = uploaded_file_signature(uploaded_file)

    # Lecture robuste: gère BOM UTF-8 + séparateurs ; ou , automatiques.
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, dtype=str, encoding="utf-8-sig")
        except Exception as exc:
            st.error(f"Impossible de lire le CSV : {exc}")
            return None

    df = normalize_leads_columns(df)

    validation_error = validate_dataframe(df)
    if validation_error:
        st.error(validation_error)
        return None

    # Nettoyage léger pour éviter les NaN dans l'UI.
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Initialise l'éditeur uniquement quand un nouveau fichier est uploadé.
    if st.session_state.current_uploaded_signature != file_sig or st.session_state.csv_editor_df is None:
        st.session_state.csv_editor_df = df[REQUIRED_COLUMNS].copy()
        st.session_state.current_uploaded_signature = file_sig

    st.markdown("### Aperçu CSV (modifiable)")
    edited_df = st.data_editor(
        st.session_state.csv_editor_df,
        key="csv_input_editor",
        use_container_width=True,
        num_rows="dynamic",
    )
    edited_df = edited_df.copy()
    for col in REQUIRED_COLUMNS:
        edited_df[col] = edited_df[col].fillna("").astype(str).str.strip()

    st.session_state.csv_editor_df = edited_df
    st.session_state.uploaded_df = edited_df
    st.caption(f"{len(df)} leads détectés.")
    _ = template  # clé conservée dans st.session_state["email_template"]
    return edited_df


def generate_drafts_for_dataframe(df: pd.DataFrame, api_key: str, template: str) -> None:
    """Génère les brouillons pour chaque lead et stocke en session_state."""
    for idx, row in df.iterrows():
        row_id = str(idx)
        email = row["Email"]
        prenom = row["Prenom"]
        entreprise = row["Entreprise"]
        commentaire = row["Commentaire"]

        with st.spinner(f"Génération pour {prenom} - {entreprise} ({email})..."):
            draft = generate_email_draft(
                api_key=api_key,
                template=template,
                prenom=prenom,
                entreprise=entreprise,
                commentaire=commentaire,
            )

        # La clé par ligne évite les collisions si des emails se ressemblent/changent.
        st.session_state.drafts_by_row[row_id] = draft


def render_output_board(df: pd.DataFrame) -> None:
    """Affiche le tableau de bord interactif et la copie ligne par ligne."""
    st.subheader("2) Brouillons personnalisés")

    header_cols = st.columns([2, 2, 4, 1.5])
    header_cols[0].markdown("**Lead**")
    header_cols[1].markdown("**Commentaire initial**")
    header_cols[2].markdown("**Preview éditable**")
    header_cols[3].markdown("**Action**")

    st.markdown("---")

    for i, row in df.iterrows():
        row_id = str(i)
        email = row["Email"]
        prenom = row["Prenom"]
        entreprise = row["Entreprise"]
        commentaire = row["Commentaire"]

        c1, c2, c3, c4 = st.columns([2, 2, 4, 1.5], vertical_alignment="top")

        c1.write(f"**{prenom}** - {entreprise}")
        c1.caption(email)

        c2.write(commentaire or "_Aucun commentaire_")

        draft_key = f"draft_editor_{row_id}"
        if draft_key not in st.session_state:
            st.session_state[draft_key] = st.session_state.drafts_by_row.get(row_id, "")

        edited_content = c3.text_area(
            "Email généré",
            key=draft_key,
            height=180,
            label_visibility="collapsed",
        )

        # Synchronisation bidirectionnelle pour garder la source de vérité stable.
        st.session_state.drafts_by_row[row_id] = edited_content

        if edited_content.strip():
            render_copy_button(text=edited_content, key=row_id)
            c4.download_button(
                "Télécharger .txt",
                data=edited_content,
                file_name=f"email_{prenom}_{entreprise}.txt".replace(" ", "_"),
                mime="text/plain",
                key=f"download_btn_{row_id}",
                use_container_width=True,
            )
        else:
            c4.info("Texte vide")

        st.markdown("---")


def main() -> None:
    st.set_page_config(page_title="Prospection CSE - Gemini", layout="wide")
    st.title("Générateur d'emails de prospection post-salon CSE")
    st.caption("Générez et copiez des emails hyper-personnalisés (sans envoi automatique).")

    init_session_state()
    api_conf = render_sidebar()
    df = render_inputs()

    if df is None:
        return

    current_signature = st.session_state.current_uploaded_signature or dataframe_signature(df)
    if st.session_state.last_uploaded_signature != current_signature:
        # Nouveau fichier détecté: reset propre des états associés.
        st.session_state.drafts_by_row = {}
        st.session_state.last_uploaded_signature = current_signature

    if st.button("Générer les brouillons", type="primary"):
        template = st.session_state.get("email_template", "").strip()
        if not api_conf["api_key"]:
            st.error("Merci de renseigner la clé API Gemini dans la barre latérale.")
        elif not template:
            st.error("Merci de renseigner le template de l'email.")
        else:
            try:
                generate_drafts_for_dataframe(df=df, api_key=api_conf["api_key"], template=template)
                st.success("Brouillons générés avec succès.")
            except Exception as exc:
                st.error(f"Erreur durant la génération Gemini : {exc}")

    if st.session_state.drafts_by_row:
        render_output_board(df=df)
    else:
        st.info("Cliquez sur 'Générer les brouillons' pour afficher la prévisualisation.")


if __name__ == "__main__":
    main()
