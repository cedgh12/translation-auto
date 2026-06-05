"""
Pipeline complet : récupération des clés non traduites depuis Lokalise + traduction via OpenAI.
Filtre : clés avec traduction FR présente mais EN absente.
Produit un fichier JSON par langue cible dans ./translated_json/.

Clés API à renseigner dans un fichier .env à la racine du projet :
  LOKALISE_API_TOKEN=...
  OPENAI_API_KEY=...
"""

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

import requests
from openai import OpenAI


class JobCancelledError(Exception):
    """Levée lorsque le job de traduction est annulé entièrement."""


class LangSkippedError(Exception):
    """Levée quand une langue est passée (skip) — le job continue sur les suivantes."""

    def __init__(self, lang_code: str):
        self.lang_code = lang_code
        super().__init__(lang_code)


# --------------------------
# CONFIGURATION
# --------------------------
LOKALISE_PROJECT_ID = "220887816194c4f77e27a2.64829025"

OPENAI_MODEL = "gpt-4o-mini"
BATCH_SIZE = 30   # clés par requête OpenAI (augmenter si peu de clés longues)
DELAY = 0.5       # secondes entre batches pour éviter le rate-limit

# Langue de référence pour le filtre "non traduit" :
#   - "en" : clés FR sans traduction EN (cas normal)
#   - None : toutes les clés FR, sans filtre (ex : ajout d'une nouvelle langue)
#FILTER_LANG = "en"
FILTER_LANG = None

TARGET_LANGS = {
    "de": "German",
    "et": "Estonian",
    "nl": "Dutch"
}

TARGET_LANGS_backup = {
    "en": "English",
    "sv": "Swedish",
    "fi": "Finnish",
    "hu": "Hungarian",
    "sl": "Slovenian",
    "es": "Spanish",
    "cy": "Welsh",
    "cs": "Czech",
}

SYSTEM_PROMPT = """You are a professional translator working on a web application for Voltalis,
a French company that provides energy management solutions for residential and professional customers.
The application allows users (called "adhérents") to manage their connected IoT devices (heaters,
water heaters, air conditioning units) and monitor their electricity consumption.

Your task is to translate UI strings (labels, buttons, messages, tooltips) from French to the
requested target language.

Rules:
- Preserve all HTML tags (<br>, <strong>, <a href="...">, etc.) exactly as-is.
- Preserve all template variables like {firstName}, {count}, {date} exactly as-is.
- Use vocabulary appropriate for an energy/utilities web application aimed at non-technical end-users.
- Do NOT add explanations, comments, or extra keys.
- Return ONLY a valid JSON object with the same keys as the input and translated values.
"""

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "translated_json"

# --------------------------
# LOKALISE
# --------------------------
def fetch_project_languages(project_id: str, token: str) -> list[dict]:
    """Retourne la liste des langues du projet Lokalise (code ISO + nom)."""
    headers = {"X-Api-Token": token}
    all_languages = []
    page = 1

    while True:
        resp = requests.get(
            f"https://api.lokalise.com/api2/projects/{project_id}/languages",
            headers=headers,
            params={"page": page, "limit": 500},
        )
        resp.raise_for_status()
        languages = resp.json().get("languages", [])
        if not languages:
            break
        all_languages.extend(languages)
        page += 1

    return [
        {"code": lang["lang_iso"], "name": lang["lang_name"]}
        for lang in all_languages
    ]


def _check_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled and is_cancelled():
        raise JobCancelledError()


def _check_lang_skipped(
    lang_code: str,
    is_lang_skipped: Callable[[str], bool] | None,
) -> None:
    if is_lang_skipped and is_lang_skipped(lang_code):
        raise LangSkippedError(lang_code)


def build_languages_progress(
    target_langs: list[str],
    lang_names: dict[str, str],
    total_keys: int,
    *,
    completed: list[str] | None = None,
    skipped: list[str] | None = None,
    current_lang: str | None = None,
    current_processed: int = 0,
) -> list[dict]:
    """Construit le tableau de progression par langue cible."""
    completed = completed or []
    skipped = skipped or []
    result = []

    for code in target_langs:
        if code in skipped:
            status = "skipped"
            processed = 0
        elif code in completed:
            status = "completed"
            processed = total_keys
        elif code == current_lang:
            status = "translating"
            processed = current_processed
        else:
            status = "pending"
            processed = 0

        result.append({
            "code": code,
            "name": lang_names.get(code, code),
            "status": status,
            "processed_keys": processed,
            "total_keys": total_keys,
        })

    return result


def fetch_untranslated_keys(
    project_id: str,
    token: str,
    source_lang: str = "fr",
    filter_lang: str | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> OrderedDict:
    """Retourne les clés à traduire depuis source_lang.

    Si filter_lang est None, retourne toutes les clés ayant une traduction source_lang.
    Sinon, retourne uniquement celles sans traduction dans filter_lang.
    """
    headers = {"X-Api-Token": token}
    all_keys = []
    page = 1

    filter_label = f"sans {filter_lang.upper()}" if filter_lang else "toutes"
    print(f"Récupération des clés depuis Lokalise ({filter_label})...")
    while True:
        _check_cancelled(is_cancelled)
        resp = requests.get(
            f"https://api.lokalise.com/api2/projects/{project_id}/keys",
            headers=headers,
            params={"page": page, "limit": 500, "include_translations": 1},
        )
        resp.raise_for_status()
        keys = resp.json().get("keys", [])
        if not keys:
            break
        all_keys.extend(keys)
        print(f"  Page {page} : {len(keys)} clés")
        page += 1

    print(f"Total récupéré : {len(all_keys)} clés")

    result = OrderedDict()
    for key in all_keys:
        key_name = key["key_name"].get("web")
        if not key_name:
            continue

        source_text = ""
        filter_text = ""
        for t in key.get("translations", []):
            if t["language_iso"] == source_lang:
                source_text = t.get("translation", "").strip()
            elif filter_lang and t["language_iso"] == filter_lang:
                filter_text = t.get("translation", "").strip()

        if source_text and (filter_lang is None or not filter_text):
            result[key_name] = source_text

    print(f"Clés à traduire ({source_lang} présent) : {len(result)}\n")
    return result


# --------------------------
# OPENAI
# --------------------------
def translate_batch(client: OpenAI, batch: dict, lang_code: str, lang_name: str) -> dict:
    """Traduit un batch de clés {key: fr_text} vers lang_name. Retourne {key: translated_text}."""
    user_message = (
        f"Translate the following JSON values from French to {lang_name}.\n"
        f"Return a JSON object with the exact same keys.\n\n"
        f"{json.dumps(batch, ensure_ascii=False, indent=2)}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    content = response.choices[0].message.content
    translated = json.loads(content)

    # Vérification : toutes les clés du batch sont présentes
    missing = [k for k in batch if k not in translated]
    if missing:
        print(f"  [WARN] Clés manquantes dans la réponse : {missing}")
        for k in missing:
            translated[k] = batch[k]  # fallback sur le texte FR

    return translated


def translate_all(
    client: OpenAI,
    source_data: dict,
    lang_code: str,
    lang_name: str,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    is_lang_skipped: Callable[[str], bool] | None = None,
) -> dict:
    """Traduit toutes les clés par batches."""
    keys = list(source_data.keys())
    translated = {}

    for i in range(0, len(keys), BATCH_SIZE):
        _check_cancelled(is_cancelled)
        _check_lang_skipped(lang_code, is_lang_skipped)
        batch_keys = keys[i : i + BATCH_SIZE]
        batch = {k: source_data[k] for k in batch_keys}

        try:
            result = translate_batch(client, batch, lang_code, lang_name)
            translated.update(result)
            print(f"  [{lang_code}] Clés {i + 1}–{i + len(batch_keys)} OK")
        except Exception as e:
            print(f"  [{lang_code}] ERREUR clés {i + 1}–{i + len(batch_keys)} : {e}")
            translated.update(batch)

        if on_progress:
            on_progress(len(translated), len(keys))

        if i + BATCH_SIZE < len(keys):
            time.sleep(DELAY)

    return translated


def run_pipeline(
    *,
    project_id: str,
    lokalise_token: str,
    openai_api_key: str,
    source_lang: str,
    target_langs: list[str],
    filter_lang: str | None,
    lang_names: dict[str, str],
    on_status: Callable[[dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    is_lang_skipped: Callable[[str], bool] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Lance le pipeline complet avec remontée de statut optionnelle."""

    completed_langs: list[str] = []
    skipped_langs: list[str] = []

    def report_lang_progress(
        current_lang: str | None = None,
        current_processed: int = 0,
        global_processed: int | None = None,
    ):
        progress = build_languages_progress(
            target_langs,
            lang_names,
            total_keys,
            completed=completed_langs,
            skipped=skipped_langs,
            current_lang=current_lang,
            current_processed=current_processed,
        )
        payload = {
            "languages_progress": progress,
            "completed_langs": list(completed_langs),
            "current_lang": current_lang,
            "current_lang_name": lang_names.get(current_lang, current_lang) if current_lang else None,
        }
        if global_processed is not None:
            payload["processed_keys"] = global_processed
            payload["total_keys"] = total_keys
        if on_status:
            on_status(payload)

    def report(**kwargs):
        if on_status:
            on_status(kwargs)

    total_keys = 0
    report(
        status="fetching",
        processed_keys=0,
        completed_langs=[],
        error=None,
        languages_progress=build_languages_progress(
            target_langs, lang_names, 0, skipped=skipped_langs
        ),
    )

    source_data = fetch_untranslated_keys(
        project_id,
        lokalise_token,
        source_lang=source_lang,
        filter_lang=filter_lang,
        is_cancelled=is_cancelled,
    )
    total_keys = len(source_data)
    if not source_data:
        report(
            status="completed",
            total_keys=0,
            processed_keys=0,
            current_lang=None,
            current_lang_name=None,
            languages_progress=build_languages_progress(
                target_langs, lang_names, 0, skipped=skipped_langs
            ),
        )
        return

    output_dir.mkdir(exist_ok=True)
    client = OpenAI(api_key=openai_api_key)

    report(status="translating", total_keys=total_keys, processed_keys=0)
    report_lang_progress()

    for lang_code in target_langs:
        _check_cancelled(is_cancelled)

        if is_lang_skipped and is_lang_skipped(lang_code):
            if lang_code not in skipped_langs:
                skipped_langs.append(lang_code)
            report_lang_progress()
            continue

        lang_name = lang_names.get(lang_code, lang_code)
        report_lang_progress(current_lang=lang_code, current_processed=0)

        def on_progress(processed: int, total: int):
            report_lang_progress(
                current_lang=lang_code,
                current_processed=processed,
                global_processed=processed,
            )

        print(f"Traduction vers {lang_name} ({lang_code})...")
        try:
            translated = translate_all(
                client,
                source_data,
                lang_code,
                lang_name,
                on_progress=on_progress,
                is_cancelled=is_cancelled,
                is_lang_skipped=is_lang_skipped,
            )
        except LangSkippedError:
            if lang_code not in skipped_langs:
                skipped_langs.append(lang_code)
            print(f"  [{lang_code}] Passée (skip)")
            report_lang_progress()
            continue

        output_file = output_dir / f"{lang_code}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=4)
        print(f"  Sauvegardé : {output_file}\n")

        completed_langs.append(lang_code)
        report_lang_progress(global_processed=total_keys)

    report(
        status="completed",
        current_lang=None,
        current_lang_name=None,
        processed_keys=total_keys,
        languages_progress=build_languages_progress(
            target_langs,
            lang_names,
            total_keys,
            completed=completed_langs,
            skipped=skipped_langs,
        ),
    )


# --------------------------
# MAIN
# --------------------------
def main():
    lokalise_token = os.getenv("LOKALISE_API_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not lokalise_token:
        raise EnvironmentError("Variable LOKALISE_API_TOKEN non définie.")
    if not openai_key:
        raise EnvironmentError("Variable OPENAI_API_KEY non définie.")

    fr_data = fetch_untranslated_keys(
        LOKALISE_PROJECT_ID, lokalise_token, source_lang="fr", filter_lang=FILTER_LANG
    )
    if not fr_data:
        print("Aucune clé à traduire.")
        return

    lang_names = {code: name for code, name in TARGET_LANGS.items()}
    run_pipeline(
        project_id=LOKALISE_PROJECT_ID,
        lokalise_token=lokalise_token,
        openai_api_key=openai_key,
        source_lang="fr",
        target_langs=list(TARGET_LANGS.keys()),
        filter_lang=FILTER_LANG,
        lang_names=lang_names,
    )

    print("Pipeline terminé.")


if __name__ == "__main__":
    main()
