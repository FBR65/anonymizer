import spacy
from collections import defaultdict
import dotenv
import os
import requests  # For calling LLM API
import json
import re  # For finding LLM terms

# Load environment variables
dotenv.load_dotenv()

# Versuche, das deutsche Modell zu laden. Gib eine Hilfestellung bei Fehler.
try:
    nlp = spacy.load("de_core_news_lg")
except OSError:
    print(
        "Das spaCy-Modell 'de_core_news_lg' wurde nicht gefunden."
        "Bitte herunterladen mit: python -m spacy download de_core_news_lg"
    )
    exit()


class Anonymizer:
    """
    Anonymisiert Text durch Ersetzen von erkannten Entitäten (Namen, Orte etc.)
    mit konsistenten Platzhaltern, unter Verwendung von spaCy und optional
    eines externen LLM (OpenAI-kompatibel) zur Erkennung zusätzlicher
    sensibler Daten. Verarbeitet Texte im Batch mit nlp.pipe().
    """

    def __init__(
        self,
        labels_to_anonymize=None,
        use_llm=False,
        llm_endpoint_url=None,
        llm_api_key=None,
        llm_model_name=None,
    ):
        """
        Initialisiert den Anonymisierer.

        Args:
            labels_to_anonymize (list, optional): Liste der spaCy-Entity-Labels,
                                                 die anonymisiert werden sollen.
                                                 Defaults to ['PER', 'LOC', 'GPE', 'ORG'].
            use_llm (bool): Ob das LLM zur zusätzlichen Erkennung verwendet werden soll.
            llm_endpoint_url (str): URL des OpenAI-kompatiblen Endpunkts.
                                    Erforderlich, wenn use_llm=True.
            llm_api_key (str): API-Schlüssel für den LLM-Endpunkt.
                               Erforderlich, wenn use_llm=True.
                               WARNUNG: Sicher behandeln, nicht hardcoden!
            llm_model_name (str): Name des LLM-Modells (optional, je nach API).
        """
        if labels_to_anonymize is None:
            self.labels_to_anonymize = {
                "PER",
                "LOC",
                "GPE",
                "ORG",
            }  # Set für schnelle Prüfung
        else:
            self.labels_to_anonymize = set(labels_to_anonymize)

        self.use_llm = use_llm
        self.llm_endpoint_url = llm_endpoint_url
        self.llm_api_key = llm_api_key  # Consider using env variables or secure storage
        self.llm_model_name = llm_model_name

        if self.use_llm and (not self.llm_endpoint_url or not self.llm_api_key):
            raise ValueError(
                "LLM Endpoint URL und API Key sind erforderlich, wenn use_llm=True."
            )

        # Shared state across batch processing
        self.anonymization_map = {}  # Speichert {original_text: placeholder}
        self.counters = defaultdict(
            int
        )  # Zählt Platzhalter pro Typ (z.B. PER_1, SENSITIVE_1)
        self.llm_prompt_template = """
Du bist ein KI-Assistent zur Identifizierung sensibler Daten in Texten. Deine Aufgabe ist es, zusätzliche personenbezogene Daten (PII) oder andere sensible Informationen zu finden, die möglicherweise von Standard-NER-Tools übersehen wurden. Konzentriere dich auf Dinge wie:
- E-Mail-Adressen
- Telefonnummern
- Spezifische Identifikationsnummern (z.B. Kunden-IDs, Sozialversicherungsnummern - falls im Kontext erwartet)
- Vertrauliche Projekt- oder Codenamen (falls im Kontext erkennbar)
- Potenziell übersehene Namen oder Adressteile, die sehr ungewöhnlich formatiert sind.

Gib **NUR** eine Liste der gefundenen sensiblen Textteile im JSON-Format zurück. Jeder Eintrag sollte den exakten Text enthalten, den du als sensibel identifiziert hast.

Beispiel-Format der Ausgabe:
{{
  "sensitive_terms": [
    "projekt_phoenix_alpha",
    "kunden_id_78349-B",
    "max.mustermann@email.com",
    "0176 / 123 456 78"
  ]
}}

Hier ist der Text, den du analysieren sollst:
--- TEXT START ---
{document_text}
--- TEXT END ---

Identifiziere die zusätzlichen sensiblen Daten und gib sie im oben genannten JSON-Format zurück. Wenn keine zusätzlichen sensiblen Daten gefunden werden, gib {"sensitive_terms": []} zurück.
"""

    def _generate_placeholder(self, entity_label):
        """Generiert einen neuen, eindeutigen Platzhalter für einen Entitätstyp."""
        self.counters[entity_label] += 1
        # Verwende eckige Klammern zur klaren Abgrenzung
        return f"[{entity_label}_{self.counters[entity_label]}]"

    def _call_llm_for_sensitive_data(self, text):
        """Ruft das LLM auf, um zusätzliche sensible Daten im Text zu finden."""
        if not self.use_llm or not text:
            return []

        prompt = self.llm_prompt_template.format(document_text=text)
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 150,  # Adjust as needed
            "temperature": 0.1,  # Low temperature for factual extraction
            "stop": ["---"],  # Optional: Stop sequences if needed
            # Add model name if the endpoint requires it
            **({"model": self.llm_model_name} if self.llm_model_name else {}),
        }
        # Adapt payload structure if your endpoint uses 'messages' format (like newer OpenAI APIs)
        # payload = {
        #    "model": self.llm_model_name or "default-model", # Specify model
        #    "messages": [
        #        {"role": "system", "content": "Du bist ein KI-Assistent zur Identifizierung sensibler Daten..."},
        #        {"role": "user", "content": f"Analysiere diesen Text:\n{text}"}
        #        # Refine system prompt and user content based on the template logic
        #    ],
        #     "max_tokens": 150,
        #     "temperature": 0.1,
        # }

        try:
            print("Sende Anfrage an LLM")  # Debug-Ausgabe
            response = requests.post(
                self.llm_endpoint_url, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            llm_result = response.json()

            # Extract sensitive terms based on expected JSON structure
            # This might need adjustment depending on the *exact* response format of your LLM
            # Example: Checking common OpenAI response structures
            if "choices" in llm_result and llm_result["choices"]:
                # Handle case where response is nested in 'choices' and potentially 'message' or 'text'
                content = ""
                if "text" in llm_result["choices"][0]:
                    content = llm_result["choices"][0]["text"]
                elif (
                    "message" in llm_result["choices"][0]
                    and "content" in llm_result["choices"][0]["message"]
                ):
                    content = llm_result["choices"][0]["message"]["content"]
                else:  # Fallback if structure is different
                    print(f"Warnung: Unerwartete LLM-Antwortstruktur: {llm_result}")
                    return []

                # Attempt to parse the JSON string within the content
                try:
                    # Sometimes LLMs wrap the JSON in ```json ... ``` or just output the JSON string
                    json_match = re.search(r"\{.*\}", content, re.DOTALL)
                    if json_match:
                        parsed_json = json.loads(json_match.group(0))
                        if "sensitive_terms" in parsed_json and isinstance(
                            parsed_json["sensitive_terms"], list
                        ):
                            return parsed_json["sensitive_terms"]
                        else:
                            print(
                                f"Warnung: 'sensitive_terms' nicht im erwarteten Format im LLM JSON gefunden: {parsed_json}"
                            )
                            return []
                    else:
                        print(
                            f"Warnung: Kein gültiges JSON im LLM-Antwort-Content gefunden: {content}"
                        )
                        return []
                except json.JSONDecodeError:
                    print(
                        f"Warnung: Konnte JSON aus LLM-Antwort nicht parsen: {content}"
                    )
                    return []

            else:  # Handle direct response or unexpected structure
                print(f"Warnung: Unerwartete LLM-Antwortstruktur: {llm_result}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"Fehler bei der LLM-Anfrage: {e}")
            return []
        except json.JSONDecodeError:
            print(f"Fehler beim Parsen der LLM-Antwort als JSON: {response.text}")
            return []
        except Exception as e:
            print(f"Unerwarteter Fehler bei der LLM-Verarbeitung: {e}")
            return []

    def anonymize_batch(self, texts):
        """
        Identifiziert und anonymisiert Entitäten in einer Liste von Texten (Batch).

        Args:
            texts (list[str]): Eine Liste von zu anonymisierenden Texten.

        Returns:
            list[str]: Eine Liste der anonymisierten Texte.
                       Gibt None zurück bei schwerwiegenden Fehlern.
        """
        if not texts:
            return []

        anonymized_texts = [""] * len(texts)
        all_replacements = [
            [] for _ in texts
        ]  # List to hold replacements for each text

        # --- Schritt 1: spaCy NER im Batch ---
        try:
            # Process texts using nlp.pipe for efficiency
            # as_tuples=True gives (doc, context) pairs; context isn't used here but good practice
            # We use enumerate to keep track of the original index
            docs_with_indices = [(doc, i) for i, doc in enumerate(nlp.pipe(texts))]

            for doc, i in docs_with_indices:
                doc_replacements = []
                for ent in doc.ents:
                    if ent.label_ in self.labels_to_anonymize:
                        original_text = ent.text
                        if original_text not in self.anonymization_map:
                            placeholder = self._generate_placeholder(ent.label_)
                            self.anonymization_map[original_text] = placeholder
                        else:
                            placeholder = self.anonymization_map[original_text]
                        # Store (start, end, placeholder, source) - source helps debugging overlaps
                        doc_replacements.append(
                            (ent.start_char, ent.end_char, placeholder, "spacy")
                        )
                all_replacements[i].extend(doc_replacements)

        except Exception as e:
            print(f"Fehler während der spaCy-Verarbeitung: {e}")
            # Depending on desired robustness, you might want to continue or stop here
            # For now, we'll print the error and proceed with potentially incomplete spaCy data.

        # --- Schritt 2: LLM Enhancement (pro Text) ---
        if self.use_llm:
            for i, text in enumerate(texts):
                if not text:
                    continue  # Skip empty texts

                llm_sensitive_terms = self._call_llm_for_sensitive_data(text)
                llm_doc_replacements = []

                # Keep track of spans found by spaCy to avoid simple overlaps
                spacy_spans = {
                    (r[0], r[1]) for r in all_replacements[i] if r[3] == "spacy"
                }

                for term in llm_sensitive_terms:
                    if not term or not isinstance(term, str) or term.isspace():
                        continue  # Skip empty or invalid terms

                    # Find *all* occurrences of the term in the original text
                    # Use regex finditer for potentially overlapping matches and ignore case option
                    try:
                        # Escape special regex characters in the term itself
                        escaped_term = re.escape(term)
                        for match in re.finditer(escaped_term, text, re.IGNORECASE):
                            start, end = match.span()
                            original_found_text = match.group(
                                0
                            )  # Get the exact text matched

                            # Basic Overlap Check: Skip if LLM span is identical to a spaCy span
                            if (start, end) in spacy_spans:
                                # print(f"LLM-Begriff '{term}' bei ({start},{end}) übersprungen, da von spaCy abgedeckt.")
                                continue

                            # TODO: Implement more sophisticated overlap check if needed
                            # (e.g., contains, is contained by, partial overlap)

                            # Generate placeholder using shared map/counter for consistency
                            # Use a generic label like SENSITIVE for LLM findings
                            if original_found_text not in self.anonymization_map:
                                placeholder = self._generate_placeholder("SENSITIVE")
                                self.anonymization_map[original_found_text] = (
                                    placeholder
                                )
                            else:
                                placeholder = self.anonymization_map[
                                    original_found_text
                                ]

                            # Add replacement info for this LLM finding
                            # Ensure we don't add duplicate spans from LLM finding the same term multiple times
                            if (
                                start,
                                end,
                                placeholder,
                                "llm",
                            ) not in llm_doc_replacements:
                                llm_doc_replacements.append(
                                    (start, end, placeholder, "llm")
                                )

                    except re.error as re_err:
                        print(
                            f"Regex-Fehler beim Suchen nach LLM-Begriff '{term}': {re_err}"
                        )
                        continue  # Skip this term if regex fails

                # Add the LLM replacements to the list for this document
                all_replacements[i].extend(llm_doc_replacements)

        # --- Schritt 3: Anonymisierung durchführen (pro Text) ---
        for i, original_text in enumerate(texts):
            current_doc_replacements = all_replacements[i]
            if not current_doc_replacements:
                anonymized_texts[i] = original_text  # No changes needed
                continue

            # De-duplication: Remove exact duplicate tuples (start, end, placeholder, source)
            # Note: This doesn't resolve complex overlaps yet.
            unique_replacements = sorted(
                list(set(current_doc_replacements)),
                key=lambda item: item[0],
                reverse=True,
            )

            # Simple Overlap Handling: If two spans have the same start or end, prioritize longer one?
            # Or just let reverse sort replacement handle it (last replacement wins on overlap)
            # For now, use the simpler reverse sort approach. More advanced logic could go here.

            processed_text = original_text
            for (
                start,
                end,
                placeholder,
                _,
            ) in unique_replacements:  # Source not needed for replacement itself
                # Boundary check (although unlikely if indices come from spaCy/regex on original text)
                if start < 0 or end > len(processed_text):
                    print(
                        f"Warnung: Ungültiger Indexbereich ({start},{end}) für Textlänge {len(processed_text)}. Ersetzung übersprungen."
                    )
                    continue
                # Ensure start is not greater than end
                if start >= end:
                    # This might happen if LLM returns empty strings or spaCy finds zero-width entities
                    # print(f"Warnung: Startindex {start} >= Endindex {end}. Ersetzung übersprungen.")
                    continue

                processed_text = (
                    processed_text[:start] + placeholder + processed_text[end:]
                )

            anonymized_texts[i] = processed_text

        return anonymized_texts

    def anonymize_file(self, input_filepath, output_filepath):
        """Liest eine Datei, anonymisiert den Inhalt (als einzelnes Dokument)
        und schreibt in eine neue Datei."""
        try:
            with open(input_filepath, "r", encoding="utf-8") as infile:
                # Read the whole file as a single text document
                content = infile.read()
        except FileNotFoundError:
            print(f"Fehler: Eingabedatei '{input_filepath}' nicht gefunden.")
            return
        except Exception as e:
            print(f"Fehler beim Lesen der Datei '{input_filepath}': {e}")
            return

        # Process the single file content as a batch of one
        anonymized_content_list = self.anonymize_batch([content])

        if anonymized_content_list is None or not anonymized_content_list:
            print("Fehler bei der Anonymisierung der Datei.")
            return

        anonymized_content = anonymized_content_list[
            0
        ]  # Get the result for the single document

        try:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, "w", encoding="utf-8") as outfile:
                outfile.write(anonymized_content)
            print(f"Anonymisierte Datei gespeichert unter: '{output_filepath}'")
        except Exception as e:
            print(f"Fehler beim Schreiben der Datei '{output_filepath}': {e}")

    def get_mapping(self):
        """Gibt das aktuelle Mapping von Originaltext zu Platzhalter zurück."""
        return self.anonymization_map

    def reset(self):
        """Setzt das Mapping und die Zähler zurück für eine neue Anonymisierungsrunde."""
        self.anonymization_map = {}
        self.counters = defaultdict(int)
        print("Anonymizer wurde zurückgesetzt.")


# --- Beispielanwendung ---
if __name__ == "__main__":
    # --- Konfiguration ---
    LLM_API_KEY = os.getenv("API_KEY", "OLLAMA")

    LLM_ENDPOINT = os.getenv("BASE_URL", "http://localhost:11434/v1/chat/completions")

    LLM_MODEL = os.getenv(
        "MODEL_NAME", "hf.co/google/gemma-3-4b-it-qat-q4_0-gguf:latest"
    )

    # LLM usage True/False
    USE_LLM_ENHANCEMENT = True

    # Initialisiere den Anonymisierer
    anonymizer = Anonymizer(
        labels_to_anonymize=["PER", "LOC", "GPE", "ORG"],  # spaCy Labels
        use_llm=USE_LLM_ENHANCEMENT,
        llm_endpoint_url=LLM_ENDPOINT,
        llm_api_key=LLM_API_KEY,
        llm_model_name=LLM_MODEL,  # Passe dies ggf. an deinen Endpunkt/Modell an
    )

    # Beispieltexte als Batch
    texts_to_process = [
        "Max Mustermann wohnt in Berlin. Seine E-Mail ist max.mustermann@email.com.",
        "Erika Mustermann arbeitet bei der Beispiel GmbH in München. Ihr Projekt heißt intern 'Projekt Sonne'.",
        "Kontaktieren Sie Klaus Müller (Tel: 0171-123456) oder Max Mustermann wegen der Lieferung nach Hamburg.",
        "Die Kundennummer lautet KDB-789-C. Das Meeting findet in Berlin statt.",
    ]

    print("--- Starte Batch-Anonymisierung ---")
    anonymized_results = anonymizer.anonymize_batch(texts_to_process)

    print("\n--- Ergebnisse ---")
    if anonymized_results:
        for i, (original, anonymized) in enumerate(
            zip(texts_to_process, anonymized_results)
        ):
            print(f"Dokument {i + 1} Original:\n{original}")
            print(f"Dokument {i + 1} Anonymisiert:\n{anonymized}\n" + "-" * 15)
    else:
        print("Anonymisierung fehlgeschlagen.")

    print("\n--- Finales Mapping ---")
    final_mapping = anonymizer.get_mapping()
    # Schönere Ausgabe des Mappings
    for original, placeholder in final_mapping.items():
        print(f"'{original}' -> {placeholder}")
    print("-" * 20)

    # Beispiel für Datei-Anonymisierung (verwende das gleiche Mapping weiter)
    input_filename = "original_daten_batch.txt"
    output_filename = "anonymisierte_daten/anonym_batch.txt"

    try:
        with open(input_filename, "w", encoding="utf-8") as f:
            f.write("Anfrage von Sabine Meier (s.meier@beispiel.org).\n")
            f.write("Projektnummer: PROJ-XYZ-123 in Köln.\n")
            f.write(
                "Bezug auf Max Mustermann und KDB-789-C."
            )  # Testet Konsistenz aus vorherigem Batch

        print(f"\n'{input_filename}' für Datei-Test erstellt.")
        anonymizer.anonymize_file(input_filename, output_filename)
        # os.remove(input_filename) # Optional cleanup

    except Exception as e:
        print(f"Fehler im Datei-Beispiel: {e}")
