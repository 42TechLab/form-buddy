# FORM_BUDDY

Assistente per la compilazione di moduli sul campo: dettatura in linguaggio informale, lettura di foto e documenti, controllo di dati mancanti e incoerenze. Funziona sullo smartphone, offline, con un modello AI che gira sul dispositivo (Gemma 3n): nessun dato viene inviato a servizi esterni.

## Fase 0 — test di fattibilità

Pagina: https://42techlab.github.io/form-buddy/fase0/

Misura sul telefono:
- tempi di caricamento e di risposta di Gemma 3n E2B (WebGPU, MediaPipe LLM Inference);
- estrazione di campi da 20 frasi dettate (dati di fantasia);
- estrazione di campi da foto di targhette e documenti;
- trascrizione ed estrazione da registrazioni vocali.

Il modello non è incluso: va scaricato da Hugging Face (`google/gemma-3n-E2B-it-litert-lm`, file `gemma-3n-E2B-it-int4-Web.litertlm`) dopo aver accettato la licenza Gemma, e selezionato nella pagina. Viene conservato nella memoria privata del browser.

I valori attesi per le foto reali sono pubblicati solo come hash SHA-256: la pagina può verificare le risposte senza esporre i dati.
