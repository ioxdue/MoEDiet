# MoEDiet MVP

Da una lista di tag a un checkpoint OLMoE potato e a un **GGUF quantizzato** per llama.cpp. I profili `config.local.json` e `config.hf.json` eseguono la stessa profilazione, selezione, verifica del checkpoint e conversione. Lo ZIP contiene codice e configurazioni. L’originale predefinito è `allenai/OLMoE-1B-7B-0125-Instruct` in formato Hugging Face/safetensors.

## Scegli dove eseguire

| | Locale | Hugging Face Jobs |
| --- | --- | --- |
| Profilo | `config.local.json` | `config.hf.json` |
| Originale | ID Hub scaricato nella cache sul PC oppure una cartella locale contenente `config.json`, tokenizer, pesi e indice degli shard | ID Hub. I pesi arrivano nella macchina del Job. Il PC invia i sorgenti del controller, i parametri e l’eventuale corpus specificato. |
| Calcolo | PyTorch CPU o CUDA. Sullo Snapdragon il controller usa la CPU Oryon. | Default GPU L4, BF16 e timeout di 90 minuti. La quantizzazione e la prova llama.cpp usano la CPU del Job. |
| Risultati | Sotto la nuova directory `out`, per default `runs/dev-q4` accanto al profilo | Nel repository privato scelto, sotto `runs/<JOB_ID>/`. Il launcher stampa i link al Job e ai risultati. |

### Locale su Windows

Apri PowerShell nella cartella estratta che contiene `run.py`. Usa un ambiente Python 3.10 aggiornato o successivo, con una build PyTorch compatibile. Per la prima compilazione automatica di llama.cpp servono CMake sul PATH e un compilatore C++ (su Windows ARM64, Visual Studio 2022 con strumenti MSVC ARM64).

```powershell
python -m pip install -r requirements-q4.txt
python .\run.py --config .\config.local.json --tags python "c++" devops
```

Eseguito nella cartella del progetto, il GGUF prodotto con il profilo predefinito è **`.\runs\dev-q4\gguf\Q4_K_M\model-Q4_K_M.gguf`** (o `$PWD/runs/dev-q4/gguf/Q4_K_M/model-Q4_K_M.gguf`). Per una nuova build scegli un’altra `out`. I percorsi relativi nel JSON sono risolti rispetto alla cartella del JSON e supportano variabili d’ambiente (es. `%USERPROFILE%`, `$HOME`).

Per controllare dove viene messo l’originale puoi scaricarlo una volta in una directory esplicita e impostare nel profilo, ad esempio, `"model": "%USERPROFILE%/Modelli/OLMoE-originale"` (o `"$HOME/Modelli/OLMoE-originale"`).

```powershell
hf download allenai/OLMoE-1B-7B-0125-Instruct --local-dir "$env:USERPROFILE\Modelli\OLMoE-originale"
```

Con l’ID Hub del profilo di esempio il download è automatico. Su Windows la cache HF predefinita è `%USERPROFILE%\.cache\huggingface\hub`. L’input della potatura è sempre il modello HF completo.

### Comandi per Hugging Face

Sul PC che invia il Job serve solo il launcher. Si può usare un ambiente Python senza PyTorch. Sostituisci `tuo-utente/moediet-runs` con il tuo repository di destinazione. Il token deve poter avviare Jobs, leggere l’eventuale sorgente privata e scrivere nella destinazione. Se chiedi la creazione del repository serve anche il permesso di creazione. HF Jobs usa credito di calcolo dell’account.

```powershell
python -m pip install -r requirements-launcher.txt
hf auth login
python .\run.py --config .\config.hf.json --repo tuo-utente/moediet-runs --dry-run
python .\run.py --config .\config.hf.json --repo tuo-utente/moediet-runs --tags python "c++" devops
```

`--dry-run` controlla il profilo offline. L’ultimo comando avvia il Job a pagamento. Puoi anche inserire l’ID in `output_repo` nel profilo HF. Le credenziali vengono lette da `hf auth login` o `HF_TOKEN` e passate al Job come secret. Una destinazione pubblica viene rifiutata. Il launcher fissa il commit dell’originale e invia una copia dei soli file necessari sotto `requests/<id>/`. Il Job legge il codice da quel commit. Un corpus personalizzato viene incluso nella richiesta e i testi delle sonde compaiono in `report.json`.

Il Job salva i due JSON e l’eventuale checkpoint sul Hub **prima** della conversione. Salva il GGUF in un secondo commit dopo la verifica nativa. Anche un rilancio con un nuovo Job ID ha una directory di risultati separata. Per seguire il Job o scaricare il solo GGUF, sostituisci `JOB_ID` con l’ID stampato.

```powershell
hf jobs logs JOB_ID
hf download tuo-utente/moediet-runs "runs/JOB_ID/gguf/Q4_K_M/model-Q4_K_M.gguf" --local-dir .\download
```

Nel repo HF il checkpoint PyTorch è una sottocartella. Per caricarlo da lì si può usare `AutoModelForCausalLM.from_pretrained("tuo-utente/moediet-runs", subfolder="runs/JOB_ID/checkpoint")` e lo stesso `subfolder` con `AutoTokenizer`. [HF Jobs](https://huggingface.co/docs/hub/jobs-overview) · [Tariffe](https://huggingface.co/docs/hub/jobs-pricing)

## Conversione Q4 e inferenza

La chiave `quantization` accetta `"Q4_K_M"` (predefinito), `"Q4_0"` oppure `null` per fermarsi al checkpoint HF. Il convertitore legge solo `checkpoint/` di una build con `status=pruned` e audit accettato. Produce un GGUF F16 temporaneo, poi `llama-quantize` applica la quantizzazione scelta. `Q4_K_M` usa una combinazione di tipi. Anche con `Q4_0` alcuni tensori, ad esempio le normalizzazioni, restano in virgola mobile.

| File sotto la cartella dell’esecuzione | Utilizzo |
| --- | --- |
| `mask.json`, `report.json` | Sorgente, esperti selezionati ed esito della potatura. In caso di ripiego all’originale si ottengono solo questi due file. |
| `checkpoint/` | Modello potato completo con tokenizer, ricaricabile con Transformers. |
| `gguf/Q4_K_M/model-Q4_K_M.gguf` | Modello per llama.cpp. Con `Q4_0` sia il nome della directory sia quello del file usano `Q4_0`. |
| `gguf/Q4_K_M/conversion.json` | Byte, SHA-256, tipi reali delle matrici degli esperti, topologia e prova di esecuzione CPU. |

Prima di pubblicare il file, MoEDiet verifica architettura, strati, numero di esperti conservati e attivi, shape delle matrici e presenza di pesi degli esperti effettivamente Q4. Quindi `llama-cli` controlla i tensori ed esegue prefill e decode. L’F16 temporaneo viene eliminato. Occorre spazio durante la conversione per checkpoint, intermedio F16 e Q4. Se la conversione fallisce, il checkpoint HF già accettato resta recuperabile. La KL in `report.json` riguarda il modello HF prima della quantizzazione. La prova nativa del GGUF misura la caricabilità e l’esecuzione su CPU. Qualità sui propri task, prestazioni su Adreno e compatibilità HTP richiedono prove dedicate.

Per convertire un risultato già esistente, anche di una versione precedente, senza ripetere la profilazione, usa `gguf_export.py`. Ogni variante ha una propria cartella. Un export esistente viene protetto dalla sovrascrittura.

```powershell
python .\gguf_export.py --run .\runs\dev-q4 --qtype Q4_0
llama-cli.exe -m .\runs\dev-q4\gguf\Q4_0\model-Q4_0.gguf -p "Spiega una closure Python." -n 64
```

Per il secondo comando usa il tuo `llama-cli.exe` sul PATH o il suo percorso completo. Il solo file `.gguf` basta per l’inferenza con un runtime compatibile.

**Toolchain.** Con `llama_dir` e `bin_dir` a `null`, al primo export il programma scarica il sorgente verificato di llama.cpp b6500, controlla lo SHA-256 e compila `llama-quantize` e `llama-cli` per CPU. La cache è `~/.cache/moediet`, modificabile con `MOEDIET_TOOLS_CACHE`. L’estrazione sicura richiede una versione Python con `tarfile.data_filter` (inclusa in Python 3.12). Per riusare una build esistente imposta nel profilo locale `llama_dir` alla directory con `convert_hf_to_gguf.py` e `gguf-py/`, e `bin_dir` a quella con i due eseguibili della stessa revisione. Anche l’export separato accetta `--llama-dir` e `--bin-dir`. I controlli sono stati provati con il commit `a7a98e0fffed794396b3fbad4dcdbbc184963645`. [llama.cpp b6500](https://github.com/ggml-org/llama.cpp/tree/b6500)

**Parametri di potatura.** Entrambi i profili accettano `tags`, `model`, `quantization`, un percorso opzionale `corpus` e un oggetto `build` con `budgets`, `lambda`, `max_kl`, `eval_tokens`, `dtype`, `device` e `revision`. Per esempio, `"build": {"budgets": [8, 16, 32], "max_kl": 0.10, "eval_tokens": 8}`. `--tags` sostituisce i tag nel JSON. In HF si possono cambiare `flavor` e `timeout`. I profili si possono copiare e rinominare.

### CLI locale compatibile

```powershell
python .\moediet.py build --tags python "c++" devops --out runs/dev
python .\moediet.py generate --mask runs/dev/mask.json --prompt "Explain Python closures."
python .\moediet.py generate --model runs/dev/checkpoint --prompt "Explain Python closures."
```

`build` conserva la semantica della versione precedente. Aggiungendo `--gguf Q4_K_M` converte anche il checkpoint accettato. `generate --mask` usa l’originale registrato nella maschera. `generate --model` carica il checkpoint HF. Per il GGUF si usa llama.cpp.

## Come viene scelta la dimensione

1. **Sonde.** Per ciascun tag il controller prepara istruzioni distinte per profilazione, selezione e verifica finale. Python, C++, DevOps, Docker, PostgreSQL e SQL hanno casi concreti. Per altri tag sono disponibili template generici. Il controllo usa lo stesso stile con argomenti esterni. Le sonde integrate sono in inglese.
2. **Profilazione.** Per ogni token del prompt e per ogni strato, il controller ricostruisce i pesi del top-k nativo dai logit del router, in FP32, prima del cast al tipo del modello. Gli esperti esclusi dal top-k contribuiscono zero. La normalizzazione segue `norm_topk_prob`. Una sonda alla volta elimina il padding. La massa accumulata viene divisa per il numero di token del gruppo in ciascuno strato. Si mediano poi i tag a pari peso e si calcola `score = media_tag − lambda × media_controllo`. `--lambda` vale 0,5 per default. Lo score misura l’uso osservato. Alcuni esperti condivisi tra dominio e controllo possono essere indispensabili.
3. **Selezione e audit.** Per ogni budget si mantengono gli esperti con score maggiore, separatamente per strato. A parità di score prevale l’ID originale più basso. Il modello intero genera continuazioni greedy brevi sui prompt di selezione. Il candidato ripercorre **gli stessi prefissi** e viene confrontato con l’intero tramite `KL(P_intero || P_candidato)`, mediata sulle posizioni generate per ciascun tag e per il controllo. Il primo budget crescente che supera la verifica in tutti i gruppi passa a un audit su sonde nuove. Se l’audit fallisce, si restituisce la maschera completa. L’audit non viene riutilizzato per scegliere un altro budget.

Per il checkpoint AllenAI ci sono 16 strati, 64 esperti per strato e 8 esperti instradati per token. I budget predefiniti sono 8, 16, 32 e il ripiego a 64. Un budget di 2 viene rifiutato. Con budget 8 tutti gli otto superstiti partecipano a ogni token, con pesi che dipendono ancora dal token. `--max-kl 0.10` imposta la tolleranza iniziale in nat/token, applicata **separatamente** a ogni gruppo. `--eval-tokens 8` imposta la lunghezza massima di ogni continuazione, che può terminare prima con EOS. Per provare una griglia più fitta o cambiare la tolleranza usare, per esempio,

```powershell
python moediet.py build --tags python "c++" devops --budgets 8 12 16 24 32 48 --max-kl 0.05 --eval-tokens 16 --out runs/dev2
```

Il minimo è relativo ai budget provati, alla graduatoria scelta e alle sonde. La KL misura la fedeltà al modello originale su quei prefissi. La correttezza di un programma generato richiede una misura separata, ad esempio test eseguibili su problemi mai usati nelle tre fasi.

## Cosa contengono i risultati

| Output | Contenuto e uso |
| --- | --- |
| `mask.json` | Sorgente, topologia, tag e ID ammessi per strato. `expert_mask[l][j]` è l’ID originale assegnato all’indice compatto `j` nello strato `l`. |
| `report.json` | Sonde, token di profilazione, score, budget provati, KL, accordo top-1, scarto massimo di log-probabilità e decisione finale. Riporta i parametri totali e, se esportato, i byte del checkpoint. |
| `checkpoint/` | Solo per un taglio accettato. Contiene le MLP superstiti reindicizzate e le corrispondenti righe dei gate. Dopo il salvataggio viene ricaricato da disco e sottoposto allo stesso audit. `checkpoint_audit.vs_mask` misura anche lo scarto rispetto al percorso mascherato, utile in precisione ridotta e nei pareggi del top-k. |

La maschera applica `−∞` ai logit esclusi prima della softmax. Il checkpoint AllenAI usa `norm_topk_prob=false`. Ridurre il denominatore della softmax può cambiare i pesi dei superstiti anche quando gli ID del top-k coincidono. Con la maschera i pesi originali restano residenti. Con il checkpoint fisico, i tensori rimossi sono assenti dal file e dal successivo caricamento. Embedding, attenzione e testa linguistica contribuiscono ancora alla memoria. La percentuale di esperti rimossi e il risparmio complessivo di parametri hanno quindi denominatori diversi. Il report misura parametri e dimensione su disco. Il picco RAM/VRAM e la latenza dipendono dal runtime e dal carico.

Con `--mask-only` viene esportata solo la configurazione di routing con il relativo report. `unchanged_no_candidate` indica che nessun budget ridotto ha superato la selezione o che è stato chiesto solo il budget intero. `unchanged_audit_failed` indica che il candidato scelto ha fallito l’audit. In entrambi i casi `mask.json` contiene tutti gli esperti. `mask_only_checkpoint_audit_failed` conserva solo la maschera accettata se il checkpoint ricaricato supera la soglia di divergenza. La profilazione iniziale richiede comunque l’originale completo. I pesi del 7B occupano circa 13,84 GB prima delle altre allocazioni. L’export GGUF si configura nei profili descritti sopra.

## Sonde proprie

`--corpus casi.json` sostituisce i template. Ogni split richiede tutti i tag normalizzati e `__control__`. Una lista vuota o un testo duplicato, anche tra split diversi, interrompono l’esecuzione. Sono accettati anche prompt in italiano. Esempio valido per `--tags python`.

```json
{
  "profile": {"python": ["Diagnose a shared mutable default argument in Python."], "__control__": ["Explain how a map compass works."]},
  "select": {"python": ["Show how yield changes file iteration."], "__control__": ["Explain why a compass needs a declination correction."]},
  "audit": {"python": ["Repair a late-bound lambda in a loop."], "__control__": ["Estimate distance using two points on a scale map."]}
}
```

Le due sonde integrate per tag e per split consentono di provare il flusso. Per decidere un taglio destinato all’uso quotidiano, il corpus può includere codice, errori e richieste reali mantenendo distinti profilazione, selezione e audit.

## Verifiche eseguite

Su CPU x86-64, con PyTorch 2.8.0 e Transformers 4.55.4, i test offline costruiscono un vero `OlmoeForCausalLM` con pesi casuali e seed 719. Verificano i conteggi per token, entrambe le politiche di normalizzazione, l’assenza di token negli esperti esclusi, la rimozione degli hook, i casi di rifiuto e le due vie di inferenza. Il confronto numerico usa i logit di tutte le posizioni del prompt e anche una generazione con KV cache.

Nella prova con `python`, `c++`, `devops`, due strati e top-2 nativo, il budget 3 su 6 ha portato i parametri da **78.720 a 50.880**. I pesi `safetensors` sono passati da 320.968 a 207.624 byte. Maschera e checkpoint hanno avuto lo stesso top-1 in tutte le 48 posizioni dell’audit. Questi dati verificano il funzionamento meccanico su un modello casuale.

Per il checkpoint AllenAI è stata verificata la corrispondenza dei **3.219 nomi dei tensori** nell’indice pubblico con il modello istanziato senza allocare i pesi. È stato verificato anche il tokenizer con il suo chat template. La revisione controllata è `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`. La qualità e il budget accettabile sul 7B addestrato restano da misurare.

L’aggiornamento Q4 aggiunge test dei profili, dell’invio privato con revisioni fissate, del salvataggio separato del checkpoint e del GGUF, e della conservazione del checkpoint quando la conversione fallisce. La prova nativa usa un secondo modello casuale: hidden size e dimensione intermedia 256, due strati, 6 esperti ridotti a 3, top-2 e tokenizer AllenAI. Esegue profilazione, audit, salvataggio, ricaricamento, conversione F16, entrambe le quantizzazioni e inferenza reale in llama.cpp. La tolleranza KL di questa fixture è volutamente ampia per verificare il meccanismo. Non stima la qualità di una potatura del 7B.

La suite completa di **10 test è passata su Linux x86-64/CPU**. Sono riusciti anche download verificato e compilazione automatica della toolchain fissata. In una prova separata sono stati eseguiti bootstrap e worker HF completi, sostituendo soltanto il trasporto Hub e l’avvio del Job con operazioni locali. Profilazione, potatura e conversione in quel worker erano reali. Sono stati controllati i tre salvataggi: richiesta, checkpoint con report e GGUF verificato. Non è stato avviato un Job HF reale né provata l’immagine CUDA remota; Windows ARM64, Adreno e HTP non sono stati eseguiti in questo ambiente.

Per ripetere i test ordinari, dalla cartella del progetto:

```powershell
python -m unittest -v
```

La prova nativa è opzionale e viene saltata senza `MOEDIET_TEST_TOKENIZER`. Per includerla scarica **solo** configurazione e tokenizer ufficiali, poi esegui:

```powershell
hf download allenai/OLMoE-1B-7B-0125-Instruct --revision b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e --include "config.json" "*token*" --local-dir .\test-tokenizer
$env:MOEDIET_TEST_TOKENIZER = (Resolve-Path .\test-tokenizer).Path
python -m unittest -v
```

Questo test richiede le dipendenze Q4 e la toolchain C++; scarica i sorgenti llama.cpp se la cache è vuota. Puoi impostare anche `MOEDIET_TEST_LLAMA_DIR` per usare un checkout già compilato. La fixture genera pesi casuali temporanei: non scarica i pesi del 7B.

[Configurazione AllenAI](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/blob/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e/config.json) · [Indice dei pesi](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/blob/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e/model.safetensors.index.json) · [Implementazione OLMoE in Transformers 4.55.4](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/olmoe/modeling_olmoe.py)
