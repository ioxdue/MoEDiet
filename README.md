# MoEDiet

Da una lista di tag a un checkpoint MoE potato, con selezione e audit separati. Gli adattatori attuali supportano OLMoE, Qwen2-MoE e Qwen3-MoE con Transformers 4.55.4. L'esportazione **GGUF quantizzata** resta disponibile per il profilo OLMoE verificato. Il percorso HF di Qwen3.8-Flash-Next (`qwen4_exp`) viene ancora bloccato prima del Job. Per i suoi **GGUF già quantizzati** è disponibile il comando locale separato `gguf_prune.py`, descritto sotto: compatta una selezione esplicita, senza profilazione o validazione della qualità.

La guida [ARCHITECTURES.md](ARCHITECTURES.md) descrive i profili JSON personalizzati, lo schema v2, i budget percentuali e i limiti. L'esempio [config.hf.v2.json](config.hf.v2.json) mantiene OLMoE e usa lambda=0. Le configurazioni v1 esistenti restano compatibili. L'originale predefinito e `allenai/OLMoE-1B-7B-0125-Instruct` in formato Hugging Face/safetensors.

Per inventariare gli header di un checkpoint safetensors senza caricare i pesi,
usare `python .\run.py --config .\config.json --inspect-safetensors`. Supporta
directory `mode=local` e repository Hub `mode=hf`; sul Hub risolve la revisione a
un commit e usa richieste HTTP Range, senza fallback al download completo. Controlla
indice, shard, forme, dtype e offset, ma non verifica il contratto di un adapter,
non legge i payload e non abilita `qwen4_exp`. È un inventario, non un preflight
di supporto o di fattibilità hardware.

## Pruning diretto GGUF Qwen3.8

`gguf_prune.py` legge tutti gli shard locali e copia i blocchi quantizzati degli
esperti conservati insieme alle righe dei router. Mantiene top-k, PLE, shared expert,
attenzione, SSM, hyper-connections e tokenizer. Non crea un intermedio F16/BF16.
Il supporto riguarda l'architettura GGUF `qwen4exp`, con matrici gate/up/down separate
e router F32. I tipi degli esperti verificati sono Q5_0, IQ3_XXS, IQ4_XS, IQ4_NL,
Q2_0 e IQ2_S; layout fusi e routing a gruppi sono rifiutati. Metadati array vuoti
o annidati e GGUF big-endian sono esclusi da questo primo incremento.

La CLI supporta anche `inspect-hub` e `compact-hub` per GGUF pubblici su una
revisione Hub fissata. Usa HTTP Range, `If-Match` e un parser header-only con un
limite massimo di directory remota; non effettua fallback al download completo.
La compattazione remota trasferisce comunque tutti i byte dei tensori mantenuti.

### Ambiente e ispezione

Il workspace locale dispone dell'ambiente `.venv-gguf` già preparato. Per ricrearlo
su un'altra macchina con Python 3.10+ e Git:

```powershell
python -m venv .venv-gguf
.\.venv-gguf\Scripts\python.exe -m pip install -r requirements-gguf.txt
```

La dipendenza è fissata al commit `19cce6919992fdffd92e0ac152caaf1e2b8f254e`
di llama.cpp. Il solo numero PyPI `gguf==0.19.0` non identifica quella revisione:
la versione precedentemente installata non include Q2_0. Questo comando non
richiede PyTorch, Transformers o la compilazione di llama.cpp.

Scaricare separatamente tutti i file della variante desiderata nella stessa
directory. `--model` indica il primo shard, compreso il caso in cui contenga
solo tokenizer/metadati. Sostituire il percorso nell'esempio:

```powershell
$modelGguf = 'D:\modelli\Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf'
.\.venv-gguf\Scripts\python.exe .\gguf_prune.py inspect --model $modelGguf --budgets 256 192 128
```

`inspect` legge le intestazioni e controlla shard, forme, tipi, offset e routing.
Stampa JSON su stdout; messaggi di avanzamento vanno su stderr. Senza `--hash`
non legge i payload per calcolare SHA-256 e non dichiara verificati gli hash.
Le stime usano i byte di ogni tensore, conservando l'overhead sorgente come
approssimazione. `weights_excluding_ple_bytes` conta i payload senza PLE e senza
overhead: **non è la RAM totale di inferenza**.

### Selezione e compattazione

Per creare un manifesto valido mantenendo inizialmente tutti gli esperti:

```powershell
.\.venv-gguf\Scripts\python.exe .\gguf_prune.py inspect --model $modelGguf --identity-selection .\selection.qwen38.json
```

Questa opzione implica `--hash`: legge tutti gli shard, anche quelli molto grandi.
Il JSON contiene versione, architettura, nomi/dimensioni/SHA-256 sorgenti,
`layers` e provenienza. Ogni elemento di `layers` ha `layer` ed `expert_ids`:
gli ID originali sono ordinati, unici e di uguale numerosità in tutti i layer.
Per il modello completo analizzato ci sono 48 layer, 512 esperti e top-k 10.

Per una selezione manuale, modificare le liste e impostare
`provenance.method` a `manual`; mantenere almeno top-k esperti per layer.
Una traccia esterna può dichiarare `routing_trace` con `trace_sha256`, ma questo
comando non ne verifica la qualità né la correttezza della profilazione.
Il template identità non sceglie gli esperti utili alla programmazione.
Le vecchie maschere HF/OLMoE non sono manifesti GGUF compatibili.

### Sorgente pubblica Hugging Face Hub

Per inventariare un GGUF pubblico senza scaricare i payload, fissare un commit
completo e indicare la directory che contiene un singolo GGUF o tutti gli shard:

```powershell
$repo = "owner/repo"
$revision = "<commit SHA completo>"
$path = "path/to/gguf"
.\.venv-gguf\Scripts\python.exe .\gguf_prune.py inspect-hub --repo-id $repo --revision $revision --path $path --identity-selection .\selection.qwen38.json
```

Il manifesto remoto identifica ogni file con percorso, dimensione, OID API e
ETag; non contiene né dichiara verificato uno SHA-256 integrale sorgente.
Letture remote e copie usano HTTP Range, `If-Match` e controlli rigidi di
risposta. Il parser header-only segue GGUF v3 little-endian e i tipi/dimensioni
della revisione `gguf-py` fissata sopra: decodifica KV e tensor directory con
range incrementali esatti, senza prefetch oltre i campi e senza leggere payload.
Per shard il budget predefinito è 64 MiB di byte header effettivamente richiesti
e 1.000.000 di elementi complessivi (KV, tensori ed elementi array), con un
massimo fisso di 20.000 richieste Range header; i budget byte/elementi si
possono abbassare con `--max-header-bytes` e `--max-directory-items`.
Conteggi/lunghezze oltre budget, array vuoti/annidati, versioni non v3,
big-endian e directory malformate vengono rifiutati senza fallback a download
completo. La compattazione legge comunque tutti i byte dei tensori mantenuti:
una selezione identità trasferisce quindi il contenuto sostanziale del modello.

```powershell
.\.venv-gguf\Scripts\python.exe .\gguf_prune.py compact-hub --repo-id $repo --revision $revision --path $path --selection .\selection.qwen38.json --out .\runs\qwen38-gguf
```

`compact-hub` crea e verifica localmente gli shard ridotti; non pubblica né
ricarica output. Il report distingue l'identità della revisione/file Hub dalla
verifica hash integrale sorgente (`source_hashes_verified=false`); hash dei
payload trasferiti e dei file finali sono verificati localmente.

```powershell
.\.venv-gguf\Scripts\python.exe .\gguf_prune.py compact --model $modelGguf --selection .\selection.qwen38.json --out .\runs\qwen38-gguf
```

L'output è una nuova directory con `model-00001-of-00004.gguf` e gli altri shard
(oppure `model.gguf` per ingresso singolo), `selection.json` e `report.json`.
Il numero degli shard viene conservato. `report.json` riporta gli hash sorgenti
e finali, la mappa `expert_ids[new_id] = old_id`, i byte effettivi, gli hash dei
payload e del codice usato, il buffer di copia e il picco residente del processo.
Il buffer è di 4 MiB; vanno aggiunti metadati/librerie e la page cache del sistema.

La directory viene promossa solo dopo aver riletto e verificato tutti i payload
finali. Input e destinazioni esistenti non vengono sovrascritti; errori e
interruzioni gestite rimuovono la directory temporanea. Occorre spazio per
sorgente più risultato: circa 146,24 GiB per sorgente UD-IQ4_XS e candidato
256, o 174,48 GiB per la copia identità, oltre a report e margine. Un arresto
forzato del processo può lasciare una directory `.gguf-prune-*` non promossa.

Lo stato finale è **`compacted_unvalidated`**: `structural_validation.passed=true`
certifica la trasformazione dei file. `runtime_smoke` e `quality_validation`
restano `not_run`. Profilazione sul corpus agentico, audit delle distribuzioni,
inferenza reale e prestazioni Snapdragon sono il passo successivo del
[piano](PLAN-qwen38-gguf-pruning.md). Non avviare `run.py` aspettandosi questo
percorso GGUF: il launcher HF mantiene le proprie precondizioni.

### Verifiche locali

```powershell
.\.venv-gguf\Scripts\python.exe -m unittest -v test_gguf_prune
```

16 test passati su Windows ARM64: file GGUF reali sintetici con i sei formati,
shard iniziale senza tensori, selezione identità/ridotta, ID non contigui e
diversi fra layer, router, metadati e payload preservati, allineamento, input
errati, corruzione dell'output, spazio insufficiente e interruzione.
Le intestazioni del checkpoint reale alla revisione
`d96201194f103c906994f283f504209dc8d2a10c` sono state confrontate con il contratto;
non sono stati scaricati o eseguiti i 87,24 GiB di pesi.
La suite storica ha inoltre completato 20 test, con un test nativo OLMoE saltato
per il tokenizer assente. Il controllo `pip check` dell'ambiente che eredita i
pacchetti del sistema segnala un'incongruenza preesistente tokenizers/Transformers;
non ha impedito questi test e il nuovo comando non importa quelle librerie.

## Scegli dove eseguire

Verifica dell'aggiornamento ad adattatori: **20 test passati su Windows ARM64/CPU**
(PyTorch 2.15.0, Transformers 4.55.4); un test nativo GGUF saltato per assenza del
tokenizer di prova. Sono inclusi checkpoint casuali reali OLMoE, Qwen2-MoE e
Qwen3-MoE, compatibilita delle maschere v1, controlli del launcher e round trip
di salvataggio. Nessun nuovo Job HF o test del Qwen3.8 completo e stato eseguito.

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

### Esperimento HF conservativo con sonde ampliate

Il profilo [config.hf.conservative.json](config.hf.conservative.json) prova tagli più piccoli
del profilo iniziale, senza allentare la soglia di fedeltà:

| Parametro | Valore |
| --- | --- |
| Esperti conservati per strato | 48, 56, 60, 62; ripiego automatico a 64 |
| Frazione di esperti rimossi | 25%, 12,5%, 6,25%, 3,125% rispettivamente |
| Soglia KL | 0,10 nat/token, separatamente per ogni gruppo |
| Penalità del controllo `lambda` | 0,5 |
| Continuazione di riferimento | Fino a 32 token per sonda; EOS può terminarla prima |
| Corpus | `corpora/agents-v1.json` |
| Esecuzione | Una L4, BF16, timeout 90 minuti, export Q4_K_M |
| Destinazione predefinita | `operator/moediet-runs`, obbligatoriamente privata |

Il modello e il commit AllenAI sono fissati alla stessa revisione verificata
`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`.
Le percentuali riguardano gli **esperti**, non il totale dei parametri o la dimensione del GGUF.

Il corpus contiene **12 sonde per gruppo e per split**, per 144 prompt complessivi:
Python, C++, DevOps e controllo, ciascuno in profilazione, selezione e audit.
Ogni gruppo di controllo comprende quattro casi di giardinaggio, quattro di navigazione
e quattro di musica. Le istruzioni finali sono bilanciate fra gruppi e split.
I casi includono gestione delle risorse, errori, concorrenza, ownership, deployment e diagnosi.
Non ci sono prompt duplicati secondo la normalizzazione del validatore.
I temi possono ricorrere fra split, ma le situazioni e i testi differiscono.

Sono sonde sintetiche in inglese, scritte per questo esperimento: **non un benchmark
indipendente di correttezza o dati reali dell'utente**. Il nuovo corpus e la maggiore
lunghezza delle continuazioni cambiano anche la misura: un confronto con il primo Job
non isola l'effetto dei soli budget. Per isolare quell'effetto serve un esperimento
separato mantenendo corpus e lunghezza originali.

La selezione e l'audit possono valutare fino a **384 posizioni generate per gruppo**
(12 × 32), contro le 16 per dominio del profilo iniziale; il conteggio effettivo dipende da EOS.
Il numero reale di token di profilazione si legge in `report.json`, non si deduce
dal solo numero di sonde. La pipeline si ferma al primo budget accettato in selezione;
l'audit non viene riutilizzato per scegliere un altro budget. Anche questo esperimento
può terminare senza checkpoint e senza GGUF.

Da PowerShell, nella radice del repository, usa l'ambiente separato del launcher:

```powershell
# Solo se l'ambiente non esiste ancora:
python -m venv .venv-launcher
.\.venv-launcher\Scripts\python.exe -m pip install -r requirements-launcher.txt
.\.venv-launcher\Scripts\python.exe -m pip check

# Verifica offline: nessun Job viene avviato.
.\.venv-launcher\Scripts\python.exe .\run.py --config .\config.hf.conservative.json --dry-run

# Avvio reale a pagamento:
.\.venv-launcher\Scripts\python.exe .\run.py --config .\config.hf.conservative.json
```

Le credenziali sono quelle già salvate da `hf auth login`.
Per un altro account aggiungi `--repo nome/repo-privato`.
Il corpus viene copiato automaticamente nella richiesta inviata al Job; non occorre
caricarlo a mano. Mantieni insieme profilo e sottocartella `corpora`.
Non cambiare `--tags` senza fornire un corpus con gli stessi gruppi.
Il carico è maggiore del primo esempio: i 90 minuti sono un limite di durata,
non una garanzia di completamento né un tetto monetario.

Dopo il Job verifica `status`, `selection`, `audit`, `checkpoint_audit`,
`profile_tokens_per_layer_by_group` e i conteggi `tokens` nel report.
Per un risultato `pruned`, controlla anche `conversion.json` e il GGUF:
la KL del checkpoint non misura il degrado aggiunto dalla quantizzazione.

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
