# Profili di architettura e configurazioni v2

MoEDiet separa la selezione e l'audit dalle operazioni sui moduli MoE. Il JSON
sceglie un'implementazione gia presente e ne descrive i percorsi. Non contiene
codice Python, import dinamici o espressioni da eseguire.

## Supporto attuale

| `model_type` | Profilazione, maschera, checkpoint HF | GGUF |
| --- | --- | --- |
| `olmoe` | Implementati; test su fixture Transformers reale | Esportatore esistente, limitato al profilo AllenAI 0125 |
| `qwen2_moe` | Implementati; test con layer densi e shared expert | Non abilitato |
| `qwen3_moe` | Implementati; test con layer densi | Non abilitato |
| `qwen4_exp` (Qwen3.8-Flash-Next) | Non implementato; errore prima del Job | Non abilitato |

Gli adattatori eseguibili sono fissati a **Transformers 4.55.4**. I test usano
pesi casuali piccoli, non misurano la qualita dei checkpoint preaddestrati.
Qwen3-MoE e Qwen3.8-Flash-Next sono architetture diverse: il primo profilo non
abilita il secondo. `architectures/qwen4_exp.json` e un descrittore di mancato
supporto, non un adattatore funzionante.

Qwen3.8 richiede la verifica di caricamento, router, esperti fusi, esperto condiviso,
configurazione testuale annidata, percorsi del modello multimodale e salvataggio.
Il runtime installato non contiene questa architettura e qui non e stato eseguito.
Non e sufficiente cambiare il nome della classe o aggiornare Transformers senza
ripetere i test del layout. Il precedente profilo L4/BF16 non e adeguato alle sue dimensioni.

## Configurazione dell'esperimento

`config.hf.v2.json` mostra il nuovo schema su OLMoE; usa lambda=0, controllo
obbligatorio, le sonde v1 e gli stessi tagli relativi 48/56/60/62 su 64.
`config.local.qwen3-moe.json` e un esempio per un checkpoint **qwen3_moe** gia
presente in `models/qwen3-moe`: occorre indicare la propria directory e dimensionare
RAM/VRAM prima dell'esecuzione. Non scarica un modello come parte del dry run.

```powershell
# Solo sintassi, percorsi e profilo dichiarato; nessuna rete o spesa.
python .\run.py --config .\config.hf.v2.json --dry-run

# Legge config.json e risolve la revisione Hub; non scarica pesi e non avvia Job.
python .\run.py --config .\config.hf.v2.json --preflight

# Avvio reale: in mode=hf questo comando avvia un Job a pagamento.
python .\run.py --config .\config.hf.v2.json
```

Il preflight verifica tipo del modello, profilo, routing, budget, quantizzazione
in ingresso ed esportazione richiesta. Non e una misura della memoria disponibile
o della qualita. Viene eseguito anche prima dell'upload e della creazione del Job.
Se il profilo e esplicitamente non supportato, anche il dry run termina con errore.
Per un ID Hub con `architecture_profile` omesso/`auto`, il dry run non deduce
l'architettura dal nome: serve il preflight online.

I profili v1 esistenti restano validi. Nel v2:

- `model` contiene `id` e opzionalmente `revision`; una ref Hub viene fissata a commit.
- `architecture_profile` e un nome incorporato o un percorso relativo al JSON.
  Se omesso, si usa il profilo incorporato per il `model_type` effettivo.
- `pruning.budgets` e `pruning.keep_fractions` si escludono. Le frazioni sono
  arrotondate per eccesso; valori che scendono sotto il top-k vengono rifiutati.
  Il ripiego al modello completo resta automatico. I budget sono uniformi fra
  i layer MoE; i layer densi vengono lasciati intatti.
- `preserve_top_k=true`, `shared_experts="preserve"` e `require_control=true`
  sono vincoli di questa versione, non opzioni per rimuovere le verifiche.
- `validation` contiene `max_kl` e `eval_tokens`; `build` conserva `lambda`,
  `device` e `dtype`. Parametri duplicati vengono rifiutati.
- `export.format="hf"` salva il checkpoint senza conversione. `gguf` richiede
  un esportatore disponibile e una quantizzazione Q4 supportata.

Le frazioni si riferiscono agli esperti instradati, non ai parametri totali.
Il 75% diventa 48 su 64 o 384 su 512, senza implicare lo stesso impatto sulla qualita.
L'esecuzione supporta CPU oppure una singola GPU CUDA; questo aggiornamento non
introduce offload, distribuzione su piu GPU, pruning di pesi quantizzati o input visivi.

## Creare un profilo personalizzato

Partire da `architectures/olmoe.json` o `architectures/qwen2_moe.json`:

- `model_type`: deve corrispondere al `config.json` sorgente.
- `implementation`: attualmente `softmax_topk_modulelist_v1`.
- `transformers_version`: versione esatta del layout verificato.
- `layout`: percorsi di backbone e lista dei layer sul modello; percorsi relativi
  al layer per il blocco MoE e relativi al blocco per router, esperti e attributi.
- `config`: percorsi dei campi relativi alla configurazione del modello.
- `layer_selection`: `all` oppure `qwen_sparse_step`, che usa `mlp_only_layers`
  e `decoder_sparse_step` per ricostruire gli indici dei layer MoE.
- `shared_experts`: lista di moduli con relativo gate sigmoid da preservare.

L'implementazione richiede router `Linear` senza bias, esperti `ModuleList`, softmax
in float32, top-k e normalizzazione opzionale. Tensori di esperti fusi, routing a
gruppi e altre formule richiedono una nuova implementazione Python. Un profilo non
puo inventare un'operazione o dichiarare da solo la compatibilita GGUF.

I campi sconosciuti, percorsi non dichiarativi e implementazioni mancanti vengono
rifiutati. Il caricamento usa `AutoModelForCausalLM` senza `trust_remote_code`.
Per estendere l'implementazione modificare la selezione in `adapters.adapter_for`
e il validatore di `architecture.py`, e aggiungere test sul modello reale piccolo.

## Verifiche e riproducibilita

Prima della profilazione, su un breve prefisso il worker ricostruisce le uscite
dei blocchi MoE secondo il profilo e le confronta con quelle native. Questo rileva,
per esempio, uno shared expert omesso, ma non dimostra compatibilita su ogni input.
Verifica inoltre la maschera identita. Se il taglio passa selezione e audit,
compatta, salva e ricarica il modello, quindi richiede sia l'audit contro l'originale
sia l'equivalenza numerica rispetto alla maschera sugli stessi prefissi di audit.

La tolleranza di equivalenza e indipendente dalla soglia KL: float32 `(atol=1e-4,
rtol=1e-5)`, float16 `(0.02, 0.001)`, bfloat16 `(0.125, 0.001)`. Si confrontano
log-probabilita; le tolleranze dichiarano equivalenza approssimata, non bit a bit.
Se fallisce, viene conservata la maschera e scartato il checkpoint temporaneo.

Report e maschera v2 incorporano il profilo completo e il suo SHA-256; il checkpoint
incorpora il profilo in `config.json`. Gli indici dei layer MoE sono espliciti.
Le maschere OLMoE v1 restano leggibili. Per HF, profili incorporati, implementazione
e profilo personalizzato risolto viaggiano nella richiesta fissata a commit:
il worker non dipende dai percorsi del PC che invia il Job.

```powershell
python -m unittest -v
```

La suite include OLMoE, Qwen2-MoE e Qwen3-MoE piccoli, normalizzazione, layer densi,
shared expert, rimozione degli hook anche dopo eccezioni, confronto delle distribuzioni,
salvataggio/ricaricamento, generazione con cache, blocco del checkpoint non equivalente,
validazione v1/v2 e trasporto HF simulato. La conversione nativa GGUF resta opzionale
e richiede `MOEDIET_TEST_TOKENIZER` e toolchain. Nessun Job o benchmark del Qwen3.8
completo e stato eseguito per questo aggiornamento.
