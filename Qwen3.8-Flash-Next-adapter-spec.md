# Qwen3.8-Flash-Next — specifica tecnica per l'adattatore MoEDiet

Data: 5 novembre 2026. Stato: progetto dell'adattatore, non supporto abilitato.

La parte MoE è adattabile mediante selezione sull'asse degli esperti di tre tensori. L'integrazione richiede però un nuovo adattatore Python, un runtime distinto da Transformers 4.55.4 e un contratto esplicito di caricamento/salvataggio. Un JSON per Qwen3-MoE non è sufficiente.

## 1. Evidenza e confini della verifica

Sono stati letti il `config.json` pubblico di Qwen/Qwen3.8-Flash-Next su `main`, il codice Transformers ufficiale e i sorgenti MoEDiet presenti in questa cartella. Il codice upstream è stato fissato al commit `cdb75de4adf71059c01f58dbd2d24d85084a397d`; i file di modellazione e configurazione letti su `main` coincidono con quelli a questo commit.

Il profilo dell'esperimento `config.hf.Qwen3.8-Flash-Next.json` indica invece la revisione del modello `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`. Non è stato possibile risolvere questa revisione né leggere l'indice completo e gli header safetensors. **Le forme qui riportate sono il contratto runtime dedotto dal codice e dalla configurazione pubblica; non sono una certificazione dei byte del checkpoint fissato nel profilo.** Prima dell'implementazione sul checkpoint reale occorre confrontarli con quel preciso snapshot. Un errore di accesso non dimostra che la revisione sia inesistente.

È stato eseguito solo un controllo isolato del blocco MoE, descritto al punto 10. Nessun caricamento del modello completo, Job HF, benchmark di qualità o conversione GGUF.

## 2. Configurazione di riferimento

I percorsi seguenti sono relativi al JSON multimodale originale [1].

| Campo | Valore |
| --- | --- |
| `model_type` | `qwen4_exp` |
| `architectures[0]` | `Qwen4ExpForConditionalGeneration` |
| `text_config.model_type` | `qwen4_exp_text` |
| `text_config.num_hidden_layers` | 48 |
| `text_config.hidden_size` | 2560 |
| `text_config.num_experts` | 512 |
| `text_config.num_experts_per_tok` | 10 |
| `text_config.moe_intermediate_size` | 640 |
| `text_config.shared_expert_intermediate_size` | 640 |
| `text_config.hidden_act` | `silu` |
| `text_config.max_position_embeddings` | 262144 |
| `text_config.vocab_size` | 248320 |
| `text_config.hc_count`, `hc_lowrank` | 4, 320 |
| `text_config.ple_layer_ids` | `[2]`, numerazione da 1 |
| `text_config.ple_embed_dim`, `split_ngram_parts` | 2560, 128 |
| `text_config.mtp_num_hidden_layers` | 1 |
| `text_config.dtype` | `bfloat16` |
| `tie_word_embeddings` | `false` |
| `transformers_version` | `5.8.0.dev0`, metadato di produzione del checkpoint |

`norm_topk_prob` non compare nel JSON letto: nel codice della configurazione fissato vale `True` per default [3]. Il preflight attuale legge i campi obbligatori direttamente dal JSON e quindi deve imparare a risolvere questo default, legandolo al runtime verificato. `5.8.0.dev0` non identifica da solo una build riproducibile e non è una versione minima di compatibilità dimostrata.

I 48 layer hanno tutti un blocco MoE. L'alternanza 3:1 riguarda l'attenzione: 36 Gated DeltaNet e 12 QSA agli indici da zero `3, 7, …, 47`. Il JSON chiama questi ultimi `full_attention`; la configurazione upstream li normalizza in `indexed_attention`. Non usare questo campo per escludere layer dal pruning MoE.

## 3. Percorsi dei moduli e layout runtime

Le due classi upstream producono gerarchie diverse [2]. L'adattatore deve scegliere il profilo sulla classe e sulla configurazione effettivamente caricate, preservando anche l'identità del checkpoint sorgente.

| Oggetto | Wrapper multimodale | Wrapper solo testo |
| --- | --- | --- |
| Classe | `Qwen4ExpForConditionalGeneration` | `Qwen4ExpForCausalLM` |
| Configurazione testuale | `model.config.text_config` | `model.config` |
| Backbone da chiamare per le sonde testuali | `model.model.language_model` | `model.model` |
| Lista dei layer | `model.model.language_model.layers` | `model.model.layers` |
| Blocco sul layer `i` | `layers[i].mlp` | `layers[i].mlp` |
| Visione | `model.model.visual` | non presente |
| Testa linguistica | `model.lm_head` | `model.lm_head` |

Nella tabella sopra `model` è la variabile Python dell'oggetto caricato. Nei nomi dello `state_dict` questo primo `model.` non compare. Perciò i prefissi sono `model.language_model.layers.{i}.mlp.` nel wrapper multimodale e `model.layers.{i}.mlp.` nel wrapper testuale.

Posti E=512, H=2560, I=640, K=10, i parametri relativi a ciascun blocco MoE sono:

| Nome relativo nello `state_dict` | Forma runtime | Operazione di pruning |
| --- | --- | --- |
| `gate.weight` | `[512, 2560]` = `[E,H]` | selezionare righe sull'asse 0 |
| `experts.gate_up_proj` | `[512, 1280, 2560]` = `[E,2I,H]` | selezionare esperti sull'asse 0 |
| `experts.down_proj` | `[512, 2560, 640]` = `[E,H,I]` | selezionare esperti sull'asse 0 |
| `shared_expert.gate_proj.weight` | `[640, 2560]` | preservare |
| `shared_expert.up_proj.weight` | `[640, 2560]` | preservare |
| `shared_expert.down_proj.weight` | `[2560, 640]` | preservare |
| `shared_expert_gate.weight` | `[1, 2560]` | preservare |

`gate` è `Qwen4ExpTextTopKRouter`, non `nn.Linear`. Gli esperti sono `Qwen4ExpTextExperts`, non una `ModuleList`. I due parametri fusi non hanno il suffisso `.weight`. Dentro `gate_up_proj[e]`, le prime I righe sono la proiezione gate e le successive I la proiezione up: nessuna trasposizione è richiesta per selezionare E.

L'input del MoE è `[batch, sequence, H]`; il router lavora su `[batch*sequence,H]`. La larghezza 4H=10240 dei flussi residui non è la larghezza dei tensori degli esperti: il modulo `mlp_hyper_connection` la riduce a H prima del MoE.

## 4. Contratto matematico e maschera

Per ciascun token x, con proiezioni senza bias:

1. `z = linear(x, gate.weight)`; softmax di z in float32.
2. Selezione dei K=10 valori maggiori; normalizzazione sulla loro somma quando `norm_topk_prob=True`.
3. Conversione dei pesi di routing al dtype dei logits.
4. Per ogni esperto selezionato e: `f_e(x) = W_down[e] @ (silu(W_gate[e] @ x) * (W_up[e] @ x))`.
5. Somma pesata degli esperti instradati, più `sigmoid(shared_expert_gate(x)) * shared_expert(x)`.

Il risultato del router è una tripla `(router_logits, routing_weights, selected_experts)` con forme `[T,E]`, `[T,K]`, `[T,K]`. Il MoE usa direttamente il secondo e il terzo elemento. Un hook che modifica soltanto il primo elemento **non applica il pruning**.

Contratto proposto per `masked(model, rows)`:

- Validare 48 liste di ID originali, unici, ordinati, in `[0,E)`, con un budget uniforme B e `10 <= B <= 512`.
- Nell'hook del router mascherare con `-inf` i logits degli esperti esclusi, ricalcolare softmax/top-k/normalizzazione/cast e restituire tutti e tre gli elementi coerenti. Gli indici rimangono nello spazio originale 0…511.
- Con B=E, lasciare passare la tripla nativa senza ricalcolarla.
- Rimuovere gli hook anche in caso di errore; evitare hook duplicati e kernel alternativi non verificati.
- Esporre alla profilazione i logits `output[0]` e riprodurre la stessa aritmetica e lo stesso cast del router per i punteggi. La selezione degli esperti deve restare in float32; non anticipare il cast.

Nella prima versione limitare il contratto al valore effettivo `norm_topk_prob=True`. Per logits finiti e selezione non ambigua, mascherare gli esclusi e compattare i mantenuti rappresentano la stessa funzione matematica. In virgola mobile la riduzione della softmax può cambiare arrotondamenti; top-k a pari merito può scegliere indici diversi. I controlli devono coprire questi casi senza promettere equivalenza bit a bit universale.

## 5. Compattazione e serializzazione

Per ogni layer, costruire un indice ordinato S e applicare `index_select(0,S)` a `gate.weight`, `experts.gate_up_proj` ed `experts.down_proj`. Ricreare `nn.Parameter` contigui preservando dtype, device e `requires_grad`; mantenere le classi native. Aggiornare `gate.num_experts` ed `experts.num_experts` a B. `gate.top_k` resta 10; sul blocco non esiste il contatore `num_experts` previsto dall'adattatore attuale.

Aggiornare il `num_experts` della configurazione testuale effettiva, le eventuali copie della configurazione e l'attributo del wrapper testuale `model.num_experts`. Incorporare il profilo e la mappa `nuovo_ID -> ID_originale` per ogni layer. Non cambiare H, I, attivazione, shared expert o top-k. Se esistono hook di dispatch/offload, la sostituzione dei parametri deve rispettarli: il normale codice per un solo device non è automaticamente valido su un modello distribuito.

Distinguere tre livelli di formato:

| Livello | Specifica |
| --- | --- |
| Runtime | tensori fusi indicati al punto 3 |
| Checkpoint originale | da verificare sull'indice e sugli header della revisione scelta |
| Checkpoint esportato | formato dichiarato e verificato con salvataggio/ricaricamento del runtime fissato |

Il registro di conversione Transformers [4] può unire proiezioni per esperto `experts.{e}.gate_proj.weight`, `up_proj.weight`, `down_proj.weight` nei tensori fusi. La presenza di questa regola non dimostra che il checkpoint specifico usi tale formato. È vietato dedurre il layout su disco dalla sola forma in memoria.

PLE richiede una trasformazione separata: `ngram_embedding.shard_*.weight` viene concatenato lungo l'asse delle righe nel parametro runtime `ngram_embedding.weight`. Con `save_original_format=True`, il runtime può invertire le conversioni; scegliere esplicitamente il formato di export e verificare indice, numero/ordine delle parti e nuovo conteggio esperti. Non copiare l'indice sorgente dopo avere riscritto i pesi.

Il checkpoint dichiara anche MTP. La classe upstream lo ignora al caricamento (`^mtp.*`); il percorso solo testo ignora inoltre `model.visual.*`. Un `loading_info` apparentemente pulito non garantisce quindi la conservazione di questi pesi. Il manifest deve enumerare anche le chiavi intenzionalmente escluse, ricavate dall'indice sorgente.

Proposta di perimetro: prima uscita HF con wrapper multimodale preservato e valutazione testuale; visione conservata ma non dichiarata validata dalla sola prova testuale. MTP va disabilitato esplicitamente nell'artefatto se il loader non lo rappresenta, con metadati coerenti verificati in round trip. Preservarlo richiede un percorso dedicato: lasciare 512 esperti nel modulo MTP mentre si riduce a B la configurazione condivisa potrebbe produrre un checkpoint incoerente. Un'eventuale uscita solo testo va descritta come derivato distinto, mai come conservazione integrale del modello multimodale.

## 6. Moduli da conservare e memoria

Non modificare `linear_attn`, `self_attn` e il suo `indexer`, `attn_hyper_connection`, `mlp_hyper_connection`, `hyper_connection_mixer`, PLE, embedding lessicali, `lm_head` e visione. I due embedding lessicali di input/output sono separati, ciascuno `[248320,2560]`.

PLE è nel secondo layer: `layers[1].ple`. I 16 head n-gram derivano da `(3-1)*8`; ciascuno produce 160 componenti. Applicando l'algoritmo upstream dei numeri primi e il padding a 128 si ottengono 320001536 righe: il parametro `ple.ple_embedding.ngram_embedding.weight` ha forma `[320001536,160]`, ossia 51200245760 parametri. È un calcolo dal codice, non una lettura degli header del file. Le 128 parti logiche PLE non sono 128 esperti e non vanno confuse con gli shard fisici safetensors.

Calcolo dei soli pesi degli esperti instradati del backbone: `48 * E * 3 * 2560 * 640`. In BF16, con 2 byte per parametro e GiB=2^30 byte:

| E mantenuti per layer | Frazione | Pesi esperti, GiB | Risparmio rispetto a 512, GiB |
| --- | --- | --- | --- |
| 512 | 100% | 225,00000 | 0 |
| 496 | 96,875% | 217,96875 | 7,03125 |
| 480 | 93,75% | 210,93750 | 14,06250 |
| 448 | 87,5% | 196,87500 | 28,12500 |
| 384 | 75% | 168,75000 | 56,25000 |

PLE aggiunge circa 95,368 GiB BF16 invariati: **esperti + PLE occupano già circa 320,368 GiB**, prima di attenzione, router, shared expert, visione, MTP, cache, attivazioni e buffer temporanei. Il pruning degli esperti non riduce la tabella PLE. Il profilo `l4x1` con caricamento globale `.to(cuda)` non è utilizzabile per questa configurazione.

La concatenazione PLE, i nuovi tensori durante la compattazione e il salvataggio possono richiedere memoria aggiuntiva. La pianificazione del Job deve includere RAM host, VRAM, disco temporaneo, originale e risultato; lo spazio dei soli parametri non è una misura del picco reale.

I budget 48/56/60/62 del profilo precedente conserverebbero rispettivamente il 9,375%, 10,9375%, 11,71875% e 12,109375% dei 512 esperti. Sono formalmente sopra top-k, ma estremamente diversi dal taglio originale su 64 esperti. Per riprodurne soltanto le frazioni usare 384/448/480/496, aggiungendo sempre 512 come controllo. Questo non garantisce qualità equivalente: selezione e audit restano necessari.

## 7. Modifiche necessarie in MoEDiet

Nome proposto dell'implementazione: `qwen4_exp_fused_topk_v1`. È un identificatore di progetto, non un adattatore già disponibile.

| Componente | Modifica pianificata |
| --- | --- |
| `architecture.py` | validazione distinta per layout fuso, configurazione annidata e default legati al runtime; selezione di tutti i 48 layer |
| `adapters.py` | nuovo adattatore con router a tripla, `index_select` sui parametri e verifica matematica indipendente; mantenere i vecchi adattatori |
| `adapter_for` | dispatch esplicito sull'implementazione e sul wrapper effettivo; rifiutare combinazioni sconosciute |
| `moediet.load_model` | scegliere classe nativa e perimetro del checkpoint, contabilizzare chiavi ignorate; introdurre una strategia di placement capace di caricare il modello |
| `routing.py` | conservare selezione e audit, adattare la cattura/cast del routing e il device degli input; valutare accumulo delle statistiche per layer per contenere memoria |
| `fidelity.py` | mantenere separati audit di qualità ed equivalenza numerica; verificare forward e generazione con cache del backbone ibrido |
| `preflight.py` | risolvere revisione, leggere config/indice/header, verificare tensori e fattibilità prima di creare il Job |
| `requirements*.txt`, `hf_jobs.py` | runtime dedicato fissato a versione/commit e dipendenze compatibili; eliminare il vincolo globale 4.55.4 solo per il nuovo percorso |
| `architectures/qwen4_exp.json` | abilitare soltanto dopo le verifiche; per ora mantenere `implementation=unsupported` |
| `gguf_export.py` | progetto separato: un checkpoint HF valido non implica export GGUF Q4 valido |

Un aggiornamento globale a Transformers 5.x potrebbe cambiare anche il layout di OLMoE/Qwen2/Qwen3: non dichiarare conservata la loro compatibilità senza ripetere la suite. Per contenere il rischio, separare inizialmente l'ambiente Qwen4-Exp.

## 8. Preflight dei pesi da implementare

Il primo passo operativo non richiede allocare il modello: risolvere il commit HF, acquisire `config.json`, `model.safetensors.index.json` e gli header safetensors mediante letture limitate. Per ogni chiave registrare nome originale, shard, dtype, forma, byte e destinazione runtime. Verificare i 48 router, tutte le banche di esperti, shared expert, PLE, visione e MTP; rifiutare conteggi, orientamenti o formati non riconosciuti. Il manifest deve distinguere forme lette dai file da forme ottenute per conversione.

Bloccare checkpoint quantizzati e varianti derivate nel primo adattatore. Non dedurre un supporto FP8/AWQ/GGUF dal nome del modello. Fissare separatamente la revisione dei pesi e quella del runtime. Non sostituire silenziosamente la revisione del profilo con `main`.

## 9. Criteri per abilitare il supporto

1. Fixture piccola della classe Transformers completa, con quattro layer per coprire GDN e QSA, PLE ridotto sul secondo layer, quattro flussi residui e shared expert.
2. Verifica delle forme e del contratto nativo; maschera identità; budget minimo K e budget intermedio; input con batch e più token; casi di pari merito e pesi di routing molto piccoli.
3. Equivalenza fra modello mascherato e compattato, prima e dopo `save_pretrained`/reload; verifica della mappa degli ID e dell'assenza di modifiche ai pesi preservati.
4. Prefill e decode con cache, EOS e padding; indici n-gram e QSA invariati; confronto eager/SDPA dove utilizzati. La tolleranza numerica deve essere motivata per dtype, distinta dalla soglia KL.
5. Verifica dei formati fuso e originale se entrambi dichiarati supportati; policy MTP e visione verificata con inventario completo dei tensori.
6. Esecuzione sul checkpoint reale soltanto dopo un piano di memoria sostenibile; audit separato sui tag del corpus e sul gruppo di controllo, poi eventuale studio GGUF.

## 10. Controllo isolato effettivamente eseguito

Sono state estratte mediante AST le quattro classi upstream `Qwen4ExpTextMLP`, `Qwen4ExpTextExperts`, `Qwen4ExpTextTopKRouter` e `Qwen4ExpTextSparseMoeBlock`. Il decoratore di selezione dei kernel è stato escluso per eseguire il percorso eager PyTorch. Ambiente: CPU, PyTorch 2.15.0+cpu; pesi casuali FP32, seed 19; E=8, K=2, H=16, I=12, shared=8, batch=2, sequenza=7; esperti mantenuti `[0,2,5,7]`.

| Controllo | Esito |
| --- | --- |
| Modificare soltanto i logits della tripla lascia identica l'uscita originale | confermato |
| Ricalcolare l'intera tripla equivale alla compattazione dei tre tensori | differenza assoluta massima 0,0 nel caso provato |
| La maschera corretta cambia l'uscita rispetto all'originale | differenza assoluta massima 0,9252310991287231 |
| Rimozione degli hook dopo il controllo | verificata |

Questo esperimento sostiene la scelta del contratto di masking. Non prova caricamento, salvataggio, backend ottimizzati, BF16, qualità, visione, cache o compatibilità del modello completo. Il runtime locale installato rimane Transformers 4.55.4; non è stato aggiornato.

## Fonti

- [1 — Configurazione ufficiale del checkpoint, main consultato il 5 novembre 2026](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json).
- [2 — Modellazione Transformers al commit ispezionato](https://github.com/huggingface/transformers/blob/cdb75de4adf71059c01f58dbd2d24d85084a397d/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py). Blob: `a3df0ac236d3510df56660b8d23097b1f6dda24b`. Parti principali: esperti/router/MoE 891–987; PLE 1067–1244; decoder 1247–1299; wrapper 1568 e 2362.
- [3 — Configurazione Transformers allo stesso commit](https://github.com/huggingface/transformers/blob/cdb75de4adf71059c01f58dbd2d24d85084a397d/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py).
- [4 — Regole di conversione dei pesi allo stesso commit](https://github.com/huggingface/transformers/blob/cdb75de4adf71059c01f58dbd2d24d85084a397d/src/transformers/conversion_mapping.py).
- [5 — Documentazione ufficiale Qwen4-Exp](https://huggingface.co/docs/transformers/main/model_doc/qwen4_exp).

I percorsi dei sorgenti MoEDiet citati sono relativi alla cartella di questa specifica. Il documento aggiunge una specifica; non modifica l'adattatore o i parametri dell'esperimento.
