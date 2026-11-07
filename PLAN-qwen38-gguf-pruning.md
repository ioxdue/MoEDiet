# Piano PR: pruning diretto del GGUF Qwen3.8-Flash-Next

Stato locale, 5 novembre 2026: primo incremento implementato in `gguf_prune.py`,
con ambiente `.venv-gguf` e istruzioni nel README. Verificata la trasformazione
dei file su fixture sintetiche; inferenza e qualità del modello reale non misurate.
La profilazione automatica descritta nel passo successivo resta da implementare.

## Obiettivo e scelta minima

Produrre un GGUF fisicamente ridotto partendo dalla variante Unsloth UD-IQ4_XS,
conservando i byte quantizzati degli esperti mantenuti. Il primo incremento
implementativo comprende inventario, stima e compattazione da una selezione
esplicita. La selezione automatica per programmazione agentica e la validazione
della qualità costituiscono il passo successivo, con dipendenze dichiarate.

Per questo percorso non occorre convertire tutto in BF16/F16, riaddestrare,
requantizzare, aggiornare Transformers o costruire un adattatore universale.
La richiesta di un formato in un JSON non ne dimostra la compatibilità.

Base esaminata: MoEDiet @ `ea7a9d7b3063fa733791218d6e6564bded8767ec` (revisione del repository).

## Stato del repository

- `run.py`, `preflight.py` e `moediet.py` avviano la pipeline HF/Transformers.
- `adapters.py` implementa router Linear ed esperti ModuleList; non tensori GGUF.
- `architectures/qwen4_exp.json` rifiuta correttamente il modello prima del Job.
- `gguf_export.py` converte un checkpoint OLMoE accettato in F16 e poi Q4.
  Non legge né pota un GGUF in ingresso.
- `routing.py` calcola massa di routing e selezioni; `fidelity.py` confronta
  distribuzioni tramite modelli PyTorch. Non sono chiamabili direttamente sul GGUF.
- `probes.py` separa profile/select/audit e controllo. Un tag sconosciuto riceve
  sonde generiche: scrivere soltanto "agentic-coding" non crea un corpus agentico.

Questi percorsi restano separati. Il descrittore HF `qwen4_exp` resta non supportato:
un'operazione su file GGUF non rende funzionante l'adattatore Transformers.

## Sorgente e contabilità

Sorgente: [unsloth/Qwen3.8-Flash-Next-GGUF, UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/d96201194f103c906994f283f504209dc8d2a10c/UD-IQ4_XS).
Revisione analizzata: `d96201194f103c906994f283f504209dc8d2a10c`.

Prefisso dei quattro file: `UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-`.

| Suffisso | Byte |
| --- | ---: |
| 00001-of-00004.gguf | 10 946 618 |
| 00002-of-00004.gguf | 49 604 491 008 |
| 00003-of-00004.gguf | 15 256 783 680 |
| 00004-of-00004.gguf | 28 800 138 432 |
| Totale | 93 672 359 738 |

L'inventario precedente ha letto le intestazioni, non tutti i payload.
Il primo shard contiene metadati/tokenizer e zero tensori; non è un modello autonomo.
Il gruppo comprende 1 224 tensori, 48 layer, 512 esperti instradati per layer e
top-k 10. MTP e mmproj non fanno parte di questo gruppo di file.

| Componente | Byte | GiB |
| --- | ---: | ---: |
| 144 tensori degli esperti instradati | 60 384 870 400 | 56,23779 |
| 48 router F32 | 251 658 240 | 0,23438 |
| PLE IQ4_NL | 28 800 138 240 | 26,82222 |
| Restanti pesi e overhead | 4 235 692 858 | 3,94480 |
| Totale | 93 672 359 738 | 87,23918 |

I sei formati degli esperti sono Q5_0, IQ3_XXS, IQ4_XS, IQ4_NL, Q2_0 e IQ2_S.
Il nome commerciale della variante non sostituisce il tipo di ogni tensore.

Per un budget uniforme B nei 48 layer, con le quantizzazioni preservate:

`GGUF_GiB(B) ≈ 30.76701527 + 56.47216797 × B / 512`

L'overhead dei nuovi shard può variare leggermente. Lo strumento deve derivare
i byte dall'inventario e dai blocchi GGML, non incorporare questa formula come
assunzione valida per altri checkpoint.

| B | GGUF stimato, GiB | Risparmio, GiB | Pesi esclusa PLE, GiB |
| ---: | ---: | ---: | ---: |
| 512 | 87,24 | 0,00 | 60,42 |
| 384 | 73,12 | 14,12 | 46,30 |
| 256 | 59,00 | 28,24 | 32,18 |
| 192 | 51,94 | 35,30 | 25,12 |
| 128 | 44,89 | 42,35 | 18,06 |

Il runtime [llama.cpp 19cce69](https://github.com/ggml-org/llama.cpp/blob/19cce6919992fdffd92e0ac152caaf1e2b8f254e/src/models/qwen4exp.cpp)
marca la PLE con `TENSOR_READ_LAZY`; la build locale espone `--lazy-mode auto`.
I 26,82 GiB sono quindi un costo su disco, non un obbligo di residenza completa
in RAM. La colonna senza PLE non è RAM totale: mancano pagine PLE effettivamente
lette, cache, attivazioni e buffer. Su Snapdragon la memoria GPU condivisa non
si somma alla RAM fisica. Questa analisi non prova velocità o compatibilità HTP/Adreno.

## Primo incremento: inventario e compattazione

### Superficie prevista

| File | Intervento |
| --- | --- |
| `gguf_prune.py` nuovo | CLI autonoma `inspect` e `compact`; convalida, contabilità, copia e report |
| `requirements-gguf.txt` nuovo | Dipendenze isolate; gguf-py da revisione verificata |
| `test_gguf_prune.py` nuovo | Piccoli GGUF sintetici, casi negativi e conservazione dei byte |
| `README.md` | Uso, limiti e distinzione tra compattazione e qualità |

Partire dalle API di [gguf-py alla revisione verificata](https://github.com/ggml-org/llama.cpp/tree/19cce6919992fdffd92e0ac152caaf1e2b8f254e/gguf-py),
che includono Q2_0 e IQ4_XS. Riutilizzare lettore, tipi, contabilità dei blocchi e
scrittura delle intestazioni. Incapsulare solo la copia a blocchi dei payload;
non sviluppare nuovi quantizzatori o una seconda specifica GGUF.

Comandi disponibili (usare il Python dell'ambiente `.venv-gguf`):

```text
python gguf_prune.py inspect --model <primo-shard.gguf> --budgets 256 192 128
python gguf_prune.py compact --model <primo-shard.gguf> --selection <selection.json> --out <nuova-directory>
```

Il primo incremento lavora su shard locali già disponibili. Download remoto,
HF Jobs, pubblicazione automatica e integrazione nel launcher non sono requisiti.

### Contratto di ingresso

1. Scoprire e controllare l'intero gruppo di shard: numerazione, split count,
   conteggio globale, metadati coerenti, nomi unici, offset e dimensioni nei file.
   Accettare il primo shard senza tensori.
2. Richiedere architettura GGUF `qwen4exp`, distinguendola dal model_type HF
   `qwen4_exp`. Controllare layout, top-k e numero di layer sul contenuto.
3. Supportare inizialmente soltanto le tre matrici esperte separate osservate:
   `blk.{layer}.ffn_gate_exps.weight`, `ffn_up_exps.weight`,
   `ffn_down_exps.weight`, con asse esperti `ne[2]`.
   Gate/up: [2560, 640, 512]; down: [640, 2560, 512], in ordine dimensioni GGUF.
   Rifiutare varianti fuse o altri layout finché non sono implementati e verificati.
4. Router `blk.{layer}.ffn_gate_inp.weight`: F32, [2560, 512],
   asse esperti `ne[1]`. Non confonderlo con il gate dell'esperto condiviso.
   Controllare eventuali ulteriori tensori associati agli esperti; se non previsti,
   interrompere prima della scrittura.
5. Manifesto versionato: identificazione della sorgente, file e SHA-256 attesi,
   layer espliciti 0..47, lista ordinata di ID originali unici per ogni layer,
   budget uniforme B e `10 <= B <= 512`. La revisione Hub è provenienza;
   gli hash locali verificati legano il manifesto ai byte realmente usati.
   Non accettare una vecchia maschera OLMoE come se fosse compatibile.
6. La provenienza della selezione può essere manuale o da una traccia verificata;
   una selezione manuale produce soltanto un artefatto sperimentale.
   L'ispezione delle intestazioni non dichiara verificati gli hash dei payload.
   La compattazione completa il controllo degli hash prima di promuovere l'output.

### Trasformazione

- Copiare le sezioni contigue degli esperti scelti nei tre tensori di ogni layer,
  senza dequantizzazione. Verificare che ogni sezione rispetti i blocchi del tipo
  GGML; una selezione per asse non è una selezione arbitraria di singoli byte.
- Applicare la stessa mappa old-ID -> new-ID alle righe del router, in ordine
  crescente degli ID originali. Preservare top-k 10 e le regole di routing native.
- Aggiornare `qwen4exp.expert_count`, forme, offset, allineamento e metadati split.
  Conservare il tipo di ciascun tensore; non etichettare tutto come IQ4_XS.
- Preservare i payload di shared expert/gate, PLE, attenzione, SSM,
  hyper-connections, embedding e output; conservare tokenizer/chat template
  e metadati architetturali non modificati.
- Scrivere in una nuova directory temporanea con buffer limitati e promuovere
  il risultato solo dopo la verifica. Nessuna sovrascrittura della sorgente;
  non riutilizzare hard link scrivibili per i payload da modificare.
- Report: sorgente e hash verificati, selezione e mappa, inventario prima/dopo,
  byte effettivi, revisioni degli strumenti e verifiche eseguite.
  Separare `structural_validation`, `runtime_smoke` e `quality_validation`:
  una fase non eseguita resta esplicitamente non eseguita.

La scrittura deve usare memoria proporzionale al buffer di copia e ai metadati,
non al modello dequantizzato. Misurare comunque il picco di memoria del processo
e distinguere mapping e page cache. Pianificare lo spazio prima di iniziare:
sorgente + candidato 256 richiedono circa 146,24 GiB, oltre ai temporanei;
una copia identità completa arriva a circa 174,48 GiB. Costruire i candidati
uno alla volta evita di accumulare tutte le varianti.

### Criteri di accettazione del primo incremento

- [x] Fixture multi-shard con primo shard vuoto di tensori e quantizzazione mista:
      caricabile dal lettore GGUF ufficiale dopo la trasformazione.
- [x] Selezione identità: payload di tutti i tensori e metadati semantici identici;
      il file intero può differire per intestazioni, split o padding.
- [x] Selezione ridotta: confronto esatto dei byte di ogni esperto e riga router
      con i rispettivi ID originali; hash identici per tutti i tensori preservati.
- [x] Errori prima dell'output definitivo per shard mancanti/troncati, duplicati,
      hash errati, layout sconosciuti, ID fuori intervallo, budget sotto top-k
      o selezioni incoerenti fra layer e tensori.
- [x] Tipo quantizzato, conteggio, allineamento e byte finali verificati anche
      con selezioni non contigue e formati diversi fra gate/up/down.
- [x] Prova di interruzione/spazio insufficiente: nessun risultato parziale
      presentato come completato; sorgente integra.
- [ ] Prova nativa identità con llama.cpp sulla fixture di architettura compatibile
      o sul modello reale, registrando prefissi e tolleranze numeriche. Se non
      eseguita, dichiarare verificata soltanto la trasformazione dei file.
- [x] Suite esistente senza regressioni: 20 test passati, un test nativo opzionale
      saltato per il tokenizer assente; 16 nuovi test GGUF passati.
      Le fixture dimostrano la correttezza
      della trasformazione, non la qualità di Qwen3.8 né il supporto acceleratori.

## Passo successivo: ottenere una selezione utile e validarla

Questo è il prerequisito per chiamare il risultato "specializzato per agenti".

1. **Baseline originale.** Eseguire il GGUF completo con una revisione llama.cpp
   fissata, prima su CPU. La baseline senza PLE contiene circa 60,42 GiB di pesi:
   non pianificare una profilazione residente interamente nei 47,6 GiB del portatile.
   Usare una macchina adeguata, oppure misurare esplicitamente un'esecuzione con
   paging accettandone il costo. Nessun Job a pagamento è avviato da questa PR.
2. **Routing osservato.** Verificare un piccolo helper nativo con callback di
   valutazione; il [grafo della revisione esaminata](https://github.com/ggml-org/llama.cpp/blob/19cce6919992fdffd92e0ac152caaf1e2b8f254e/src/llama-graph.cpp)
   espone `ffn_moe_topk` e `ffn_moe_weights_norm`/scaled. La presenza dei nomi
   non prova ancora un'acquisizione corretta: servono layer, token, ID e pesi
   effettivi allineati, senza perdere token fra microbatch o contare due volte.
   Non assumere che l'API chat di llama-server esponga questi dati.
3. **Corpus.** Usare casi distinti profile/select/audit: lettura repository,
   modifica di codice, interpretazione di output degli strumenti, test e recupero
   dagli errori. Le tracce devono includere continuazioni/turni rappresentativi,
   non solo domande iniziali. Usare il template nativo e registrare gli ID token.
   Conservare un controllo separato; partire da lambda=0, senza penalizzare
   competenze generali sulla base di un'etichetta.
4. **Selezione.** Riutilizzare la regola semplice della massa media di routing,
   con ordinamento stabile e provenienza del corpus. Valutare il primo budget
   sul set select; congelare la decisione prima dell'audit. Una frequenza bassa
   non dimostra che un esperto sia superfluo su casi non osservati.
5. **Validazione.** Prima il caso identità 512; poi un taglio lieve 496 per
   verificare il percorso. Solo dopo sperimentare 384 -> 256 -> 192 -> 128,
   fermandosi quando falliscono i criteri di qualità. Ogni candidato deriva
   dalla stessa sorgente originale, senza potare ripetutamente il precedente.
   Confrontare su prefissi identici le distribuzioni del candidato e della
   baseline; separare la KL di qualità dalla tolleranza numerica del writer.
   Se si implementa una baseline mascherata, la restrizione deve precedere
   softmax/top-k: azzerare righe del router non equivale a escludere esperti.
6. **Qualità e hardware.** Definire prima dei run le soglie KL e la perdita
   ammessa sui task eseguibili di programmazione/uso degli strumenti, riportando
   numerosità e risultati del controllo. Uno smoke test di generazione non basta.
   Sullo Snapdragon misurare memoria, prefill/decode e PLE a cache fredda/calda,
   con contesto e batch dichiarati. Passare poi a GPU e HTP separatamente:
   il supporto dei sei formati e delle operazioni va dimostrato per ciascuna build.

Fino a questi risultati non dichiarare qualità conservata, velocità ottenuta,
compatibilità ibrida CPU/GPU/NPU o una percentuale "sicura" di pruning.

## Fuori dal primo incremento

Pruning PLE o shared expert, Q2 aggiuntivo, distillazione, fine-tuning,
budget differenti per layer, supporto a tutte le architetture, MTP, input visivi,
patch di kernel Qualcomm e nuovi servizi di orchestrazione.

