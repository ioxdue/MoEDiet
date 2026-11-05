# Esperimento HF conservativo con sonde ampliate

Il profilo [config.hf.conservative.json](config.hf.conservative.json) prova tagli più piccoli
del profilo iniziale, senza allentare la soglia di fedeltà:

| Parametro | Valore |
| --- | --- |
| Esperti conservati per strato | 48, 56, 60, 62; ripiego automatico a 64 |
| Frazione di esperti rimossi | 25%, 12,5%, 6,25%, 3,125% rispettivamente |
| Soglia KL | 0,10 nat/token, separatamente per ogni gruppo |
| Penalità del controllo `lambda` | 0,5 |
| Continuazione di riferimento | Fino a 32 token per sonda; EOS può terminarla prima |
| Corpus | `corpora/dev-expanded-v1.json` |
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
