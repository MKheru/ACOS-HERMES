# Audit auto-apprentissage Hermes Agent — État réel

> **Date** : 2026-05-01
> **Auditeur** : Claude (via SSH côté Khéri)
> **Cible** : install hermes-agent sur `acos-hermes-01` (VPS Hetzner CCX23)
> **Contexte** : Khéri a soulevé que "une des forces de Hermes nativement c'est l'autoapprentissage. Pour ma part, sans doute que l'on a pas encore assez de recule, mais je ne vois aucun auto-apprentissage."

---

## TL;DR

AH **apprend partiellement et passivement** mais **n'auto-améliore pas activement**.
Le projet d'auto-amélioration est **designé mais pas implémenté**.
La base est solide, les outils existent — **rien n'est branché en boucle**.

→ **Verdict : 30 % du potentiel d'auto-apprentissage exploité.** Le reste dort.

---

## 1. Ce qui MARCHE (auto-apprentissage passif)

| Composant | État | Évaluation |
|---|---|---|
| `~/.hermes/memories/MEMORY.md` (1060 octets) | ✅ Actif — 4 entrées sur SMCP, ACOS repo distinction, WS14 status | 🟡 dernière modif **2026-04-29 20:34** (il y a 2 jours alors qu'on a eu 100+ messages depuis). AH ne l'updatait pas spontanément |
| `~/.hermes/memories/USER.md` (357 octets) | ✅ Actif — orthographe Kheru, langue FR, préférence "pas de 36 questions", "incohérences m'irritent" | 🟢 4 lignes utiles. Suffisant comme base |
| `~/.hermes/state.db` (4.3 MB SQLite) | ✅ **18 sessions, 1442 messages**, FTS index full-text | 🟢 base d'apprentissage très riche, **mais utilisée seulement en lecture sur demande explicite** (`session_search` 17 calls / 30j = 2.5%) |
| `hermes insights` (analyse 30j) | ✅ Fonctionne — tool patterns, top skills, activity, sessions notables | 🟡 observation passive, aucune action automatique sur les patterns détectés |
| Sessions journaliers (`~/.hermes/sessions/*.jsonl`) | ✅ 7 fichiers, 39 KB → 360 KB chacun | 🟢 archive complète conversationnelle |

## 2. Ce qui DORT

| Composant | État | Pourquoi ça dort |
|---|---|---|
| **8 memory providers externes** : mem0, honcho, hindsight, byterover, retaindb, openviking, holographic, supermemory | ❌ TOUS désactivés | Aucun n'a été configuré. Le built-in (MEMORY.md/USER.md) est seul actif |
| **Lab self-improv** (`/home/hermes/lab-self-improv/`) | ⏳ **Designé, pas implémenté** | Un seul fichier `SELF_IMPROV_STATUS.md` (5926 octets) qui décrit le loop d'auto-amélioration en 4 axes (déclencheurs / réflexion / actions / prévention). **Aucun code, aucun run** |
| **Plugins bundled** (disk-cleanup, spotify) | ❌ TOUS désactivés | Marqués "not enabled" |
| **24 catégories de skills disponibles** (acos, autonomous-ai-agents, devops, github, mcp, mlops, red-teaming, research, software-development, etc.) | ⚠️ **Seulement 2 utilisées** : `acos-apex-workstream` (6 loads), `xurl` (1 load) | Disponibles mais jamais déclenchés sur les autres workflows |
| **Auto-création de skills** depuis patterns détectés | ❌ Pas câblé | Le code existe (`agent/memory_manager.py`) mais aucune routine ne l'invoque sur erreur |
| **Cherry-pick automatique upstream NousResearch** | ❌ Manuel | On a fait un cherry-pick à la main (`8ed599dc` auto-backup HERMES_HOME). Pas de cron qui détecte / propose |

## 3. Diagnostic — pourquoi AH ne s'auto-améliore pas

AH lui-même a écrit `SELF_IMPROV_STATUS.md` le 2026-04-29 et a parfaitement identifié le problème :

> *"AH a des outils d'auto-amélioration. Mais **personne ne les enchaîne** dans un loop réflexif : après une erreur, pourquoi ai-je échoué ? Quelle pattern dois-je retenir ? Le gap est **le cement entre les briques**, pas les briques elles-mêmes."*

**Incident déclencheur** : 2026-04-29 — AH a créé un doublon `SMCP_STATUS.md` au lieu de mettre à jour l'existant. Erreur évitable avec un simple `test -f` avant write. Aucun mécanisme préventif n'a empêché l'erreur ni n'en a tiré une leçon automatique.

## 4. Données mesurables (Insights 30 jours)

```
Sessions actives     : 18 (5 jours actifs)
Messages totaux      : 1442
Tokens (max session) : 2 091 320 (Apr 27)
Messages (max sess.) : 262 (Apr 29)

Top tools used (30j) :
   terminal           36 % (247 calls)
   execute_code       19 % (128)
   search_files       15 % (106)
   read_file          14 % (99)
   patch               6 % (40)
   session_search    2.5 % (17)   ← retrieval mémoire
   memory            1.2 % (8)    ← writes mémoire (RARE !)

Skills loaded        : 2 / 24 catégories disponibles
MEMORY.md size       : 1060 octets (~4 entrées)
USER.md size         : 357 octets (~4 lignes)
Last memory update   : 48h ago (alors qu'on a eu 100+ msgs)
```

**Ratio reads / writes mémoire** : 17 reads / 8 writes en 30 jours. AH **lit plus qu'il n'apprend**.

---

## 5. Recommandations actionnables (par priorité)

### 🟢 Quick wins — ~30 min total

1. **Activer un memory provider externe** : `mem0` ou `hindsight` (free tier). Le code est déjà là, juste à configurer.
   ```bash
   sudo systemd-run --uid=hermes ... hermes memory setup
   # → choisir mem0 (free tier 1000 facts) ou hindsight (local SQLite, pas d'API)
   ```
   AH stocke automatiquement les facts importants pendant chaque session.

2. **Activer le plugin `disk-cleanup`** :
   ```bash
   sudo systemd-run --uid=hermes ... hermes plugins enable disk-cleanup
   ```

3. **Updater `~/.hermes/memories/USER.md`** avec plus d'info sur Khéri (timezone GMT-3, plan MiniMax en cours, refus de subscriptions tiers, etc.) — peut s'écrire à la main via SSH+sudoedit.

### 🟡 Medium effort — 1-2h

4. **Implémenter le self-improv loop** que AH a designé : ~150 lignes dans `agent/self_improv.py`. Cron qui :
   - Détecte les erreurs / Denys / feedbacks user dans la session courante
   - Classifie (factuel / procédural / raisonnement / vigilance)
   - Crée un memory write OU un skill update OU un announce Discord
   - Stocke dans `~/lab-self-improv/lessons.jsonl`

5. **Hook MEMORY.md auto-update** dans le gateway : à chaque fin de session, AH écrit dans MEMORY.md les "facts" extraits via le LLM lui-même. Pas besoin que Khéri lui dise "remember X" à chaque fois.

### 🔵 Long-term — weeks

6. **Skills auto-promotion** : observer quels patterns reviennent (3+ fois → proposer la création d'un skill). Implémenter un classifier qui propose à AH d'écrire un skill quand un workflow se répète.

7. **Insights → action** : actuellement Insights produit un rapport passif. Le brancher à des actions automatiques (ex: si "memory" tool < 2% → alerte "AH n'apprend pas assez", trigger d'un self-improv check).

8. **Cherry-pick upstream automatique** : cron weekly qui scanne NousResearch/hermes-agent pour les commits récents, identifie ceux pertinents (auto-update, security, productivity), poste sur Discord pour arbitrage Khéri. **Documenté dans HERMES.md §7 mais pas implémenté**.

---

## 6. Action immédiate proposée — Patch 13.5 self-improv loop

**Effort estimé** : 2-3h. **Bénéfice direct mesurable** : AH écrit dans MEMORY.md à chaque fin de session, classe ses erreurs, propose des skills.

**Pré-requis** : aucun (peut être codé dès maintenant, indépendant du système vocal premium).

**Fichiers prévus** :
- `agent/self_improv.py` (nouveau, ~150 lignes)
- Hook dans `gateway/run.py:_start_cron_ticker` (1 appel toutes les N sessions)
- `tests/agent/test_self_improv.py` (~15 tests unitaires)

---

## 7. Annexes — fichiers / commandes utiles

```bash
# Voir l'état actuel mémoire
sudo cat /home/hermes/.hermes/memories/MEMORY.md
sudo cat /home/hermes/.hermes/memories/USER.md

# Voir les sessions stockées
sudo /home/hermes/hermes-agent/venv/bin/python3 -c \
  "import sqlite3;c=sqlite3.connect('/home/hermes/.hermes/state.db'); \
  print(c.execute('SELECT COUNT(*) FROM sessions').fetchone(), \
  c.execute('SELECT COUNT(*) FROM messages').fetchone())"

# Insights 30 jours
sudo systemd-run --uid=hermes --gid=hermes \
  -p EnvironmentFile=/etc/hermes/env.list \
  -p Environment=HERMES_HOME=/home/hermes/.hermes \
  --pipe --wait --collect \
  /home/hermes/hermes-agent/venv/bin/hermes insights --days 30

# Lister memory providers disponibles
sudo systemd-run --uid=hermes --gid=hermes \
  -p EnvironmentFile=/etc/hermes/env.list \
  -p Environment=HERMES_HOME=/home/hermes/.hermes \
  --pipe --wait --collect \
  /home/hermes/hermes-agent/venv/bin/hermes memory status
```
