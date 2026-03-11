# Multimodal OCT + Fundus pipeline (`train_multimodal.py`)

Script prêt pour expérimentation avec 3 modes :
- `prob_fusion` : fusion simple des probabilités OCT + fundus
- `late_fusion` : entraînement multimodal tardif sur embeddings + gating + modality dropout
- `xai` : attribution spatiale (`saliency` gradient input-level honnête) + occlusion

## Audit de compatibilité effectué
Le script a été réaligné sur les APIs réelles du dépôt :
- `oct_volume_search.py` : `OCTVolumeNet.forward()` retourne **`(logits, emb)`**.
- `train_fundus_only.py` : logique reprise de la vraie architecture fundus (`DinoBackbone`, double branche IR/FAF, concat, tête `LayerNorm -> Linear -> GELU -> Dropout -> Linear`, unfreeze derniers blocs + norm, class weights `sqrt(inv_freq)`).

## Réutilisation des scripts existants
- OCT : réutilise `OCTVolumeNet`, `infer_volume_layout`, `robust_normalize`, `confusion_matrix`, `metrics_from_cm`.
- Fundus : implémentation alignée structurellement avec `train_fundus_only.py` pour compatibilité checkpoint.

## Hypothèses sur le manifest CSV
Format confirmé côté utilisateur : `export_manifest.csv` avec colonnes
`sgene,patient_key,laterality,oct_path,faf_path,ir_path,has_oct,has_faf,has_ir,split,export_dir`.

Le script supporte maintenant :
- `sgene` (prioritaire) ou `gene` comme colonne label (auto-détection),
- `patient_key` + `laterality` pour construire `case_id` si absent,
- les colonnes de disponibilité et chemins (`has_*`, `*_path`).

## Données manquantes
- `has_fundus = 1` seulement si IR + FAF disponibles (mode strict par défaut).
- Les tenseurs manquants sont remplacés par des zéros + masque explicite (`has_oct`, `has_fundus`).
- En **probability fusion**, le calcul est fait par sous-batch conditionnel : une modalité absente n'est pas exécutée pour l'échantillon.
- Le modèle late fusion applique un modality dropout configurable sans masquer les 2 modalités simultanément.

## Embeddings et shapes
- OCT embedding dim : inféré de l'API réelle via `OCTVolumeNet.head.in_features`.
- Fundus embedding dim : inféré dynamiquement par passage dummy via `extract_embedding`.
- Pas de dimensions d'embedding hardcodées pour les projections multimodales.

## Chargement de checkpoints
- Support des formats fréquents (`model`, `state_dict`, `model_state_dict`, `net`).
- Logs explicites : `missing_keys` / `unexpected_keys`.
- Option CLI `--strict-load` pour imposer `strict=True`.

## Exemple d'utilisation

### 1) Fusion simple
```bash
python train_multimodal.py \
  --mode prob_fusion \
  --manifest /path/manifest.csv \
  --split val \
  --oct-ckpt /path/oct.pt \
  --fundus-ckpt /path/fundus.pt \
  --w-oct 0.5 --w-fundus 0.5 \
  --out-dir runs_mm_prob
```

### 2) Entraînement late fusion
```bash
python train_multimodal.py \
  --mode late_fusion \
  --manifest /path/manifest.csv \
  --oct-ckpt /path/oct.pt \
  --fundus-ckpt /path/fundus.pt \
  --epochs 30 --batch-size 4 --amp \
  --lr-head 1e-3 \
  --lr-oct-backbone 1e-5 \
  --lr-fundus-backbone 5e-5 \
  --modality-dropout-oct 0.15 \
  --modality-dropout-fundus 0.15 \
  --freeze-backbones \
  --unfreeze-last-oct-stages 1 \
  --unfreeze-last-fundus-blocks 4 \
  --out-dir runs_mm_late
```

### 3) XAI
```bash
python train_multimodal.py \
  --mode xai \
  --manifest /path/manifest.csv \
  --oct-ckpt /path/oct.pt \
  --fundus-ckpt /path/fundus.pt \
  --checkpoint runs_mm_late/best_multimodal.pt \
  --split val \
  --case-id SOME_CASE \
  --xai-method saliency \
  --out-dir runs_mm_xai
```

## Sorties sauvegardées
- `config.json`
- `metrics_*.json`
- `metrics_history.json`
- `best_multimodal.pt`
- `predictions_val.csv`
- `ablation.json`
- figures confusion matrix (`.png`)
- dossier `xai/<case_id>/` avec PNG + JSON

## Important pour l'interprétation
- Les poids de gating sont des indicateurs internes, **pas** des preuves causales.
- La contribution des modalités doit être évaluée principalement par l'ablation (`full`, `no_oct`, `no_fundus`).
- Les cartes spatiales (saliency/occlusion) sont exploratoires.
- La validité des attributions dépend de l'architecture des backbones (ConvNeXt/ViT ici).

## Points à adapter manuellement si besoin
- Si votre manifest utilise d'autres noms de colonnes, adapter `MultimodalDataset`.
- Si vos checkpoints ont un mapping de classes différent, vérifier l'ordre des classes dans `gene`.
- Si votre OCT n'est pas stocké en volumes compatibles `infer_volume_layout`, adapter `_load_oct`.
