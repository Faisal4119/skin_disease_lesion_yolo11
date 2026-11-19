"""Stage 2c — the label map: every (source, raw_label) -> (disease, malignancy).

The mapping is written out as code on purpose. A dict is diffable, reviewable and
re-runnable; a hand-edited spreadsheet is none of those. Edit the dicts below, re-run,
and Stage 4 picks up the new map.

Taxonomy design
---------------
Model A (disease) uses DermNet's 23 coarse groups as the backbone, because DermNet is
the largest source and its folders are already clinically coherent groupings. Other
sources fold into those groups. Where a source is more specific AND the specific class
is large enough to stand alone (basal cell carcinoma, squamous cell carcinoma, actinic
keratosis, melanoma, melanocytic nevus), it gets its own class instead. Stage 4 drops
any Model A class that ends up under MIN_IMAGES.

Model B (malignancy) is derived from the canonical disease, never from the source, so
the same condition can never carry two different malignancy labels.

Two special cases are resolved per image, not per label:
  - DermNet folder "Melanoma Skin Cancer Nevi and Moles": data/dermnet_folder11_labels.csv
    says melanoma / nevus / unsure per image. Unsure -> excluded from both models.
  - DermNet folder "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions":
    every image is malignant, but which malignancy is unknown -> Model B only.

SCIN labels not listed here are EXCLUDED from both models. The 164-label tail (804
images) is too thin per label and contains a few neoplastic diagnoses, so a blanket
"non-neoplastic" default would be wrong for some of them.

Output: data/label_map.csv  (source, raw_label, n_images, disease, malignancy, model_a, model_b, note)
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ROOT / "data" / "label_map_skeleton.csv"
OUT = ROOT / "data" / "label_map.csv"
MIN_IMAGES = 100  # informational here; enforced in Stage 4

EXCLUDE = "__exclude__"
PER_IMAGE = "__per_image__"

# ----------------------------------------------------------------------------------
# Canonical disease -> malignancy. Single source of truth for Model B.
# ----------------------------------------------------------------------------------
MALIGNANCY: dict[str, str] = {
    # malignant
    "melanoma": "malignant",
    "basal_cell_carcinoma": "malignant",
    "squamous_cell_carcinoma": "malignant",
    "actinic_keratosis": "malignant",          # pre-cancerous; Fitzpatrick17k convention
    "cutaneous_lymphoma": "malignant",
    "kaposi_sarcoma": "malignant",
    "malignant_lesion_nos": "malignant",       # DermNet folder 1, Model B only
    # benign neoplasms
    "melanocytic_nevus": "benign",
    "seborrheic_keratosis_benign_tumor": "benign",
    "vascular_tumor": "benign",
    # non-neoplastic
    "acne_rosacea": "non-neoplastic",
    "eczema_dermatitis": "non-neoplastic",
    "contact_dermatitis": "non-neoplastic",
    "psoriasis_lichen_planus": "non-neoplastic",
    "urticaria": "non-neoplastic",
    "fungal_infection": "non-neoplastic",
    "bacterial_infection": "non-neoplastic",
    "viral_warts_molluscum": "non-neoplastic",
    "herpes_std": "non-neoplastic",
    "infestation_bite": "non-neoplastic",
    "exanthem_drug_eruption": "non-neoplastic",
    "bullous_disease": "non-neoplastic",
    "vasculitis_purpura": "non-neoplastic",
    "lupus_connective_tissue": "non-neoplastic",
    "systemic_genetic_disease": "non-neoplastic",
    "pigmentation_disorder": "non-neoplastic",
    "hair_disease": "non-neoplastic",
    "nail_disease": "non-neoplastic",
    "granulomatous_disorder": "non-neoplastic",
    "photodermatosis": "non-neoplastic",
    "keratosis_pilaris": "non-neoplastic",
    "trauma_scar_bruise": "non-neoplastic",
    "miliaria": "non-neoplastic",
}

# ----------------------------------------------------------------------------------
# raw_label -> canonical disease, per source
# ----------------------------------------------------------------------------------
DERMNET = {
    "Acne and Rosacea Photos": "acne_rosacea",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "malignant_lesion_nos",
    "Atopic Dermatitis Photos": "eczema_dermatitis",
    "Bullous Disease Photos": "bullous_disease",
    "Cellulitis Impetigo and other Bacterial Infections": "bacterial_infection",
    "Eczema Photos": "eczema_dermatitis",
    "Exanthems and Drug Eruptions": "exanthem_drug_eruption",
    "Hair Loss Photos Alopecia and other Hair Diseases": "hair_disease",
    "Herpes HPV and other STDs Photos": "herpes_std",
    "Light Diseases and Disorders of Pigmentation": "pigmentation_disorder",
    "Lupus and other Connective Tissue diseases": "lupus_connective_tissue",
    "Melanoma Skin Cancer Nevi and Moles": PER_IMAGE,
    "Nail Fungus and other Nail Disease": "nail_disease",
    "Poison Ivy Photos and other Contact Dermatitis": "contact_dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases": "psoriasis_lichen_planus",
    "Scabies Lyme Disease and other Infestations and Bites": "infestation_bite",
    "Seborrheic Keratoses and other Benign Tumors": "seborrheic_keratosis_benign_tumor",
    "Systemic Disease": "systemic_genetic_disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "fungal_infection",
    "Urticaria Hives": "urticaria",
    "Vascular Tumors": "vascular_tumor",
    "Vasculitis Photos": "vasculitis_purpura",
    "Warts Molluscum and other Viral Infections": "viral_warts_molluscum",
}

PADUFES20 = {
    "BCC": "basal_cell_carcinoma",
    "SCC": "squamous_cell_carcinoma",
    "MEL": "melanoma",
    "ACK": "actinic_keratosis",
    "NEV": "melanocytic_nevus",
    "SEK": "seborrheic_keratosis_benign_tumor",
}

FITZPATRICK17K = {
    # malignant
    "basal cell carcinoma": "basal_cell_carcinoma",
    "squamous cell carcinoma": "squamous_cell_carcinoma",
    "melanoma": "melanoma",
    "actinic keratosis": "actinic_keratosis",
    "mycosis fungoides": "cutaneous_lymphoma",
    "kaposi sarcoma": "kaposi_sarcoma",
    # benign neoplasms
    "congenital nevus": "melanocytic_nevus",
    "porokeratosis actinic": "seborrheic_keratosis_benign_tumor",
    "porokeratosis of mibelli": "seborrheic_keratosis_benign_tumor",
    "syringoma": "seborrheic_keratosis_benign_tumor",
    "pilar cyst": "seborrheic_keratosis_benign_tumor",
    "milia": "seborrheic_keratosis_benign_tumor",
    "mucous cyst": "seborrheic_keratosis_benign_tumor",
    "dermatofibroma": "seborrheic_keratosis_benign_tumor",
    "pilomatricoma": "seborrheic_keratosis_benign_tumor",
    "fordyce spots": "seborrheic_keratosis_benign_tumor",
    "naevus comedonicus": "seborrheic_keratosis_benign_tumor",
    "epidermal nevus": "seborrheic_keratosis_benign_tumor",
    "juvenile xanthogranuloma": "seborrheic_keratosis_benign_tumor",
    "granuloma pyogenic": "vascular_tumor",
    "lymphangioma": "vascular_tumor",
    "telangiectases": "vascular_tumor",
    # papulosquamous
    "psoriasis": "psoriasis_lichen_planus",
    "lichen planus": "psoriasis_lichen_planus",
    "pityriasis rubra pilaris": "psoriasis_lichen_planus",
    "pityriasis rosea": "psoriasis_lichen_planus",
    "pityriasis lichenoides chronica": "psoriasis_lichen_planus",
    # eczema family
    "seborrheic dermatitis": "eczema_dermatitis",
    "lichen simplex": "eczema_dermatitis",
    "dyshidrotic eczema": "eczema_dermatitis",
    "eczema": "eczema_dermatitis",
    "cheilitis": "eczema_dermatitis",
    "prurigo nodularis": "eczema_dermatitis",     # inflammatory; overrides Fitz "benign" (see note)
    "allergic contact dermatitis": "contact_dermatitis",
    "urticaria": "urticaria",
    # infections / infestations
    "scabies": "infestation_bite",
    "tungiasis": "infestation_bite",
    "myiasis": "infestation_bite",
    "pediculosis lids": "infestation_bite",
    "nematode infection": "infestation_bite",
    "tick bite": "infestation_bite",
    "lyme disease": "infestation_bite",
    "folliculitis": "bacterial_infection",
    "paronychia": "nail_disease",
    # reactive / drug
    "fixed eruptions": "exanthem_drug_eruption",
    "erythema multiforme": "exanthem_drug_eruption",
    "erythema annulare centrifigum": "exanthem_drug_eruption",
    "photodermatoses": "photodermatosis",
    # bullous / keratinisation
    "hailey hailey disease": "bullous_disease",
    "dariers disease": "systemic_genetic_disease",
    "ichthyosis vulgaris": "systemic_genetic_disease",
    "keratosis pilaris": "keratosis_pilaris",
    # vasculitis / neutrophilic
    "neutrophilic dermatoses": "vasculitis_purpura",
    "erythema elevatum diutinum": "vasculitis_purpura",
    "erythema nodosum": "vasculitis_purpura",
    "behcets disease": "vasculitis_purpura",
    # connective tissue
    "lupus erythematosus": "lupus_connective_tissue",
    "lupus subacute": "lupus_connective_tissue",
    "scleroderma": "lupus_connective_tissue",
    "dermatomyositis": "lupus_connective_tissue",
    # granulomatous
    "granuloma annulare": "granulomatous_disorder",
    "sarcoidosis": "granulomatous_disorder",
    "necrobiosis lipoidica": "granulomatous_disorder",
    # systemic / genetic / deposition
    "neurofibromatosis": "systemic_genetic_disease",
    "tuberous sclerosis": "systemic_genetic_disease",
    "ehlers danlos syndrome": "systemic_genetic_disease",
    "acrodermatitis enteropathica": "systemic_genetic_disease",
    "scleromyxedema": "systemic_genetic_disease",
    "mucinosis": "systemic_genetic_disease",
    "lichen amyloidosis": "systemic_genetic_disease",
    "calcinosis cutis": "systemic_genetic_disease",
    "porphyria": "systemic_genetic_disease",
    "xeroderma pigmentosum": "systemic_genetic_disease",
    "langerhans cell histiocytosis": "systemic_genetic_disease",
    "aplasia cutis": "systemic_genetic_disease",
    "urticaria pigmentosa": "systemic_genetic_disease",
    "stasis edema": "systemic_genetic_disease",
    # pigmentation
    "vitiligo": "pigmentation_disorder",
    "acanthosis nigricans": "pigmentation_disorder",
    "drug induced pigmentary changes": "pigmentation_disorder",
    "papilomatosis confluentes and reticulate": "pigmentation_disorder",
    # acne family
    "acne": "acne_rosacea",
    "rosacea": "acne_rosacea",
    "rhinophyma": "acne_rosacea",
    "perioral dermatitis": "acne_rosacea",
    "hidradenitis": "acne_rosacea",
    # trauma / self-inflicted / scar
    "keloid": "trauma_scar_bruise",
    "neurotic excoriations": "trauma_scar_bruise",
    "factitial dermatitis": "trauma_scar_bruise",
    "striae": "trauma_scar_bruise",
}

SCIN = {
    "Eczema": "eczema_dermatitis",
    "Acute dermatitis, NOS": "eczema_dermatitis",
    "Chronic dermatitis, NOS": "eczema_dermatitis",
    "Acute and chronic dermatitis": "eczema_dermatitis",
    "Lichen Simplex Chronicus": "eczema_dermatitis",
    "Stasis Dermatitis": "eczema_dermatitis",
    "Seborrheic Dermatitis": "eczema_dermatitis",
    "Prurigo nodularis": "eczema_dermatitis",
    "Allergic Contact Dermatitis": "contact_dermatitis",
    "CD - Contact dermatitis": "contact_dermatitis",
    "Irritant Contact Dermatitis": "contact_dermatitis",
    "Urticaria": "urticaria",
    "Hypersensitivity": "exanthem_drug_eruption",
    "Drug Rash": "exanthem_drug_eruption",
    "Viral Exanthem": "exanthem_drug_eruption",
    "Insect Bite": "infestation_bite",
    "Scabies": "infestation_bite",
    "Folliculitis": "bacterial_infection",
    "Impetigo": "bacterial_infection",
    "Cellulitis": "bacterial_infection",
    "Abscess": "bacterial_infection",
    "Ecthyma": "bacterial_infection",
    "Psoriasis": "psoriasis_lichen_planus",
    "Lichen planus/lichenoid eruption": "psoriasis_lichen_planus",
    "Pityriasis rosea": "psoriasis_lichen_planus",
    "Tinea": "fungal_infection",
    "Tinea Versicolor": "fungal_infection",
    "Intertrigo": "fungal_infection",
    "Herpes Zoster": "herpes_std",
    "Herpes Simplex": "herpes_std",
    "Verruca vulgaris": "viral_warts_molluscum",
    "Molluscum Contagiosum": "viral_warts_molluscum",
    "Acne": "acne_rosacea",
    "Rosacea": "acne_rosacea",
    "Perioral Dermatitis": "acne_rosacea",
    "Pigmented purpuric eruption": "vasculitis_purpura",
    "Leukocytoclastic Vasculitis": "vasculitis_purpura",
    "Purpura": "vasculitis_purpura",
    "Granuloma annulare": "granulomatous_disorder",
    "Photodermatitis": "photodermatosis",
    "Keratosis pilaris": "keratosis_pilaris",
    "Miliaria": "miliaria",
    "O/E - ecchymoses present": "trauma_scar_bruise",
    "Abrasion, scrape, or scab": "trauma_scar_bruise",
    "Scar Condition": "trauma_scar_bruise",
    "Inflicted skin lesions": "trauma_scar_bruise",
    "Actinic Keratosis": "actinic_keratosis",
}

NOTES = {
    ("fitzpatrick17k", "prurigo nodularis"):
        "Fitz three_partition says benign; mapped non-neoplastic (inflammatory) to match SCIN. Deliberate override.",
    ("scin", "Actinic Keratosis"):
        "Skeleton default was non-neoplastic; corrected to malignant to match PAD/Fitz convention.",
    ("dermnet", "Melanoma Skin Cancer Nevi and Moles"):
        "Per-image via data/dermnet_folder11_labels.csv: melanoma->melanoma/malignant, nevus->melanocytic_nevus/benign, unsure->excluded.",
    ("dermnet", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions"):
        "Mixed malignancies. Model B only (malignant). Excluded from Model A.",
}

MAPS = {"dermnet": DERMNET, "padufes20": PADUFES20, "fitzpatrick17k": FITZPATRICK17K, "scin": SCIN}


def main() -> int:
    with SKELETON.open(encoding="utf-8") as f:
        skel = list(csv.DictReader(f))

    out_rows = []
    unmapped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    a_counts: Counter = Counter()
    b_counts: Counter = Counter()
    excluded_imgs = 0

    for r in skel:
        src, raw, n = r["source"], r["raw_label"], int(r["n_images"])
        disease = MAPS[src].get(raw)
        note = NOTES.get((src, raw), "")
        if disease is None:
            if src == "scin":
                disease, malig, model_a, model_b = EXCLUDE, EXCLUDE, 0, 0
                note = "SCIN tail label, not mapped -> excluded from both models"
                excluded_imgs += n
            else:
                unmapped[src].append((raw, n))
                continue
        elif disease == PER_IMAGE:
            malig, model_a, model_b = PER_IMAGE, 1, 1
        elif disease == "malignant_lesion_nos":
            malig, model_a, model_b = MALIGNANCY[disease], 0, 1
            b_counts[malig] += n
        else:
            malig = MALIGNANCY[disease]
            model_a, model_b = 1, 1
            a_counts[disease] += n
            b_counts[malig] += n
        out_rows.append({
            "source": src, "raw_label": raw, "n_images": n, "disease": disease,
            "malignancy": malig, "model_a": model_a, "model_b": model_b, "note": note,
        })

    if unmapped:
        for src, items in unmapped.items():
            print(f"UNMAPPED in {src}:")
            for raw, n in items:
                print(f"   {n:5d}  {raw}")
        return 1

    # folder 11 contribution, from the per-image file
    f11 = ROOT / "data" / "dermnet_folder11_labels.csv"
    if f11.exists():
        last: dict[str, str] = {}
        with f11.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                last[r["path"]] = r["label"]
        c = Counter(last.values())
        a_counts["melanoma"] += c.get("melanoma", 0)
        a_counts["melanocytic_nevus"] += c.get("nevus", 0)
        b_counts["malignant"] += c.get("melanoma", 0)
        b_counts["benign"] += c.get("nevus", 0)
        excluded_imgs += c.get("unsure", 0)
        print(f"folder 11 per-image: melanoma={c.get('melanoma', 0)} nevus={c.get('nevus', 0)} unsure(excluded)={c.get('unsure', 0)}")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"label_map: {len(out_rows)} rows -> {OUT.relative_to(ROOT)}")

    print(f"\nModel A candidate classes (pre-dedup, pre-split). MIN_IMAGES={MIN_IMAGES}")
    keep = drop = 0
    for d, n in sorted(a_counts.items(), key=lambda kv: -kv[1]):
        flag = "" if n >= MIN_IMAGES else "  <- below floor, Stage 4 drops"
        keep += n >= MIN_IMAGES
        drop += n < MIN_IMAGES
        print(f"  {n:6d}  {d:36s} {MALIGNANCY[d]}{flag}")
    print(f"  classes kept={keep} dropped={drop}")
    print("\nModel B distribution (pre-dedup):")
    for k in ("malignant", "benign", "non-neoplastic"):
        print(f"  {k:15s} {b_counts[k]:6d}")
    print(f"  excluded images (SCIN tail + folder11 unsure): {excluded_imgs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
