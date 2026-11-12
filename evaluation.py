"""
Model değerlendirme fonksiyonları (İklim)

Bir model ve test DataLoader'ı verilince şunları üretir:
accuracy, precision, recall, F1, ROC-AUC, karışıklık matrisi ve ROC eğrisi.

Varsayımlar:
- Sınıf sırası ImageFolder'ın alfabetik sırasıdır: 0 = NORMAL, 1 = PNEUMONIA
- Pozitif sınıf = PNEUMONIA (precision / recall / F1 bu sınıf için hesaplanır)
- Model çıktısı 2 sınıflı logit'tir: (batch, 2)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


@torch.no_grad()
def predict_all(model, loader, device):
    """Tüm loader üzerinde gerçek etiketleri ve zatürre olasılıklarını döndürür."""
    model.eval()
    model.to(device)
    y_true, y_prob = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1]  # PNEUMONIA olasılığı
        y_true.extend(labels.numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_prob)


def evaluate_model(model, loader, device, model_name="model",
                   threshold=0.5, save_dir="results"):
    """
    Tek çağrıda tüm metrikleri hesaplar, grafikleri kaydeder.
    Dönüş: metrik sözlüğü (deney günlüğü / karşılaştırma tablosu için).
    """
    os.makedirs(save_dir, exist_ok=True)

    y_true, y_prob = predict_all(model, loader, device)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    # Karışıklık matrisi
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Karışıklık matrisi - {model_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), dpi=150)
    plt.show()

    # ROC eğrisi
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Yanlış pozitif oranı")
    ax.set_ylabel("Doğru pozitif oranı (recall)")
    ax.set_title(f"ROC eğrisi - {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{model_name}_roc.png"), dpi=150)
    plt.show()

    # Sonuçları JSON olarak da kaydet
    with open(os.path.join(save_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{model_name}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:10s}: {v:.4f}")
    return metrics


def compare_models(metrics_list, save_path="results/comparison.csv"):
    """evaluate_model çıktılarından tek karşılaştırma tablosu üretir."""
    import pandas as pd
    df = pd.DataFrame(metrics_list).set_index("model").round(4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    return df


# ---------------------------------------------------------------
# Kullanım örneği (Colab'da):
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = ...  # Beyza'nın baseline modeli, ağırlıkları yüklenmiş
# m = evaluate_model(model, test_loader, device, model_name="baseline_cnn")
# table = compare_models([m])
# ---------------------------------------------------------------
