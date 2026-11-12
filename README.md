# Derin Öğrenme ile Göğüs Röntgeni Görüntülerinden Zatürre Tespiti

Göğüs röntgenlerini **normal** ve **zatürre** olarak sınıflandıran, kararını **Grad-CAM** ısı haritasıyla açıklayan ve **Gradio** arayüzüyle kullanılabilen bir derin öğrenme projesi.

> **Uyarı:** Bu sistem tıbbi teşhis aracı değildir. Yalnızca eğitim amaçlıdır ve tek bir veri setiyle eğitilmiştir.

## Ekip

| Kişi | GitHub | Sorumluluk |
|---|---|---|
| Beyza | [@ferhanbeyza](https://github.com/ferhanbeyza) | Veri hazırlığı, model eğitimi, ince ayar |
| İklim | [@iklimcelebi](https://github.com/iklimcelebi) | Değerlendirme, Grad-CAM, arayüz |

## Veri Seti

Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Yaklaşık 5.900 görüntü, yaklaşık 1,2 GB
- Sınıflar: NORMAL, PNEUMONIA
- Veri ve model dosyaları bu depoya **yüklenmez**, Google Drive'da tutulur.

> Kesin görüntü sayıları veri indirildikten sonra buraya yazılacak.

## Yöntem

1. Basit CNN (baseline)
2. Transfer öğrenme: ResNet50, DenseNet121, EfficientNet-B0
3. Ablasyon: augmentation ve sınıf ağırlığı açık / kapalı, öğrenme oranı
4. Grad-CAM ile açıklanabilirlik ve hata analizi
5. Gradio demo

## Klasör Yapısı

```
pneumonia-detection-cxr/
├── notebooks/
│   ├── 01_veri_analizi.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_transfer.ipynb
│   ├── 04_gradcam.ipynb
│   └── 05_demo.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Nasıl Çalıştırılır

1. Depoyu klonlayın veya notebook'ları Google Colab'a yükleyin.
2. Kaggle hesabınızdan API anahtarı (`kaggle.json`) alın.
3. Notebook'ları sırayla çalıştırın: `01` → `02` → `03` → `04` → `05`.
4. Gerekli paketler: `pip install -r requirements.txt`
5. Demo için `05_demo.ipynb` çalıştırılır; Gradio paylaşılabilir bir link üretir.

Tüm notebook'larda rastgelelik tohumu (seed) sabittir, sonuçlar tekrarlanabilir.

## Sonuçlar

_Final sonuçlar ve model karşılaştırma tablosu proje sonunda eklenecek._

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Eğitim süresi |
|---|---|---|---|---|---|---|
| Baseline CNN | | | | | | |
| ResNet50 | | | | | | |
| DenseNet121 | | | | | | |
| EfficientNet-B0 | | | | | | |

## Sınırlılıklar

- Sistem tıbbi teşhis aracı değildir.
- Tek bir veri setiyle eğitilmiştir; başka hastanelerdeki görüntülerde başarısı düşebilir.
