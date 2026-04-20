# CAI DGA Project

Bu proje, güç transformatörlerinde kullanılan Dissolved Gas Analysis (DGA) verileri üzerinden arıza tipi sınıflandırması yapmak için geliştirilmiş bir makine öğrenmesi çalışmasıdır. Projede iki ayrı DGA veri seti ortak bir etiket uzayında birleştirilmiş, oran tabanlı öznitelikler üretilmiş, veri temizlenmiş ve farklı modellerle karşılaştırmalı deneyler yapılmıştır.

Projede ilk aşamadaki ikili sınıflandırma yaklaşımı daha sonra genişletilerek 7 sınıflı çok sınıflı bir yapıya taşınmıştır.

## Projenin Amacı

Bu projenin amacı:

- farklı kaynaklardan gelen DGA verilerini ortak formatta birleştirmek,
- gaz konsantrasyonlarından anlamlı oran özellikleri üretmek,
- arıza sınıflarını standardize etmek,
- birden fazla model ile performans karşılaştırması yapmak,
- hangi modelin bu problem için daha uygun olduğunu göstermek,
- confusion matrix ve sınıflandırma raporları ile sonuçları görselleştirmek.

## Problem Tanımı

DGA, transformatör yağında çözünmüş gazların miktarına bakarak iç arızaların tipini tahmin etmeye yarar. Bu projede hedef değişken, 7 sınıflı arıza yapısıdır:

- `PD`: Partial Discharge
- `D1`: Low Energy Discharge
- `D2`: High Energy Discharge
- `T1`: Thermal Fault < 300C
- `T2`: Thermal Fault 300C - 700C
- `T3`: Thermal Fault > 700C
- `NF`: No Fault / Normal

Kod tarafında bu sınıflar `LabelEncoder` ile şu şekilde sayısallaştırılmıştır:

- `D1 -> 0`
- `D2 -> 1`
- `NF -> 2`
- `PD -> 3`
- `T1 -> 4`
- `T2 -> 5`
- `T3 -> 6`

## Proje Yapısı

```text
CAI_DGA_Project/
├── data/
│   ├── raw/
│   │   ├── DGA-dataset.csv
│   │   ├── DGA_dataset2.csv
│   │   └── dga_merged_dataset.csv
│   └── processed/
│       └── dga_merged_processed.csv
├── results/
│   └── *_confusion_matrix.png
├── src/
│   ├── data_preprocessing.py
│   ├── DecisionTree.py
│   ├── RandomForrestModel.py
│   ├── SVM.py
│   ├── neural_network.py
│   └── visulization.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Kullanılan Teknolojiler

- Python 3.12
- pandas
- numpy
- scikit-learn
- keras
- tensorflow
- matplotlib
- openpyxl
- `uv`

## Veri Kaynakları

Projede iki farklı ham veri seti kullanılmıştır:

1. `data/raw/DGA-dataset.csv`
2. `data/raw/DGA_dataset2.csv`

İlk veri setindeki etiketler:

- `Partial discharge`
- `Spark discharge`
- `Arc discharge`
- `Low-temperature overheating`
- `Low/Middle-temperature overheating`
- `Middle-temperature overheating`
- `High-temperature overheating`

İkinci veri setindeki etiketler:

- `PD`
- `D1`
- `D2`
- `T1`
- `T2`
- `T3`
- `NF`

## Etiket Eşleme Mantığı

İlk veri setindeki orijinal etiketler ikinci veri setindeki standarda eşlenmiştir:

| 1. dataset etiketi | Ortak etiket |
|---|---|
| `Partial discharge` | `PD` |
| `Spark discharge` | `D1` |
| `Arc discharge` | `D2` |
| `Low-temperature overheating` | `T1` |
| `Low/Middle-temperature overheating` | `T2` |
| `Middle-temperature overheating` | `T2` |
| `High-temperature overheating` | `T3` |

Bu eşleme ile iki ayrı veri kaynağı tek bir ortak çok sınıflı problem haline getirilmiştir.

## Veri Ön İşleme Akışı

Ön işleme akışı [src/data_preprocessing.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/data_preprocessing.py:1) içinde tanımlanmıştır.

Sırasıyla yapılan işlemler:

1. `DGA-dataset.csv` ve `DGA_dataset2.csv` okunur.
2. İlk veri setindeki etiketler ortak sınıf isimlerine map edilir.
3. İkinci veri setindeki `Fail` sütunu temizlenir ve `Type` olarak yeniden adlandırılır.
4. İki veri seti birleştirilir.
5. Ortak geçerli sınıflar filtrelenir.
6. `LabelEncoder` ile `Fault_Type` sütunu üretilir.
7. Gaz oranı temelli yeni öznitelikler oluşturulur.
8. `inf`, `-inf` ve `NaN` değerler temizlenir.
9. IQR tabanlı scaling uygulanır.
10. Çıktılar hem ham birleşik veri hem de işlenmiş veri olarak kaydedilir.

### Üretilen Oran Özellikleri

- `R1 = CH4 / H2`
- `R2 = C2H2 / C2H4`
- `R4 = C2H6 / CH4`
- `R5 = C2H4 / C2H6`

### IQR Scaling

Projede kullanılan ölçekleme formülü:

```text
(deger - medyan) / (Q3 - Q1)
```

Bu yaklaşım, uç değerlerden klasik standart sapma tabanlı ölçeklemeye göre daha az etkilenir.

## Veri Seti Özeti

`data_preprocessing.py` çalıştırıldığında elde edilen özet:

- `Dataset 1 shape`: `201 x 8`
- `Dataset 2 shape`: `4150 x 8`
- `Merged raw shape`: `4351 x 8`
- `Processed shape`: `4226 x 13`

Temizlik sonrası sınıf dağılımı:

| Sınıf | Adet |
|---|---:|
| `D1` | 629 |
| `D2` | 808 |
| `NF` | 722 |
| `PD` | 335 |
| `T1` | 462 |
| `T2` | 361 |
| `T3` | 909 |

Üretilen dosyalar:

- Ham birleşik veri: [data/raw/dga_merged_dataset.csv](/home/emirfurkan/Desktop/CAI_DGA_Project/data/raw/dga_merged_dataset.csv:1)
- İşlenmiş veri: [data/processed/dga_merged_processed.csv](/home/emirfurkan/Desktop/CAI_DGA_Project/data/processed/dga_merged_processed.csv:1)

## Modelleme Yaklaşımı

Projede 4 farklı model denenmiştir:

1. Decision Tree
2. Random Forest
3. Support Vector Machine
4. Dense Neural Network (`neural_network.py` içinde, ancak tabular veri için fully connected yapı kullanılmıştır)

Ortak temel akış:

1. Veri okunur.
2. `X` ve `y` ayrılır.
3. `train_test_split(..., stratify=y)` ile eğitim-test ayrımı yapılır.
4. Model eğitilir.
5. Accuracy ve classification report hesaplanır.
6. Confusion matrix görseli oluşturulur.

## Modellerin Detaylı Analizi

### 1. Decision Tree

Dosya: [src/DecisionTree.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/DecisionTree.py:1)

Kullanılan teknikler:

- `DecisionTreeClassifier`
- `GridSearchCV`
- hiperparametre araması:
  - `max_depth`
  - `min_samples_split`
  - `criterion`
  - `splitter`

En iyi parametreler:

```text
{'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 2, 'splitter': 'best'}
```

Sonuçlar:

- Test accuracy: `0.8959810874704491`

Sınıf bazlı güçlü yönler:

- `NF`, `D2` ve `T3` sınıflarında oldukça güçlü
- `T2` sınıfında diğer sınıflara göre nispeten daha düşük performans

Yorum:

Decision Tree, tek ağaç olmasına rağmen çok güçlü sonuç üretmiştir. Özellikle uygun derinlik bulunduğunda veri üzerindeki ayrım gücü belirgin şekilde yükselmiştir.

### 2. Random Forest

Dosya: [src/RandomForrestModel.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/RandomForrestModel.py:1)

Kullanılan teknikler:

- `RandomForestClassifier`
- `n_estimators=200`
- `max_depth=10`
- `class_weight="balanced"`
- 5-fold cross validation

Sonuçlar:

- CV skorları: `0.8617, 0.9302, 0.9077, 0.8533, 0.7160`
- Ortalama CV accuracy: `0.8537605438751102`
- Test accuracy: `0.8936170212765957`

Yorum:

Random Forest, genelde en dengeli performansı veren modellerden biri olmuştur. Özellikle `D1`, `D2`, `NF`, `PD` ve `T3` sınıflarında güçlü sonuçlar üretmiştir. CV skorlarındaki son fold düşüşü, veri dağılımı veya fold bazlı örnek zorluğu nedeniyle değişkenlik olduğunu gösterir.

### 3. SVM

Dosya: [src/SVM.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/SVM.py:1)

Kullanılan teknikler:

- `SVC(kernel="rbf", C=100, gamma="auto", class_weight="balanced")`
- `RobustScaler`
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

Önemli not:

SVM modeli diğer modellerden farklı olarak `data/raw/dga_merged_dataset.csv` üzerinden çalıştırılmış ve model öncesinde `RobustScaler` uygulanmıştır.

Sonuçlar:

- CV skorları: `0.8003, 0.7945, 0.7931, 0.7917, 0.8017`
- Ortalama CV accuracy: `0.7962643678160919`
- Test accuracy: `0.7933409873708381`

Yorum:

SVM başlangıçta zayıf sonuç vermiş olsa da, kernel ve scaler ayarları iyileştirildikten sonra anlamlı biçimde toparlanmıştır. Yine de bu projede ağaç tabanlı yöntemler kadar güçlü değildir.

### 4. Neural Network (`neural_network.py`)

Dosya: [src/neural_network.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/neural_network.py:1)

Not:

Tabular veri için çok katmanlı bir dense neural network kullanılmıştır.

Kullanılan teknikler:

- `Dense(128) -> Dense(64) -> Output`
- `Dropout`
- `EarlyStopping`
- `Adam`
- `class_weight`

Sonuçlar:

- Test accuracy: `0.7529550827423168`

Gözlem:

- Accuracy genelde `0.70 - 0.80` bandında dalgalanmaktadır.
- Sonuçlar klasik ML modellerine göre daha oynaktır.

Yorum:

Neural network bu veri üzerinde kabul edilebilir sonuç verse de en iyi model değildir. Tabular DGA verisi için ağaç tabanlı yöntemler daha kararlı ve daha yüksek doğruluk üretmiştir.

## Güncel Performans Karşılaştırması

| Model | CV Mean Accuracy | Test Accuracy |
|---|---:|---:|
| Decision Tree | GridSearch kullanıldı | 0.8960 |
| Random Forest | 0.8538 | 0.8936 |
| SVM | 0.7963 | 0.7933 |
| Neural Network | Belirtilmedi | 0.7470 |

## Sınıf Bazlı Gözlemler

Çıktılara göre öne çıkan noktalar:

- `NF` sınıfı tüm güçlü modellerde en rahat öğrenilen sınıflardan biri.
- `T2` sınıfı çoğu model için nispeten daha zor.
- `D1` ve `T1` sınıflarında CNN daha kırılgan kalıyor.
- `T3` sınıfı özellikle SVM, Random Forest ve Decision Tree tarafından iyi öğreniliyor.
- Neural network sonuçları aynı kodla tekrar çalıştırıldığında daha değişken davranabiliyor.

## Confusion Matrix ve Görselleştirme

Ortak confusion matrix fonksiyonu [src/visulization.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/visulization.py:1) altında yer alır.

`display_cm(...)` fonksiyonu:

- confusion matrix üretir,
- ekranda gösterir,
- `results/` klasörüne `.png` olarak kaydeder.

Üretilen örnek dosyalar:

- `results/decisiontree_confusion_matrix.png`
- `results/randomforest_confusion_matrix.png`
- `results/svm_confusion_matrix.png`
- `results/neural_network_confusion_matrix.png`

## Çalıştırma Komutları

Ön işleme:

```bash
./.venv/bin/python src/data_preprocessing.py
```

Karar ağacı:

```bash
./.venv/bin/python src/DecisionTree.py
```

Random Forest:

```bash
./.venv/bin/python src/RandomForrestModel.py
```

SVM:

```bash
./.venv/bin/python src/SVM.py
```

Neural network:

```bash
./.venv/bin/python src/neural_network.py
```

## Genel Sonuç

Bu proje, iki farklı DGA veri kaynağını ortak 7 sınıflı arıza probleminde başarıyla birleştiren ve karşılaştırmalı modelleme yapan bir çalışma haline gelmiştir. Veri ön işleme tarafında etiket standardizasyonu, oran tabanlı özellik üretimi, temizlik ve IQR scaling uygulanmıştır. Deneyler sonucunda en güçlü modellerin `Decision Tree` ve `Random Forest` olduğu görülmüştür. `SVM` optimize edildikten sonra orta-üst düzey performans vermiş, neural network ise kabul edilebilir ama daha değişken sonuçlar üretmiştir.

## Güçlü Yönler

- İki farklı veri seti ortak bir etiket uzayında birleştirilmiş durumda.
- Problem alanına uygun oran bazlı öznitelikler kullanılıyor.
- Çok sınıflı problem net biçimde tanımlanmış.
- Dört farklı model ile karşılaştırma yapılmış.
- Confusion matrix çıktıları otomatik kaydediliyor.
- Sonuçlar hem accuracy hem de sınıf bazlı metriklerle raporlanıyor.

## Sınırlılıklar

- `SVM.py` diğer modellerden farklı olarak ham birleşik veri üzerinde çalışıyor; bu karşılaştırmayı kısmen etkileyebilir.
- Neural network sonuçları deterministic değil, tekrar çalıştırmalarda değişebiliyor.
- Tüm modeller için ortak, tek tip experiment pipeline henüz yok.

## Gelecek Geliştirmeler

- Tüm modelleri tek bir ortak veri pipeline’ına taşımak
- ortak train/test split kaydetmek
- seed sabitleyerek neural network sonuçlarını daha kararlı hale getirmek
- normalize confusion matrix eklemek
- feature importance ve SHAP benzeri açıklanabilirlik analizleri eklemek
