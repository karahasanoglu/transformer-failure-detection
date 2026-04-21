# Elektriksel Arıza Tespiti

Bu proje, akım ve gerilim ölçümlerini kullanarak elektriksel arıza türlerini makine öğrenmesi modelleri ile sınıflandırmayı amaçlamaktadır.

Veri setinde elektrik sistemine ait sinyaller giriş olarak kullanılmakta, modelin görevi ise sistemde oluşan arıza türünü tahmin etmektir.

## Projenin Amacı

Bu projenin temel amacı, bir elektrik güç sisteminde oluşan arıza durumlarını tespit etmek ve sınıflandırmaktır.

Projede kullanılan modeller aşağıdaki arıza sınıflarını tanımak üzere eğitilmiştir:

- `No Fault`
- `LG Fault`
- `LL Fault`
- `LLG Fault`
- `LLL Fault`
- `GLLL Fault`

## Veri Seti Bilgileri

Veri seti dosyası şu konumdadır:

- [data/classData.csv](/home/emirfurkan/Desktop/Electrical_Fault_Detection/data/classData.csv)

### Giriş Özellikleri

Veri setinde kullanılan giriş değişkenleri şunlardır:

- `Ia`
- `Ib`
- `Ic`
- `Va`
- `Vb`
- `Vc`

Bu değişkenler, üç fazlı bir elektrik sisteminden elde edilen akım ve gerilim değerlerini temsil eder.

### Çıkış Etiketleri

Çıkış sınıfları dört adet arıza gösterge sütunundan türetilmektedir:

- `G`
- `C`
- `B`
- `A`

Mevcut proje kodunda bu gösterge kombinasyonları aşağıdaki şekilde arıza sınıflarına dönüştürülmektedir:

- `[0, 0, 0, 0]` -> `No Fault`
- `[1, 0, 0, 1]` -> `LG Fault`
- `[0, 1, 1, 0]` -> `LL Fault`
- `[1, 0, 1, 1]` -> `LLG Fault`
- `[0, 1, 1, 1]` -> `LLL Fault`
- `[1, 1, 1, 1]` -> `GLLL Fault`

### Arıza Türlerinin Anlamı

- `No Fault`: Sistemde arıza yok, normal çalışma durumu
- `LG Fault`: Faz-toprak kısa devresi
- `LL Fault`: Faz-faz kısa devresi
- `LLG Fault`: İki faz-toprak arızası
- `LLL Fault`: Üç faz arızası
- `GLLL Fault`: Üç faz-toprak arızası

Not: Veri seti açıklamasında bazı örnek arıza kombinasyonları yer alsa da bu README dosyasındaki sınıf eşlemesi, projede fiilen kullanılan [src/preprocessing.py](/home/emirfurkan/Desktop/Electrical_Fault_Detection/src/preprocessing.py) dosyasına göre hazırlanmıştır.

## Proje Yapısı

```text
Electrical_Fault_Detection/
├── data/
│   └── classData.csv
├── results/
│   ├── DecisionTree_confusion_matrix.png
│   ├── Logistic Regression_confusion_matrix.png
│   ├── Neural Network_confusion_matrix.png
│   ├── RandomForest_confusion_matrix.png
│   └── Support Vector Machine_confusion_matrix.png
├── src/
│   ├── preprocessing.py
│   ├── visulazition.py
│   └── models/
│       ├── DecisionTree.py
│       ├── LogisticRegression.py
│       ├── NeuralNetwork.py
│       ├── RandomForrest.py
│       └── SVM.py
├── pyproject.toml
└── README.md
```

## Veri Ön İşleme

Ön işleme adımında şu işlemler yapılmaktadır:

- Veri setinin okunması
- Arıza gösterge kombinasyonlarının sınıf etiketlerine dönüştürülmesi
- Verinin eğitim ve test olarak ayrılması
- Giriş özelliklerinin standardize edilmesi
- Sınıf etiketlerinin sayısal olarak kodlanması

Yapay sinir ağı modeli için ayrıca etiketler kategorik formata dönüştürülmektedir.

## Kullanılan Modeller

Projede aşağıdaki makine öğrenmesi modelleri kullanılmıştır:

- Decision Tree
- Logistic Regression
- Neural Network
- Random Forest
- Support Vector Machine

## Model Sonuçları

Aşağıda yapılan deneylerden elde edilen sonuçlar yer almaktadır.

### 1. Decision Tree

- Accuracy: `0.8442`
- `LG Fault`, `LL Fault`, `LLG Fault` ve `No Fault` sınıflarında güçlü sonuç vermiştir
- `GLLL Fault` ve `LLL Fault` sınıflarında performans daha düşüktür

Sınıflandırma özeti:

```text
              precision    recall  f1-score   support

  GLLL Fault       0.48      0.54      0.51       227
    LG Fault       0.97      1.00      0.98       226
    LL Fault       1.00      0.96      0.98       201
   LLG Fault       1.00      0.98      0.99       227
   LLL Fault       0.47      0.42      0.45       219
    No Fault       0.99      1.00      1.00       473

    accuracy                           0.84      1573
   macro avg       0.82      0.82      0.82      1573
weighted avg       0.84      0.84      0.84      1573
```

Confusion matrix:

- [results/DecisionTree_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/DecisionTree_confusion_matrix.png)

### 2. Logistic Regression

- Accuracy: `0.3223`
- Bu model veri seti üzerinde oldukça zayıf performans göstermiştir
- Model çoğunlukla `No Fault` sınıfını tahmin etmiş ve diğer arıza türlerini ayırmakta başarısız kalmıştır

Sınıflandırma özeti:

```text
              precision    recall  f1-score   support

  GLLL Fault       0.29      0.15      0.20       227
    LG Fault       0.00      0.00      0.00       226
    LL Fault       0.00      0.00      0.00       201
   LLG Fault       0.00      0.00      0.00       227
   LLL Fault       0.00      0.00      0.00       219
    No Fault       0.33      1.00      0.49       473

    accuracy                           0.32      1573
   macro avg       0.10      0.19      0.12      1573
weighted avg       0.14      0.32      0.18      1573
```

Confusion matrix:

- [results/Logistic Regression_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/Logistic%20Regression_confusion_matrix.png)

### 3. Neural Network

- Test Accuracy: `0.856`
- Decision Tree ve Logistic Regression modelinden daha iyi sonuç vermiştir
- Doğrusal olmayan arıza sınıflarını ayırmada başarılıdır

Confusion matrix:

- [results/Neural Network_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/Neural%20Network_confusion_matrix.png)

### 4. Random Forest

- Accuracy: `0.8709`
- Verilen sonuçlar içinde en yüksek başarıyı elde etmiştir
- Sınıfların büyük kısmında oldukça güçlü performans göstermiştir
- `GLLL Fault` ile `LLL Fault` sınıfları arasında hâlâ belirli bir karışıklık görülmektedir

Sınıflandırma özeti:

```text
              precision    recall  f1-score   support

  GLLL Fault       0.56      0.55      0.55       227
    LG Fault       1.00      1.00      1.00       226
    LL Fault       1.00      1.00      1.00       201
   LLG Fault       0.99      0.99      0.99       227
   LLL Fault       0.55      0.55      0.55       219
    No Fault       1.00      1.00      1.00       473

    accuracy                           0.87      1573
   macro avg       0.85      0.85      0.85      1573
weighted avg       0.87      0.87      0.87      1573
```

Confusion matrix:

- [results/RandomForest_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/RandomForest_confusion_matrix.png)

### 5. Support Vector Machine

- Accuracy: `0.8531`
- Genel olarak güçlü bir performans göstermiştir
- Özellikle `LG Fault`, `LL Fault`, `LLG Fault` ve `No Fault` sınıflarında başarılıdır
- `GLLL Fault` ile `LLL Fault` sınıflarını ayırmada orta seviyede zorlanmıştır

En iyi hiperparametreler:

```text
{'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}
```

Sınıflandırma özeti:

```text
              precision    recall  f1-score   support

  GLLL Fault       0.51      0.34      0.41       227
    LG Fault       0.98      1.00      0.99       226
    LL Fault       1.00      1.00      1.00       201
   LLG Fault       1.00      0.97      0.99       227
   LLL Fault       0.49      0.66      0.56       219
    No Fault       1.00      1.00      1.00       473

    accuracy                           0.85      1573
   macro avg       0.83      0.83      0.82      1573
weighted avg       0.85      0.85      0.85      1573
```

Confusion matrix:

- [results/Support Vector Machine_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/Support%20Vector%20Machine_confusion_matrix.png)

## Modellerin Karşılaştırılması

Elde edilen sonuçlara göre:

1. `Random Forest`, `%87.09` doğruluk oranı ile en başarılı model olmuştur.
2. `Neural Network` ve `SVM` modelleri de güçlü ve rekabetçi sonuçlar vermiştir.
3. `Decision Tree`, iyi bir performans göstermesine rağmen `Random Forest` ve `SVM` modellerinin bir miktar gerisinde kalmıştır.
4. `Logistic Regression`, bu veri seti için mevcut haliyle uygun görünmemektedir.

## Projeyi Çalıştırma

Model dosyalarını proje ana dizininden çalıştırabilirsiniz.

Örnek komutlar:

```bash
python3 src/models/DecisionTree.py
python3 src/models/RandomForrest.py
python3 src/models/LogisticRegression.py
python3 src/models/SVM.py
python3 src/models/NeuralNetwork.py
```

Eğer `uv` kullanıyorsanız şu şekilde de çalıştırabilirsiniz:

```bash
uv run python src/models/DecisionTree.py
```

## Bağımlılıklar

Projede kullanılan temel kütüphaneler şunlardır:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

Bağımlılık listesi şu dosyada tanımlıdır:

- [pyproject.toml](/home/emirfurkan/Desktop/Electrical_Fault_Detection/pyproject.toml)

## Sonuç

Bu proje, akım ve gerilim ölçümleri kullanılarak elektriksel arızaların makine öğrenmesi ile başarılı şekilde sınıflandırılabileceğini göstermektedir.

Denenen modeller arasında en iyi sonuç `Random Forest` ile elde edilmiştir. `Neural Network` ve `SVM` modelleri de güçlü sonuçlar vermiştir. Veri setindeki en belirgin sınıflandırma zorluğu, `GLLL Fault` ile `LLL Fault` sınıflarının birbirine daha yakın özellik göstermesinden kaynaklanmaktadır.
