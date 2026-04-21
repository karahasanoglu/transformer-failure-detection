# Distributed Transformer Monitoring

Bu proje, dağıtım transformatörlerinden IoT cihazları aracılığıyla toplanan ölçüm verilerini kullanarak olası arıza durumlarını sınıflandırmayı amaçlamaktadır. Elektrik şebekesinin en kritik bileşenlerinden biri olan transformatörler, genel olarak güvenilir sistemler olsa da hem içsel hem de dışsal etkenlerden dolayı arızaya açık yapılardır. Özellikle mekanik arızalar ve dielektrik arızalar, ciddi ve yıkıcı sonuçlara yol açabilecek başlıca riskler arasında yer alır.

Projede kullanılan veri seti, 25 Haziran 2019 ile 14 Nisan 2020 tarihleri arasında, 15 dakikalık aralıklarla IoT cihazları üzerinden toplanmıştır. Amaç; sıcaklık, yağ seviyesi, akım ve gerilim gibi operasyonel parametrelerden yararlanarak transformatörün alarm veya arıza davranışını erken aşamada tespit edebilen makine öğrenmesi modelleri geliştirmektir.

## Projenin Amacı

Bu projenin temel amacı, transformatör izleme verileri üzerinden `MOG_A` değişkenini tahmin etmektir. `MOG_A`, manyetik yağ göstergesi alarmını temsil eder ve transformatörde anormal ya da riskli bir durumun göstergesi olarak değerlendirilebilir.

Bu kapsamda proje aşağıdaki hedeflere odaklanır:

- IoT verilerinden anlamlı bir sınıflandırma problemi oluşturmak
- Farklı makine öğrenmesi modellerinin performansını karşılaştırmak
- Hangi modelin daha güvenilir arıza/alarm tespiti yaptığını görmek
- Confusion matrix ve sınıflandırma raporları üzerinden modelleri yorumlamak
- Gelecekte kestirimci bakım uygulamalarına temel oluşturmak

## Veri Seti İçeriği

Veri seti iki ana dosyadan oluşmaktadır:

- `data/Overview.csv`
- `data/CurrentVoltage.csv`

Bu iki dosya `DeviceTimeStamp` alanı üzerinden birleştirilmektedir.

### CurrentVoltage Parametreleri

- `VL1`: Faz 1 gerilimi
- `VL2`: Faz 2 gerilimi
- `VL3`: Faz 3 gerilimi
- `IL1`: Hat/Faz 1 akımı
- `IL2`: Hat/Faz 2 akımı
- `IL3`: Hat/Faz 3 akımı
- `VL12`: Faz 1-2 arası gerilim
- `VL23`: Faz 2-3 arası gerilim
- `VL31`: Faz 3-1 arası gerilim
- `INUT`: Nötr akımı

### Overview Parametreleri

- `OTI`: Oil Temperature Indicator
- `WTI`: Winding Temperature Indicator
- `ATI`: Ambient Temperature Indicator
- `OLI`: Oil Level Indicator
- `OTI_A`: Oil Temperature Indicator Alarm
- `OTI_T`: Oil Temperature Indicator Trip
- `MOG_A`: Magnetic Oil Gauge Indicator Alarm

## Kullanılan Yaklaşım

Projede veri ön işleme ve modelleme akışı oldukça nettir:

1. `Overview.csv` ve `CurrentVoltage.csv` dosyaları okunur.
2. `DeviceTimeStamp` sütunu tarih tipine çevrilir.
3. İki veri seti zaman damgasına göre birleştirilir.
4. Hedef değişken olarak `MOG_A` seçilir.
5. `DeviceTimeStamp` ve `MOG_A` dışındaki tüm sütunlar özellik olarak kullanılır.
6. Veri, eğitim ve test kümelerine `%80 - %20` oranında ayrılır.
7. Özellikler `MinMaxScaler` ile normalize edilir.
8. Farklı sınıflandırma modelleri eğitilip test edilir.
9. Her model için sınıflandırma raporu ve confusion matrix üretilir.

Bu akış [src/preprocessing.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/preprocessing.py:1) ve [src/visualization.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/visualization.py:1) dosyalarında tanımlanmıştır.

## Proje Yapısı

```text
Distributed_Transformer_Monitoring/
├── data/
│   ├── CurrentVoltage.csv
│   └── Overview.csv
├── results/
│   ├── GaussianNB_confusion_matrix.png
│   ├── knn_confusion_matrix.png
│   ├── Logistic Regression_confusion_matrix.png
│   ├── RandomForest_confusion_matrix.png
│   └── XGP Classifier_confusion_matrix.png
├── src/
│   ├── preprocessing.py
│   ├── visualization.py
│   └── models/
│       ├── GaussianNB.py
│       ├── knn.py
│       ├── LogisticRegression.py
│       ├── RandomForrest.py
│       └── XGP_Classifier.py
├── pyproject.toml
└── README.md
```

## Kullanılan Modeller

Projede beş farklı sınıflandırma modeli kullanılmıştır.

### 1. K-Nearest Neighbors (KNN)

KNN, yeni bir örneğin sınıfını, kendisine en yakın komşu örneklerin sınıflarına bakarak belirleyen uzaklık tabanlı bir algoritmadır. Özellikle benzer örneklerin aynı sınıfta toplandığı veri setlerinde başarılı olabilir. Parametre seçimine duyarlıdır ve veri ölçekleme bu model için oldukça önemlidir.

Kod dosyası: [src/models/knn.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/knn.py:1)

### 2. Gaussian Naive Bayes

Gaussian Naive Bayes, özelliklerin birbirinden bağımsız olduğu varsayımıyla çalışan olasılıksal bir sınıflandırma modelidir. Sürekli sayısal veriler için her özelliğin Gaussian dağılıma sahip olduğunu kabul eder. Hızlı çalışması ve düşük hesaplama maliyeti ile güçlü bir başlangıç modelidir.

Kod dosyası: [src/models/GaussianNB.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/GaussianNB.py:1)

### 3. Logistic Regression

Logistic Regression, iki sınıflı sınıflandırma problemlerinde sık kullanılan doğrusal bir modeldir. Tahmin sonucunu olasılık olarak üretir ve yorumlanabilirliği yüksektir. Özellikle baseline model olarak çok değerlidir.

Kod dosyası: [src/models/LogisticRegression.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/LogisticRegression.py:1)

### 4. Random Forest

Random Forest, çok sayıda karar ağacının birlikte çalıştığı bir ensemble öğrenme yöntemidir. Tek bir karar ağacına göre daha dengeli ve genellenebilir sonuçlar verir. Tabular veri üzerinde oldukça güçlü ve güvenilir bir modeldir.

Kod dosyası: [src/models/RandomForrest.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/RandomForrest.py:1)

### 5. XGBoost Classifier

XGBoost, gradient boosting yaklaşımını verimli ve güçlü bir şekilde uygulayan gelişmiş bir ensemble modelidir. Özellikle yapılandırılmış tablo verilerinde yüksek performans göstermesiyle öne çıkar. Bu projede en güçlü modellerden biri olarak değerlendirilmiştir.

Kod dosyası: [src/models/XGP_Classifier.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/XGP_Classifier.py:1)

## Model Çıktıları

Aşağıda proje kapsamında elde edilen model sonuçları yer almaktadır.

### KNN Sonuçları

```text
Acc score : 0.9538236012704617
              precision    recall  f1-score   support

         0.0       0.97      0.98      0.97      3655
         1.0       0.81      0.74      0.77       438

    accuracy                           0.95      4093
   macro avg       0.89      0.86      0.87      4093
weighted avg       0.95      0.95      0.95      4093
```

Değerlendirme: KNN modeli genel doğrulukta iyi bir sonuç vermiştir. Ancak `1.0` sınıfında recall değerinin `0.74` olması, bazı alarm durumlarının kaçırıldığını göstermektedir.

### Gaussian Naive Bayes Sonuçları

```text
              precision    recall  f1-score   support

         0.0       1.00      0.83      0.91      3655
         1.0       0.41      1.00      0.58       438

    accuracy                           0.85      4093
   macro avg       0.71      0.91      0.75      4093
weighted avg       0.94      0.85      0.87      4093

Accuracy Score: 0.8480332274615197
```

Değerlendirme: GaussianNB, `1.0` sınıfı için recall değerinde çok güçlü görünmektedir. Yani alarm durumlarını yakalama konusunda oldukça hassastır. Ancak precision değerinin düşük olması, yanlış alarm üretme oranının yüksek olduğunu göstermektedir.

### Logistic Regression Sonuçları

```text
              precision    recall  f1-score   support

         0.0       0.94      1.00      0.97      3655
         1.0       0.94      0.44      0.60       438

    accuracy                           0.94      4093
   macro avg       0.94      0.72      0.78      4093
weighted avg       0.94      0.94      0.93      4093

Accuracy Score : 0.9369655509406304
```

Değerlendirme: Logistic Regression yüksek precision sunmasına rağmen `1.0` sınıfındaki recall değeri düşüktür. Bu durum, modelin alarm durumlarını tahmin ederken temkinli davrandığını fakat bazı kritik örnekleri kaçırdığını düşündürmektedir.

### Random Forest Sonuçları

```text
              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99      3655
         1.0       0.95      0.94      0.94       438

    accuracy                           0.99      4093
   macro avg       0.97      0.97      0.97      4093
weighted avg       0.99      0.99      0.99      4093

ACC SCORE
0.9882726606401173
```

Değerlendirme: Random Forest, hem genel doğruluk hem de sınıf bazlı denge açısından en başarılı modellerden biridir. Özellikle `1.0` sınıfında hem precision hem recall değerlerinin yüksek olması, gerçek kullanım senaryoları açısından oldukça değerlidir.

### XGBoost Sonuçları

```text
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99      3655
         1.0       0.96      0.93      0.94       438

    accuracy                           0.99      4093
   macro avg       0.98      0.96      0.97      4093
weighted avg       0.99      0.99      0.99      4093

Acc score  0.9882726606401173
```

Değerlendirme: XGBoost modeli de Random Forest ile çok yakın, oldukça güçlü bir performans sergilemiştir. Sınıf dengesini koruyarak yüksek başarı sağlaması, bu modeli proje için güçlü adaylardan biri haline getirmektedir.

## Modellerin Genel Karşılaştırması

Sonuçlar birlikte değerlendirildiğinde:

- `Random Forest` ve `XGBoost`, en yüksek doğruluk ve en dengeli sınıf performansını sunmuştur.
- `KNN`, iyi bir başarı göstermiş ancak azınlık sınıfında daha fazla hata yapmıştır.
- `Logistic Regression`, açıklanabilir bir baseline model olsa da alarm sınıfını yakalamada sınırlı kalmıştır.
- `GaussianNB`, alarm durumlarını kaçırmama açısından dikkat çekici bir recall değeri sunmuş, fakat çok fazla yanlış pozitif üretmiştir.

Eğer amaç genel başarı ve dengeli tahmin ise `Random Forest` ve `XGBoost` en uygun modellerdir. Eğer amaç olası riskli durumları kaçırmamaya öncelik vermekse, `GaussianNB` gibi yüksek recall üreten modeller de belirli senaryolarda değerlendirilebilir.

## Confusion Matrix Çıktıları

Her model için confusion matrix görselleri `results/` klasörüne kaydedilmektedir:

- `results/knn_confusion_matrix.png`
- `results/GaussianNB_confusion_matrix.png`
- `results/Logistic Regression_confusion_matrix.png`
- `results/RandomForest_confusion_matrix.png`
- `results/XGP Classifier_confusion_matrix.png`

Bu görseller [src/visualization.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/visualization.py:1) içindeki `plot_cm()` fonksiyonu ile oluşturulmaktadır.

## Projede Kullanılan Temel Dosyalar

- [src/preprocessing.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/preprocessing.py:1): Veri okuma, birleştirme, eğitim-test ayırma ve ölçekleme işlemleri
- [src/visualization.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/visualization.py:1): Confusion matrix oluşturma ve kaydetme işlemleri
- [src/models/knn.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/knn.py:1): KNN modeli
- [src/models/GaussianNB.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/GaussianNB.py:1): Gaussian Naive Bayes modeli
- [src/models/LogisticRegression.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/LogisticRegression.py:1): Logistic Regression modeli
- [src/models/RandomForrest.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/RandomForrest.py:1): Random Forest modeli
- [src/models/XGP_Classifier.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/XGP_Classifier.py:1): XGBoost sınıflandırıcısı

## Kurulum

Bağımlılıklar `pyproject.toml` dosyasında tanımlıdır. Ortamı hazırlamak için aşağıdaki yöntemlerden biri kullanılabilir:

```bash
uv sync
```

veya

```bash
pip install -e .
```

## Modelleri Çalıştırma

Her model ayrı bir Python dosyası olarak çalıştırılabilir:

```bash
python3 src/models/knn.py
python3 src/models/GaussianNB.py
python3 src/models/LogisticRegression.py
python3 src/models/RandomForrest.py
python3 src/models/XGP_Classifier.py
```

## Geliştirme Önerileri

Bu proje ileride aşağıdaki başlıklarda geliştirilebilir:

- Hyperparameter optimization eklemek
- Cross-validation ile daha sağlam model değerlendirmesi yapmak
- Sınıf dengesizliği için `class_weight`, SMOTE veya benzeri yöntemler denemek
- ROC-AUC, PR-AUC ve F1-score odaklı ek analizler yapmak
- Tüm modellerin sonuçlarını tek tabloda toplayan bir karşılaştırma scripti yazmak
- Derin öğrenme tarafında `MLPClassifier` veya `TensorFlow/Keras` ile yeni modeller eklemek
- Özellik önem analizi yaparak transformatör davranışını en çok etkileyen ölçümleri belirlemek

## Sonuç

Bu proje, transformatör izleme verileri üzerinden arıza/alarm tahmini yapılabileceğini göstermektedir. Özellikle ensemble tabanlı yöntemler olan Random Forest ve XGBoost, hem yüksek doğruluk hem de dengeli sınıf performansı ile öne çıkmıştır. Bu çalışma, kestirimci bakım, erken uyarı sistemleri ve enerji altyapısında güvenilirliğin artırılması için faydalı bir temel oluşturmaktadır.
