# Transformer Failure Detection

Bu depo, transformatör arıza tespiti, durum izleme, kestirimci bakım, arıza sınıflandırması ve kalan faydalı ömür tahmini gibi birbiriyle ilişkili problemleri ele alan çok kollu bir araştırma çatısıdır. `main` branch, aktif deney kodlarını tek başına barındıran bir çalışma dalı olmaktan ziyade, farklı branch'lerde geliştirilen projelerin üst düzey haritasını sunan bir giriş noktası olarak kurgulanmıştır.

Depoda yer alan çalışmalar ortak olarak enerji sistemlerinde güvenilirlik artışı, erken uyarı, bakım önceliklendirmesi ve veri temelli karar desteği hedeflerine odaklanmaktadır. Ancak her branch farklı bir problem tanımı, farklı veri yapısı ve farklı modelleme yaklaşımı benimsemektedir. Bu nedenle bu README, branch bazlı içeriklerin akademik ve sistematik bir özetini sunmak amacıyla hazırlanmıştır.

## Repository Yapısı ve Erişim Mantığı

Bu depodaki ana çalışmalar aşağıdaki branch'lerde yer almaktadır:

- [`DGA_Analysis`](https://github.com/karahasanoglu/transformer-failure-detection/tree/DGA_Analysis)
- [`EFRI`](https://github.com/karahasanoglu/transformer-failure-detection/tree/EFRI)
- [`Transformer_Monitoring`](https://github.com/karahasanoglu/transformer-failure-detection/tree/Transformer_Monitoring)
- [`predictive_maintenance`](https://github.com/karahasanoglu/transformer-failure-detection/tree/predictive_maintenance)
- [`rul_and_failureType_prediction`](https://github.com/karahasanoglu/transformer-failure-detection/tree/rul_and_failureType_prediction)

Her branch, ayrı bir araştırma problemi ve deneysel iş akışı temsil etmektedir. Aşağıda bu branch'lerin kapsamı ayrıntılı biçimde açıklanmıştır.

## Branch Bazlı Akademik Analiz

### 1. `DGA_Analysis`

Branch bağlantısı:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/DGA_Analysis

#### Çalışmanın Amacı

Bu çalışma, güç transformatörlerinde Dissolved Gas Analysis (DGA) verilerini kullanarak çok sınıflı arıza tipi sınıflandırması yapmayı amaçlamaktadır. Ana hedef, farklı kaynaklardan gelen DGA veri setlerini ortak bir etiket uzayında birleştirerek, gaz konsantrasyonları ve oran tabanlı öznitelikler üzerinden arıza teşhisi gerçekleştirmektir.

#### Problem Tanımı

Problem, 7 sınıflı bir arıza teşhisi problemidir. Hedef sınıflar şunlardır:

- `PD`: Partial Discharge
- `D1`: Low Energy Discharge
- `D2`: High Energy Discharge
- `T1`: Thermal Fault < 300C
- `T2`: Thermal Fault 300C - 700C
- `T3`: Thermal Fault > 700C
- `NF`: No Fault / Normal

#### Kullanılan Veri Setleri

Bu branch iki ayrı DGA veri setini birlikte kullanmaktadır:

- `data/raw/DGA-dataset.csv`
- `data/raw/DGA_dataset2.csv`

İki veri seti ön işleme aşamasında birleştirilmekte ve ortak etiket standardına dönüştürülmektedir. Kod düzeyinde bu dönüşüm `src/data_preprocessing.py` içinde tanımlanmıştır.

#### Kullanılan Girdi Değişkenleri

Temel gaz değişkenleri şunlardır:

- `H2`
- `CH4`
- `C2H6`
- `C2H4`
- `C2H2`

Bunlara ek olarak alan bilgisini yansıtan oran temelli öznitelikler üretilmektedir:

- `R1 = CH4 / H2`
- `R2 = C2H2 / C2H4`
- `R4 = C2H6 / CH4`
- `R5 = C2H4 / C2H6`

Ölçekleme tarafında klasik standart sapma tabanlı yaklaşım yerine IQR tabanlı robust bir dönüşüm kullanılmaktadır.

#### Kullanılan Modeller

Bu branch'te karşılaştırmalı olarak şu modeller kullanılmıştır:

- Decision Tree
- Random Forest
- Support Vector Machine
- Dense Neural Network

Kod doğrulamasına göre kullanılan başlıca model sınıfları:

- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `SVC`
- `keras.Sequential`

#### Elde Edilen Bulgular

Mevcut branch README ve kod çıktıları birlikte değerlendirildiğinde:

- Decision Tree yaklaşık `0.896` test doğruluğu ile en güçlü adaylardan biridir.
- Random Forest yaklaşık `0.894` test doğruluğu ile dengeli ve güçlü sonuçlar üretmektedir.
- SVM yaklaşık `0.793` test doğruluğunda kalmaktadır.
- Dense Neural Network yaklaşık `0.75` bandında sonuç vermektedir.

#### Akademik Değerlendirme

Bu branch, DGA tabanlı arıza teşhisinde öznitelik mühendisliği ile klasik makine öğrenmesi yöntemlerinin hâlen çok güçlü olduğunu göstermektedir. Özellikle ağaç tabanlı modellerin, sınırlı boyutlu ve açıklanabilir tablo verisinde derin öğrenmeye göre daha kararlı sonuçlar verdiği görülmektedir.

### 2. `EFRI`

Branch bağlantısı:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/EFRI

#### Çalışmanın Amacı

Bu çalışma, elektrik güç sistemlerinde oluşan arıza türlerini akım ve gerilim ölçümleri üzerinden sınıflandırmayı hedeflemektedir. Temel amaç, üç fazlı elektriksel sinyaller yardımıyla farklı kısa devre ve arıza durumlarını ayırt edebilen bir sınıflandırma altyapısı geliştirmektir.

#### Problem Tanımı

Bu branch bir çok sınıflı elektriksel arıza sınıflandırma problemidir. Hedef sınıflar:

- `No Fault`
- `LG Fault`
- `LL Fault`
- `LLG Fault`
- `LLL Fault`
- `GLLL Fault`

Sınıf etiketleri, veri setindeki dört arıza gösterge sütununun kombinasyonlarından türetilmektedir:

- `G`
- `C`
- `B`
- `A`

#### Kullanılan Veri Seti

Bu branch tek bir ana veri seti kullanmaktadır:

- `data/classData.csv`

#### Kullanılan Girdi Değişkenleri

Kod doğrulamasına göre model girdileri, veri setindeki aşağıdaki altı sürekli ölçümden oluşmaktadır:

- `Ia`
- `Ib`
- `Ic`
- `Va`
- `Vb`
- `Vc`

Ön işleme adımında:

- `StandardScaler` ile ölçekleme yapılmakta,
- `LabelEncoder` ile hedef sınıflar sayısallaştırılmakta,
- yapay sinir ağı için etiketler ayrıca kategorik forma dönüştürülmektedir.

#### Kullanılan Modeller

Bu branch'te aşağıdaki modeller denenmiştir:

- Decision Tree
- Logistic Regression
- Neural Network
- Random Forest
- Support Vector Machine

Kod tarafında kullanılan başlıca model sınıfları:

- `DecisionTreeClassifier`
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `tensorflow.keras.Sequential`

#### Elde Edilen Bulgular

README'de raporlanan temel sonuçlar şunlardır:

- Random Forest: yaklaşık `0.8709` accuracy
- Neural Network: yaklaşık `0.856` test accuracy
- SVM: yaklaşık `0.8531` accuracy
- Decision Tree: yaklaşık `0.8442` accuracy
- Logistic Regression: yaklaşık `0.3223` accuracy

#### Akademik Değerlendirme

Bu branch, doğrusal olmayan sınıf ayrımlarının baskın olduğu elektriksel arıza problemlerinde ensemble yöntemlerinin ve çekirdek tabanlı yöntemlerin daha uygun olduğunu göstermektedir. Logistic Regression'ın zayıf kalması, problem uzayının doğrusal ayrılabilirlikten uzak olduğunu düşündürmektedir.

### 3. `Transformer_Monitoring`

Branch bağlantısı:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/Transformer_Monitoring

#### Çalışmanın Amacı

Bu çalışma, IoT tabanlı dağıtım transformatörü izleme verileri üzerinden alarm veya anomali benzeri durumların tahmin edilmesini hedeflemektedir. Hedef değişken `MOG_A` olup, manyetik yağ göstergesi alarmını temsil etmektedir.

#### Problem Tanımı

Bu branch ikili sınıflandırma problemine odaklanmaktadır. Amaç, operasyonel ölçümlerden hareketle `MOG_A` alarmını önceden tahmin etmektir.

#### Kullanılan Veri Setleri

Bu branch iki zaman damgalı veri dosyasını birleştirmektedir:

- `data/Overview.csv`
- `data/CurrentVoltage.csv`

Kod düzeyinde bu iki dosya `DeviceTimeStamp` sütunu üzerinden birleştirilmektedir.

#### Kullanılan Girdi Değişkenleri

`CurrentVoltage.csv` içinden kullanılan başlıca değişkenler:

- `VL1`
- `VL2`
- `VL3`
- `IL1`
- `IL2`
- `IL3`
- `VL12`
- `VL23`
- `VL31`
- `INUT`

`Overview.csv` içinden kullanılan başlıca değişkenler:

- `OTI`
- `WTI`
- `ATI`
- `OLI`
- `OTI_A`
- `OTI_T`

Hedef değişken:

- `MOG_A`

Ön işleme tarafında `MinMaxScaler` kullanılmaktadır.

#### Kullanılan Modeller

Bu branch'te şu modeller kullanılmıştır:

- K-Nearest Neighbors
- Gaussian Naive Bayes
- Logistic Regression
- Random Forest
- XGBoost Classifier

Kod doğrulamasına göre kullanılan model sınıfları:

- `KNeighborsClassifier`
- `GaussianNB`
- `LogisticRegression`
- `RandomForestClassifier`
- `xgboost.XGBClassifier`

#### Elde Edilen Bulgular

README'de raporlanan sonuçlara göre:

- Random Forest: yaklaşık `0.9883` accuracy
- XGBoost: yaklaşık `0.9883` accuracy
- KNN: yaklaşık `0.9538` accuracy
- Logistic Regression: yaklaşık `0.9370` accuracy
- GaussianNB: yaklaşık `0.8480` accuracy

#### Akademik Değerlendirme

Bu branch, IoT tabanlı operasyonel izleme verilerinde ensemble yöntemlerinin çok güçlü sonuçlar verdiğini göstermektedir. Özellikle Random Forest ve XGBoost'un birlikte öne çıkması, bu problemin doğrusal olmayan fakat tablo yapısında düzenli örüntüler içerdiğini düşündürmektedir.

### 4. `predictive_maintenance`

Branch bağlantısı:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/predictive_maintenance

#### Çalışmanın Amacı

Bu çalışma, dağıtım transformatörlerinin arıza riskini yıllık bazda öngörmeyi amaçlayan bir kestirimci bakım prototipidir. Çalışma, makale-temelli bir problem kurgusunu yeniden üretmekte; ancak mevcut veri yapısına daha uygun, denetimli öğrenme tabanlı bir yaklaşım benimsemektedir.

#### Problem Tanımı

Bu branch'te temel problem, trafoların `burned` veya `not burned` olarak sınıflandırılmasıdır. Ayrıca 2019 ve 2020 verilerinden hareketle 2021 yılı için bir risk projeksiyonu üretilmektedir.

Bu yönüyle branch iki katmanlıdır:

- ikili arıza riski sınıflandırması,
- gelecek yıl için filo düzeyinde risk projeksiyonu.

#### Kullanılan Veri Setleri

Bu branch aşağıdaki iki Excel veri setini kullanmaktadır:

- `data/raw/Dataset_Year_2019.xlsx`
- `data/raw/Dataset_Year_2020.xlsx`

Kod düzeyinde hedef sütun dosya adına göre çözülmektedir:

- `Burned transformers 2019`
- `Burned transformers 2020`

#### Kullanılan Girdi Değişkenleri

Kod doğrulamasına göre ana özellik seti şu değişkenlerden oluşmaktadır:

- `LOCATION`
- `POWER`
- `SELF_PROTECTION`
- `AVG_DISCHARGE`
- `MAX_DISCHARGE`
- `BURNING_RATE`
- `CRITICALITY`
- `REMOVABLE_CONNECTORS`
- `NUM_USERS`
- `ENERGY_NOT_SUPPLIED`
- `AIR_NETWORK`
- `CIRCUIT_QUEUE`
- `NETWORK_LENGTH`
- `IS_RESIDENTIAL`
- `IS_POLE`

Branch ayrıca daha geniş bir mühendislik özellik havuzu da tanımlamaktadır. Bunlar arasında:

- `ENERGY_PER_USER`
- `LIGHTNING_RISK`
- `NETWORK_PER_POWER`
- `DISCHARGE_RANGE`
- `IS_MACRO`
- `LOW_POWER`
- `POWER_LIGHTNING`
- `NETWORK_RISK`

yer almaktadır. Bununla birlikte `main.py` ve proje README anlatısına göre ana karşılaştırma hattı, makale ile daha karşılaştırılabilir olan sadeleştirilmiş supervised SVM akışına odaklanmaktadır.

#### Kullanılan Modeller

Bu branch'in merkezinde SVM tabanlı bir yapı yer almaktadır:

- RBF kernel SVM
- `class_weight='balanced'`
- validation tabanlı threshold optimization

Kod doğrulamasına göre temel model `sklearn.svm.SVC` ile kurulmuştur.

#### Elde Edilen Bulgular

README'de aktarılan son gözlemlere göre:

- 2019 için en iyi model: `Reduced SVM`, `F1 = 0.3467`, `Recall = 0.5098`, `Accuracy = 0.7164`
- 2020 için en iyi model: `Full SVM`, `F1 = 0.2710`, `Recall = 0.5250`, `Accuracy = 0.7135`

2021 projeksiyonunda README içinde verilen son değerler:

- proje tahmini: `1275`
- makale referansı: `852`

#### Akademik Değerlendirme

Bu branch, yüksek sınıf dengesizliği içeren bakım problemlerinde accuracy metriğinin tek başına yeterli olmadığını açık biçimde göstermektedir. Threshold tuning, recall, PR-AUC ve balanced accuracy gibi metriklerin öne çıkarılması metodolojik açıdan yerindedir. Bu nedenle branch, yalnızca modelleme değil, değerlendirme tasarımı açısından da akademik olarak kıymetli bir katkı sunmaktadır.

### 5. `rul_and_failureType_prediction`

Branch bağlantısı:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/rul_and_failureType_prediction

#### Çalışmanın Amacı

Bu çalışma, güç transformatörlerinde iki farklı ama birbiriyle ilişkili problemi aynı çatı altında ele almaktadır:

- `FDD`: Fault Detection and Diagnosis
- `RUL`: Remaining Useful Life prediction

Amaç, DGA tabanlı zaman serilerinden hem arıza tipi teşhisi yapmak hem de ekipmanın kalan faydalı ömrünü tahmin etmektir.

#### Problem Tanımı

Bu branch çok görevli bir kestirimci bakım yaklaşımı temsil etmektedir:

- FDD tarafı çok sınıflı sınıflandırma problemidir.
- RUL tarafı sürekli hedef değişkenli regresyon problemidir.

#### Kullanılan Veri Seti Yapısı

Bu branch ham veriyi çok sayıda CSV zaman serisi dosyası şeklinde tutmaktadır:

- `data/raw/data_train/`
- `data/raw/data_test/`
- `data/raw/data_labels/`

Etiket dosyaları iki hedef taşımaktadır:

- `FDD label`
- `RUL label`

İşlenmiş veri kümeleri `.npy` olarak saklanmaktadır:

- `data/processed_merged/train_set/`
- `data/processed_merged/val_set/`
- `data/processed_merged/test_set/`

README'de raporlanan mevcut tensör boyutları:

- `train: (1680, 200, 4)`
- `val: (420, 200, 4)`
- `test: (900, 200, 4)`

#### Kullanılan Girdi Değişkenleri

Kod doğrulamasına göre sabit uzunluklu zaman serileri aşağıdaki dört gaz değişkeninden oluşmaktadır:

- `H2`
- `CO`
- `C2H4`
- `C2H2`

Her örnek, `200` zaman adımlı ve `4` özellikli diziye dönüştürülmektedir.

FDD tarafında ayrıca istatistiksel özet(mean , std , max , min , last) ve oran tabanlı öznitelikler çıkarılmaktadır. Bu çıkartılan öznitelikler datasete özellik olarak eklenmiştir. README'ye göre başlıca oranlar:

- `R1 = H2 / CO`
- `R2 = C2H2 / C2H4`
- `R3 = H2 / C2H4`
- `R4 = CO / C2H2`

#### Kullanılan Modeller

Bu branch iki model ailesi içermektedir.

FDD için:

- GRNN benzeri sınıflandırıcı
- Random Forest

RUL için:

- GRU
- LSTM

Kod doğrulamasına göre kullanılan başlıca model sınıfları:

- `RandomForestClassifier`
- `tensorflow.keras.layers.GRU`
- `tensorflow.keras.layers.LSTM`

#### Elde Edilen Bulgular

README'de raporlanan temel sonuçlar şunlardır:

FDD tarafı:

- Random Forest, branch içindeki en başarılı FDD modeli olarak öne çıkmaktadır.

RUL tarafı:

- GRU: `Validation MAE = 9.981`, `Validation RMSE = 15.052`, `Test MAE = 10.438`, `Test RMSE = 15.789`
- LSTM: `Validation MAE = 221.588`, `Validation RMSE = 245.454`, `Test MAE = 218.608`, `Test RMSE = 243.766`

#### Akademik Değerlendirme

Bu branch, transformatör kestirimci bakım literatüründe çok önemli iki problemi aynı veri çatısı altında bir araya getirmesi bakımından dikkat çekicidir. Özellikle zaman serisi tabanlı RUL modellemesi ile tablo özniteliği tabanlı FDD modellemesini aynı depoda sunması, deneysel çeşitlilik açısından güçlü bir yapı oluşturmaktadır. Mevcut sonuçlar, bu kurulumda GRU'nun LSTM'den belirgin biçimde daha uygun olduğunu göstermektedir.

## Branch'ler Arası Karşılaştırmalı Özet

| Branch | Temel Amaç | Veri Türü | Ana Görev | Kullanılan Başlıca Modeller |
|---|---|---|---|---|
| `DGA_Analysis` | DGA ile arıza tipi sınıflandırması | Tablo verisi | Çok sınıflı sınıflandırma | Decision Tree, Random Forest, SVM, Dense NN |
| `EFRI` | Elektriksel arıza sınıflandırması | Akım-gerilim tablosu | Çok sınıflı sınıflandırma | Decision Tree, Logistic Regression, Random Forest, SVM, NN |
| `Transformer_Monitoring` | IoT verisiyle alarm tahmini | Zaman damgalı tablo verisi | İkili sınıflandırma | KNN, GaussianNB, Logistic Regression, Random Forest, XGBoost |
| `predictive_maintenance` | Yıllık arıza riski ve bakım önceliklendirmesi | Yıllık işletme/şebeke verisi | İkili sınıflandırma + risk projeksiyonu | RBF SVM |
| `rul_and_failureType_prediction` | FDD ve RUL birlikte | DGA zaman serisi | Çok sınıflı sınıflandırma + regresyon | GRNN, Random Forest, GRU, LSTM |

## Genel Yorum

Depodaki branch'ler birlikte ele alındığında üç önemli eğilim görülmektedir:

1. Tablo yapısındaki transformatör verilerinde ağaç tabanlı ensemble modeller çoğu zaman en güçlü sonuçları üretmektedir.
2. Problem sınıf dengesizliği içerdiğinde yalnızca accuracy kullanımı yetersiz kalmakta; recall, F1, PR-AUC ve balanced accuracy gibi ölçütler daha anlamlı hale gelmektedir.
3. DGA verisi söz konusu olduğunda hem oran temelli öznitelik mühendisliği hem de zaman serisi modelleme yaklaşımı değer üretmektedir; ancak görev tipine göre uygun model ailesi değişmektedir.

Bu yönüyle depo, tek bir modelin üstünlüğünü ilan eden bir yapıdan çok, farklı veri ve problem tiplerinde hangi yaklaşımın neden uygun olabileceğini gösteren deneysel bir araştırma portföyü niteliğindedir.

## Referanslar

Bu bölüm, mevcut klasör yapısında bulunan makalelerin adlarını içermektedir:

1. `Distribution Transformer Failure Prediction for Predictive Maintenance Using Hybrid One-Class Deep SVDD Classification and Lightning Strike Failures Data`
2. `A cognitive system for fault prognosis in power transformers`
3. `AI-Enabled Predictive Maintenance for Distribution Transformers`
4. `On the use of Machine Learning for predictive maintenance of power transformers`
5. `Predictive Maintenance for Distribution System Operators in Increasing Transformers' Reliability`
6. `Dataset of distribution transformers for predictive maintenance`

## Geliştirici

**Emir Furkan Karahasanoglu**  
**Optiway Solutions**  
**2026**
