# Transformer Failure Detection

Bu repository, güç ve dağıtım transformatörlerinde arıza tespiti, elektriksel arıza sınıflandırması, IoT tabanlı durum izleme, kestirimci bakım, pseudo-anomali tabanlı risk modelleme, arıza tipi teşhisi ve kalan faydalı ömür tahmini problemlerini aynı araştırma çatısı altında toplayan çok dallı bir çalışmadır.

`main` branch, tek bir deney kodu barındıran operasyonel bir dal olarak değil, diğer branch'lerde geliştirilen projelerin akademik ve karşılaştırmalı özetini sunan ana giriş noktası olarak tasarlanmıştır. Her branch farklı bir veri yapısı, problem tanımı ve modelleme stratejisi kullanır. Bu nedenle repository bütüncül olarak incelendiğinde, transformatör güvenilirliği ve kestirimci bakım literatüründe farklı makine öğrenmesi yaklaşımlarının hangi koşullarda anlamlı olabileceğini gösteren deneysel bir portföy niteliği taşır.

## Branch Haritası

| Branch | Araştırma Odağı | Veri Tipi | Ana Görev | Başlıca Modeller |
|---|---|---|---|---|
| [`DGA_Analysis`](https://github.com/karahasanoglu/transformer-failure-detection/tree/DGA_Analysis) | DGA verisiyle transformatör arıza tipi sınıflandırması | Gaz konsantrasyonu tablosu | 7 sınıflı sınıflandırma | Decision Tree, Random Forest, SVM, Dense NN |
| [`EFRI`](https://github.com/karahasanoglu/transformer-failure-detection/tree/EFRI) | Akım-gerilim ölçümleriyle elektriksel arıza tespiti | Üç faz akım-gerilim verisi | Çok sınıflı sınıflandırma | Decision Tree, Logistic Regression, Random Forest, SVM, NN |
| [`Transformer_Monitoring`](https://github.com/karahasanoglu/transformer-failure-detection/tree/Transformer_Monitoring) | IoT tabanlı dağıtım transformatörü alarm tahmini | Zaman damgalı operasyonel veri | İkili sınıflandırma | KNN, GaussianNB, Logistic Regression, Random Forest, XGBoost |
| [`predictive_maintenance`](https://github.com/karahasanoglu/transformer-failure-detection/tree/predictive_maintenance) | Dağıtım trafoları için pseudo risk etiketi ve SVM tabanlı sınıflandırma | Yıllık trafo/şebeke verisi | Pseudo-anomali sınıflandırması | KMeans, Isolation Forest, RBF SVM |
| [`2019-2020-Predictive`](https://github.com/karahasanoglu/transformer-failure-detection/tree/2019-2020-Predictive) | 2019'dan öğrenip 2020 üzerinde pseudo risk/anomali tahmini | Yıllık trafo/şebeke verisi | İleri-yıl pseudo risk sınıflandırması | Random Forest, XGBoost, Autoencoder |
| [`rul_and_failureType_prediction`](https://github.com/karahasanoglu/transformer-failure-detection/tree/rul_and_failureType_prediction) | DGA zaman serilerinden FDD ve RUL tahmini | Çok değişkenli DGA zaman serisi | Sınıflandırma + regresyon | GRNN benzeri model, Random Forest, GRU, LSTM |

## Genel Araştırma Kapsamı

Repository'deki projeler tek bir veri setinin varyasyonları değildir. Aksine, transformatör güvenilirliği problemini farklı ölçüm kaynakları ve farklı karar seviyeleri üzerinden ele alır:

- DGA gaz ölçümleriyle iç arıza tiplerinin teşhisi,
- üç fazlı akım ve gerilim sinyalleriyle elektriksel arıza sınıflandırması,
- IoT tabanlı sıcaklık, yağ seviyesi, akım ve gerilim ölçümleriyle alarm tahmini,
- yıllık şebeke ve trafo karakteristiklerinden pseudo risk/anomali üretimi,
- zaman serisi temelli FDD ve RUL tahmini.

Bu çeşitlilik, repository'nin temel katkısını oluşturur. Çalışmalar birlikte değerlendirildiğinde, enerji varlıklarında kestirimci bakımın yalnızca tek bir model seçimi problemi olmadığı; veri kaynağı, etiket güvenilirliği, sınıf dengesizliği, zamansal genelleme ve değerlendirme metriği seçimi gibi çok sayıda metodolojik karar içerdiği görülür.

## 1. DGA_Analysis

### Amaç

`DGA_Analysis` branch'i, güç transformatörlerinde Dissolved Gas Analysis (DGA) verilerini kullanarak arıza tipi sınıflandırması yapmayı amaçlar. Projede iki ayrı DGA veri seti ortak bir etiket uzayında birleştirilmiş, gaz konsantrasyonlarından oran tabanlı öznitelikler üretilmiş ve farklı makine öğrenmesi modelleri karşılaştırılmıştır.

### Veri Setleri ve Etiket Yapısı

Kullanılan veri dosyaları:

- `data/raw/DGA-dataset.csv`
- `data/raw/DGA_dataset2.csv`
- `data/raw/dga_merged_dataset.csv`

Problem, 7 sınıflı bir transformatör arıza teşhisi problemidir:

| Sınıf | Anlam |
|---|---|
| `PD` | Partial Discharge |
| `D1` | Low Energy Discharge |
| `D2` | High Energy Discharge |
| `T1` | Thermal Fault < 300C |
| `T2` | Thermal Fault 300C - 700C |
| `T3` | Thermal Fault > 700C |
| `NF` | No Fault / Normal |

İlk veri setindeki açıklayıcı arıza etiketleri, ikinci veri setindeki IEC benzeri kısa sınıflara eşlenmiştir. Bu sayede iki farklı kaynak tek bir çok sınıflı problem olarak modellenebilmiştir.

### Öznitelikler ve Ön İşleme

Temel gaz değişkenleri:

- `H2`
- `CH4`
- `C2H6`
- `C2H4`
- `C2H2`

Alan bilgisini yansıtan oran değişkenleri:

- `R1 = CH4 / H2`
- `R2 = C2H2 / C2H4`
- `R4 = C2H6 / CH4`
- `R5 = C2H4 / C2H6`

Ön işleme aşamasında etiket standardizasyonu, eksik ve sonsuz değer temizliği, oran öznitelik üretimi ve IQR tabanlı robust scaling uygulanmıştır. İşlenmiş veri seti `data/processed/dga_merged_processed.csv` olarak saklanır.

### Model Sonuçları

| Model | Raporlanan Sonuç |
|---|---:|
| Decision Tree | Test accuracy yaklaşık `0.8960` |
| Random Forest | Test accuracy yaklaşık `0.8936` |
| SVM | Test accuracy yaklaşık `0.7933` |
| Dense Neural Network | Test accuracy yaklaşık `0.75` |

### Akademik Değerlendirme

Bu branch, DGA tabanlı tabular verilerde klasik makine öğrenmesi yaklaşımlarının hâlâ çok güçlü olduğunu göstermektedir. Özellikle Decision Tree ve Random Forest modelleri, hem yüksek doğruluk hem de yorumlanabilirlik açısından öne çıkar. Neural network modeli kabul edilebilir sonuç verse de veri yapısının sınırlı ve tabular olması nedeniyle ağaç tabanlı modellere göre daha değişken performans üretmektedir.

## 2. EFRI

### Amaç

`EFRI` branch'i, elektrik güç sistemlerinde oluşan kısa devre ve arıza tiplerini akım-gerilim ölçümleri üzerinden sınıflandırmayı hedefler. Bu branch transformatör odaklı bakım çalışmalarını tamamlayıcı nitelikte, elektriksel arıza tanıma problemi sunar.

### Veri Seti ve Girdi Değişkenleri

Kullanılan veri dosyası:

- `data/classData.csv`

Model girdileri:

- `Ia`, `Ib`, `Ic`
- `Va`, `Vb`, `Vc`

Bu değişkenler üç fazlı sistemin akım ve gerilim ölçümlerini temsil eder.

### Hedef Sınıflar

Etiketler `G`, `C`, `B`, `A` arıza gösterge sütunlarının kombinasyonlarından türetilir:

| Sınıf | Anlam |
|---|---|
| `No Fault` | Normal çalışma |
| `LG Fault` | Faz-toprak arızası |
| `LL Fault` | Faz-faz arızası |
| `LLG Fault` | İki faz-toprak arızası |
| `LLL Fault` | Üç faz arızası |
| `GLLL Fault` | Üç faz-toprak arızası |

### Model Sonuçları

| Model | Accuracy |
|---|---:|
| Random Forest | `0.8709` |
| Neural Network | `0.8560` |
| Support Vector Machine | `0.8531` |
| Decision Tree | `0.8442` |
| Logistic Regression | `0.3223` |

### Akademik Değerlendirme

Sonuçlar, elektriksel arıza sınıflandırmasının doğrusal modeller için zor bir problem olduğunu göstermektedir. Logistic Regression'ın düşük performansı, sınıf sınırlarının doğrusal ayrılabilirlikten uzak olduğunu düşündürür. Random Forest, SVM ve Neural Network modellerinin daha güçlü sonuç üretmesi, doğrusal olmayan karar yüzeylerinin bu problemde kritik olduğunu gösterir. En belirgin hata kaynağı, birbirine yakın elektriksel karakteristikler taşıyan `GLLL Fault` ve `LLL Fault` sınıflarının karışmasıdır.

## 3. Transformer_Monitoring

### Amaç

`Transformer_Monitoring` branch'i, IoT cihazlarıyla toplanmış dağıtım transformatörü ölçümlerinden `MOG_A` alarm değişkenini tahmin etmeyi amaçlar. Veri, 25 Haziran 2019 ile 14 Nisan 2020 arasında 15 dakikalık aralıklarla toplanmış operasyonel ölçümleri içerir.

### Veri Setleri

Kullanılan dosyalar:

- `data/Overview.csv`
- `data/CurrentVoltage.csv`

Bu iki dosya `DeviceTimeStamp` alanı üzerinden birleştirilir.

### Girdi Değişkenleri

`CurrentVoltage.csv` içinden:

- `VL1`, `VL2`, `VL3`
- `IL1`, `IL2`, `IL3`
- `VL12`, `VL23`, `VL31`
- `INUT`

`Overview.csv` içinden:

- `OTI`
- `WTI`
- `ATI`
- `OLI`
- `OTI_A`
- `OTI_T`

Hedef değişken:

- `MOG_A`

### Model Sonuçları

| Model | Accuracy | Yorum |
|---|---:|---|
| Random Forest | `0.9883` | En dengeli ve güçlü modellerden biri |
| XGBoost | `0.9883` | Random Forest ile benzer düzeyde güçlü |
| KNN | `0.9538` | Genel başarı iyi, alarm sınıfında bazı kaçırmalar var |
| Logistic Regression | `0.9370` | Açıklanabilir baseline, alarm recall değeri sınırlı |
| GaussianNB | `0.8480` | Alarm recall değeri yüksek, false positive maliyeti fazla |

### Akademik Değerlendirme

Bu branch, zaman damgalı operasyonel ölçümlerin alarm tahmini için yüksek bilgi değeri taşıdığını göstermektedir. Ensemble tabanlı Random Forest ve XGBoost modellerinin başarısı, transformatör izleme verilerinde doğrusal olmayan etkileşimlerin güçlü olduğunu düşündürür. GaussianNB'nin alarm sınıfında yüksek recall üretmesi, farklı modellerin kullanım amacına göre seçilmesi gerektiğini de gösterir: genel doğruluk için ensemble modeller, kritik durumları kaçırmamak için daha duyarlı modeller değerlendirilebilir.

## 4. predictive_maintenance

### Amaç

`predictive_maintenance` branch'i, dağıtım transformatörlerinin yıllık teknik ve çevresel özelliklerinden risk/anomali yapısını çıkarmayı amaçlayan akademik bir kestirimci bakım prototipidir. Branch'in güncel yaklaşımı, ham verideki `Burned transformers 2019/2020` kolonlarını doğrudan hedef değişken olarak kullanmaz. Bu kolonlar olası hedef sızıntısı ve etiket güvenilirliği sorunları nedeniyle ön işleme sırasında çıkarılır.

### Veri Setleri

Kullanılan ham dosyalar:

- `data/raw/Dataset_Year_2019.xlsx`
- `data/raw/Dataset_Year_2020.xlsx`

Her iki dosya da `15873` gözlem içerir. Değişkenler trafo gücü, koruma durumu, yıldırım yoğunluğu, kullanıcı tipi, kurulum tipi, kullanıcı sayısı, ağ uzunluğu ve geçmiş risk/kritiklik göstergeleri gibi teknik ve çevresel bilgileri kapsar.

### Özellik Mühendisliği

Ön işleme sonucunda üretilen temel değişkenler:

- `power_per_user`
- `lightning_risk_score`
- `network_density`
- `historical_risk_index`

Kategorik değişkenler one-hot encoding ile sayısallaştırılır. 2019 ve 2020 veri setlerinin aynı kolon şemasını paylaşması için gerekli kolon hizalama işlemleri yapılır.

### Pseudo Etiketleme

Branch'in metodolojik merkezi pseudo etiketleme adımıdır. `target` değişkeni gerçek arıza etiketi değil, algoritmik olarak üretilmiş risk/anomali göstergesidir.

Kullanılan yöntemler:

- `KMeans(n_clusters=2)`
- `IsolationForest(contamination=0.05)`

Nihai hedef mantığı:

```text
target = 1, eğer KMeans veya Isolation Forest gözlemi riskli/anormal işaretlediyse
target = 0, aksi halde
```

Pseudo target dağılımı:

| Veri Seti | target=0 | target=1 | Risk Oranı |
|---|---:|---:|---:|
| 2019 | 12736 | 3137 | 19.76% |
| 2020 | 12778 | 3095 | 19.50% |

### SVM Sonuçları

| Deney | Confusion Matrix | Genel Sonuç |
|---|---|---|
| 2019 SVM | `[[2490, 63], [3, 619]]` | Accuracy `0.9792`, macro F1 `0.9682` |
| 2020 SVM | `[[2491, 65], [6, 613]]` | Accuracy `0.9776`, macro F1 `0.9656` |
| 2019-2020 SVM | `[[4995, 95], [9, 1251]]` | Accuracy yaklaşık `0.98`, macro F1 yaklaşık `0.97` |

### Akademik Değerlendirme

Bu branch'in sonuçları yüksek görünse de gerçek saha arızası tahmini olarak yorumlanmamalıdır. SVM modeli, KMeans ve Isolation Forest tarafından üretilen pseudo hedefi öğrenmektedir. Bu nedenle başarı, esas olarak pseudo risk tanımının sınıflandırılabilirliğini gösterir. Yine de çalışma, güvenilir etiketlerin sınırlı olduğu bakım problemlerinde denetimsiz sinyallerden risk grubu üretme fikrini sistematik biçimde test ettiği için değerlidir.

## 5. 2019-2020-Predictive

### Amaç

`2019-2020-Predictive` branch'i, `predictive_maintenance` hattını daha ileri bir deney tasarımıyla genişletir. Temel amaç, 2019 verisini eğitim/validation kaynağı, 2020 verisini ise ileri-yıl test seti olarak kullanarak pseudo risk/anomali tahminini değerlendirmektir.

Bu tasarım rastgele train-test ayrımına göre daha gerçekçi bir genelleme senaryosu sunar; çünkü model geçmiş yıl yapısından öğrenir ve sonraki yılın risk örüntüsü üzerinde sınanır.

### Veri ve Hedef Tanımı

Veri kaynakları yine 2019 ve 2020 yıllarına ait dağıtım transformatörü Excel dosyalarıdır. `Burned transformers` kolonları güvenilir doğrudan hedef olarak kullanılmaz; preprocessing sırasında çıkarılır. Hedef değişken, KMeans ve Isolation Forest tabanlı pseudo etiketleme sürecinden gelir.

`target` değişkeninin yorumu:

| Etiket | Anlam |
|---:|---|
| `0` | Normal veya düşük riskli gözlem |
| `1` | Pseudo anomali veya yüksek riskli gözlem |

### Kullanılan Modeller

Branch üç model ailesini karşılaştırır:

- Random Forest
- XGBoost
- Autoencoder tabanlı anomali tespiti

Random Forest ve XGBoost pseudo hedefi denetimli biçimde öğrenir. Autoencoder ise normal sınıfı yeniden oluşturmayı öğrenir ve yüksek reconstruction error değerlerini anomali olarak işaretler.

### Model Sonuçları

| Model | Accuracy | target=1 Precision | target=1 Recall | target=1 F1 | Yorum |
|---|---:|---:|---:|---:|---|
| Random Forest | `0.9911` | `0.98` | `0.97` | `0.98` | Dengeli ve yüksek performans |
| XGBoost | `0.9915` | `0.99` | `0.96` | `0.98` | False positive sayısı daha düşük, false negative biraz daha yüksek |
| Autoencoder | `0.9214` | `0.7276` | `0.9538` | `0.8255` | Riskli sınıfı yakalamada duyarlı, false positive maliyeti yüksek |

Confusion matrix özetleri:

```text
Random Forest:
[[12714, 64],
 [78, 3017]]

XGBoost:
[[12755, 23],
 [112, 2983]]

Autoencoder:
[[11673, 1105],
 [143, 2952]]
```

### Akademik Değerlendirme

Bu branch, pseudo etiketli risk tahmininde model seçimindeki temel ödünleşimi açık biçimde gösterir. Random Forest ve XGBoost çok yüksek genel başarı üretirken, Autoencoder daha düşük accuracy ile daha yüksek risk sınıfı duyarlılığı sağlar. Bu fark, operasyonel bakım senaryolarında model seçiminin yalnızca accuracy üzerinden değil, yanlış negatif ve yanlış pozitif maliyetleri üzerinden yapılması gerektiğini gösterir.

## 6. rul_and_failureType_prediction

### Amaç

`rul_and_failureType_prediction` branch'i, güç transformatörlerinde iki ilişkili kestirimci bakım problemini aynı veri çatısı altında ele alır:

- `FDD`: Fault Detection and Diagnosis
- `RUL`: Remaining Useful Life prediction

Amaç, DGA tabanlı zaman serilerinden hem arıza sınıfını belirlemek hem de ekipmanın kalan faydalı ömrünü tahmin etmektir.

### Veri Yapısı

Ham veri, her biri tek bir transformatör örneğine ait zaman serisi CSV dosyalarından oluşur:

- `data/raw/data_train/`
- `data/raw/data_test/`
- `data/raw/data_labels/`

Kullanılan gaz değişkenleri:

- `H2`
- `CO`
- `C2H4`
- `C2H2`

Ön işleme sonucunda her örnek sabit uzunluklu bir tensöre dönüştürülür:

```text
(sequence_length, feature_count) = (200, 4)
```

İşlenmiş veri boyutları:

| Bölüm | Boyut |
|---|---|
| Train | `(1680, 200, 4)` |
| Validation | `(420, 200, 4)` |
| Test | `(900, 200, 4)` |

### FDD Yaklaşımı

FDD tarafında ham zaman serileri doğrudan modele verilmez; her gaz için istatistiksel özetler çıkarılır:

- mean
- std
- max
- min
- last

Bunlara ek olarak oran tabanlı DGA öznitelikleri kullanılır:

- `R1 = H2 / CO`
- `R2 = C2H2 / C2H4`
- `R3 = H2 / C2H4`
- `R4 = CO / C2H2`

FDD modelleri:

- GRNN benzeri çekirdek tabanlı sınıflandırıcı
- Random Forest

FDD sonuçları:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| GRNN benzeri model | `0.92` | `0.80` | `0.92` |
| Random Forest | `0.96` | `0.91` | `0.96` |

Random Forest, özellikle azınlık sınıflarında daha dengeli sonuç üretmiştir. Öznitelik önemleri incelendiğinde `R1_H2_CO`, `CO_last`, `CO_max`, `R3_H2_C2H4` ve `CO_mean` gibi CO ve oran tabanlı değişkenlerin üst sıralarda yer aldığı görülmektedir.

### RUL Yaklaşımı

RUL tarafında zaman sırası korunur ve derin öğrenme tabanlı regresyon modelleri kullanılır:

- GRU
- LSTM

RUL sonuçları:

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| GRU | `9.981` | `15.052` | `10.438` | `15.789` |
| LSTM | `221.588` | `245.454` | `218.608` | `243.766` |

### Akademik Değerlendirme

Bu branch, aynı DGA zaman serisi veri yapısından hem sınıflandırma hem de regresyon problemi üretmesi açısından repository'nin en kapsamlı deneylerinden biridir. FDD tarafında Random Forest, sınıf dengesizliğine rağmen güçlü sonuç verir. RUL tarafında GRU modeli mevcut deney düzeninde LSTM'den belirgin biçimde daha başarılıdır. Ancak GRU ve LSTM sonuçları yorumlanırken iki modelin farklı pencere ve hedef ölçekleme stratejileri kullandığı dikkate alınmalıdır.

## Branch'ler Arası Metodolojik Karşılaştırma

Repository genelinde dört ana metodolojik eksen görülür:

1. **Tabular sınıflandırma:** `DGA_Analysis`, `EFRI` ve `Transformer_Monitoring` branch'lerinde klasik makine öğrenmesi ve ensemble modeller çok güçlüdür.
2. **Pseudo etiketli risk modelleme:** `predictive_maintenance` ve `2019-2020-Predictive` branch'lerinde gerçek saha etiketinin güvenilirliği tartışmalı olduğundan, KMeans ve Isolation Forest ile üretilen pseudo hedefler kullanılır.
3. **Zaman serisi tabanlı ömür tahmini:** `rul_and_failureType_prediction` branch'i, DGA dizilerini sabit uzunluklu tensörlere dönüştürerek GRU ve LSTM tabanlı RUL modellemesi yapar.
4. **Değerlendirme metriği duyarlılığı:** Sınıf dengesizliği ve bakım maliyetleri nedeniyle accuracy tek başına yeterli değildir; recall, precision, F1, macro F1, confusion matrix, MAE ve RMSE gibi görev odaklı metrikler birlikte değerlendirilmelidir.

## Genel Bulgular

Repository bütüncül olarak incelendiğinde aşağıdaki sonuçlar öne çıkar:

- Yapılandırılmış tabular verilerde Random Forest ve XGBoost gibi ağaç tabanlı ensemble yöntemler tekrar tekrar güçlü sonuç vermektedir.
- DGA verilerinde oran tabanlı öznitelik mühendisliği, arıza teşhisi performansını destekleyen önemli bir bileşendir.
- Elektriksel arıza sınıflandırmasında doğrusal modeller yetersiz kalırken, doğrusal olmayan modeller sınıfları daha iyi ayırmaktadır.
- IoT tabanlı izleme verilerinde alarm tahmini yüksek doğrulukla yapılabilmektedir; ancak alarm sınıfı için recall ve precision dengesi ayrıca incelenmelidir.
- Pseudo etiketli bakım çalışmalarında yüksek metrikler gerçek arıza tahmini anlamına gelmez; sonuçlar pseudo risk tanımının öğrenilebilirliği olarak yorumlanmalıdır.
- RUL tahmininde model mimarisi kadar hedef ölçekleme, zaman penceresi seçimi ve veri dağılımı da sonuçları güçlü biçimde etkiler.

## Akademik Sınırlılıklar

Bu repository araştırma ve prototipleme amacı taşır. Mevcut branch'lerin ortak sınırlılıkları şunlardır:

- Bazı branch'lerde dosya yolları mutlak path biçiminde yazılmıştır; taşınabilirlik için göreli path yapısı önerilir.
- Deneylerin tamamında ortak bir experiment tracking veya tek tip pipeline bulunmamaktadır.
- Bazı sonuçlar tek train-test ayrımı üzerinden raporlanmıştır; çapraz doğrulama ve zaman temelli validasyon kapsamı genişletilebilir.
- Pseudo etiketli branch'lerde `target=1` gerçek arıza anlamına gelmez; saha doğrulaması olmadan operasyonel karar mekanizmasına doğrudan aktarılmamalıdır.
- Sınıf dengesizliği bulunan problemlerde yalnızca accuracy ile yorum yapmak yanıltıcı olabilir.

## Sonuç

Bu repository, transformatör arıza tespiti ve kestirimci bakım problemlerini farklı veri kaynakları ve model aileleri üzerinden inceleyen açıklayıcı bir araştırma portföyüdür. Branch'ler birlikte ele alındığında, tek bir modelin her senaryo için üstün olmadığı; veri tipi, etiket güvenilirliği, sınıf dağılımı, zamansal yapı ve bakım kararının maliyeti gibi faktörlerin model seçimini doğrudan etkilediği görülmektedir.

Ana bulgu şudur: transformatör bakımında başarılı bir yapay zeka yaklaşımı yalnızca yüksek doğruluk üreten bir modelden ibaret değildir. Güvenilir veri hazırlama, anlamlı öznitelik mühendisliği, doğru problem formülasyonu, uygun metrik seçimi ve sonuçların sınırlılıklarıyla birlikte yorumlanması en az model performansı kadar önemlidir.

## Geliştirici

**Emir Furkan Karahasanoglu**  
**Optiway Solutions**  
**2026**
