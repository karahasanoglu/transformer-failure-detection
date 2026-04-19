# Power Transformer RUL and FDD Prediction

Bu çalışma, güç transformatörlerinde kestirimci bakım yaklaşımını desteklemek amacıyla geliştirilmiş bir veri bilimi ve makine öğrenmesi projesidir. Projenin temel amacı, yağ içinde çözünmüş gaz analizi (Dissolved Gas Analysis, DGA) verilerinden yararlanarak iki kritik problemi eş zamanlı biçimde ele almaktır: arıza tespiti ve teşhisi (Fault Detection and Diagnosis, FDD) ile kalan faydalı ömür tahmini (Remaining Useful Life, RUL).

Endüstriyel enerji sistemlerinde transformatör arızaları yüksek bakım maliyeti, plansız duruş ve güvenilirlik kaybı gibi ciddi sonuçlar doğurabilmektedir. Bu bağlamda erken uyarı ve ömür tahmini, bakım kararlarının veri temelli olarak alınmasında önemli bir rol üstlenmektedir. Bu depo, söz konusu probleme yönelik olarak hem klasik makine öğrenmesi hem de derin öğrenme temelli zaman serisi yaklaşımlarını aynı deneysel çatı altında bir araya getirmektedir.

## Çalışmanın Kapsamı

Proje aşağıdaki iki analitik görevi hedeflemektedir:

1. `FDD`: Transformatörde gözlenen gaz davranışlarına bağlı olarak arıza sınıfının belirlenmesi.
2. `RUL`: Mevcut operasyonel geçmişe dayanarak ekipmanın kalan faydalı ömrünün tahmin edilmesi.

Bu kapsamda veri yükleme, doğrulama, sabit uzunlukta dizi üretimi, öznitelik çıkarımı, sınıf dağılımı analizi, model eğitimi ve değerlendirme adımları modüler bir yapı içinde ele alınmıştır.

## Veri Kümesi ve Problem Tanımı

Veri kümesi, her biri tek bir transformatöre ait zaman serisi gözlemlerini temsil eden CSV dosyalarından oluşmaktadır. Her örnek, dört temel DGA değişkeni içermektedir:

| Değişken | Açıklama |
| --- | --- |
| `H2` | Hidrojen |
| `CO` | Karbon monoksit |
| `C2H4` | Etilen |
| `C2H2` | Asetilen |

Her ham örnek, zaman boyunca ölçülmüş çok değişkenli bir seri biçimindedir. Ön işleme aşamasında tüm örnekler ortak bir forma dönüştürülmektedir:

```text
(sequence_length, feature_count) = (200, 4)
```

Ham veri ve etiket yapısı aşağıdaki dizinlerde tutulmaktadır:

```text
data/raw/
├── data_train/
├── data_test/
└── data_labels/
```

Etiket dosyaları iki farklı hedef değişken içermektedir:

| Etiket | Açıklama |
| --- | --- |
| `FDD label` | Arıza sınıfı |
| `RUL label` | Kalan faydalı ömür |

## Veri Büyüklüğü

Depoda bulunan veri yapısı mevcut durumda aşağıdaki ölçeklere sahiptir:

| Bölüm | Örnek sayısı |
| --- | ---: |
| Ham eğitim dosyaları | 2100 |
| Ham test dosyaları | 900 |
| İşlenmiş eğitim kümesi | 1680 |
| İşlenmiş doğrulama kümesi | 420 |
| İşlenmiş test kümesi | 900 |

İşlenmiş veri tensörlerinin boyutları aşağıdaki gibidir:

```text
train: (1680, 200, 4)
val  : (420, 200, 4)
test : (900, 200, 4)
```

RUL etiketlerinin eğitim kümesindeki gözlenen temel aralığı aşağıdaki şekildedir:

```text
min = 362.0
max = 1093.0
mean = 785.89526
```

## Veri İşleme ve Deney Akışı

Projede benimsenen genel işlem hattı aşağıdaki gibi özetlenebilir:

```text
Raw CSV files
-> Data loading
-> Data validation
-> Sequence standardization
-> Feature extraction / normalization
-> Model training
-> Evaluation
```

Bu akış, iki görev için farklı modelleme stratejileri kullansa da ortak bir veri hazırlama mantığı üzerine kuruludur.

## Proje Mimarisi

Kod tabanı ana hatlarıyla aşağıdaki yapıyı izlemektedir:

```text
src/
├── data_preprocessing/
│   ├── load_data.py
│   ├── validate_data.py
│   └── build_sequences.py
└── models/
    ├── check_class_distrubiton.py
    ├── fdd_prediction_models/
    │   ├── train_grnn_fdd.py
    │   └── train_random_forrest_fdd.py
    └── rul_predictiction_model/
        ├── train_gru.py
        └── train_lstm.py
```

## Ön İşleme Bileşenleri

### `load_data.py`

Bu modül, ham CSV dosyalarını ve etiket dosyalarını okuyarak her örnek için yapılandırılmış bir veri nesnesi üretir. Her örnekte şu bilgiler saklanmaktadır:

- dosya adı,
- veri ayrımı (`train` veya `test`),
- çalışma koşulu,
- transformatör kimliği,
- zaman serisi veri çerçevesi,
- FDD etiketi,
- RUL etiketi.

Dosya adları üzerinden çalışma koşulu ve transformatör kimliği ayrıştırılmakta, ardından ilgili etiketlerle örnek bazında eşleştirme yapılmaktadır.

### `validate_data.py`

Bu modül, veri kalitesini sistematik olarak doğrulamak amacıyla aşağıdaki kontrolleri uygular:

- veri nesnesinin beklenen tipte olup olmadığı,
- dosya adının geçerli biçimde bulunup bulunmadığı,
- veri çerçevesinin boş olup olmadığı,
- zorunlu gaz sütunlarının mevcut olup olmadığı,
- gaz sütunlarının sayısal tipte olup olmadığı,
- eksik değer bulunup bulunmadığı,
- FDD ve RUL etiketlerinin varlığı ve sayısal uygunluğu.

Bu aşama, özellikle deneylerin tekrar üretilebilirliğini ve veri kaynaklı sessiz hataların engellenmesini desteklemektedir.

### `build_sequences.py`

Bu modül, tüm zaman serilerini sabit uzunluklu dizilere dönüştürür. Eğer örnek uzunluğu `200` adımın altındaysa başa sıfır dolgusu eklenir; daha uzun serilerde ise son `200` zaman adımı korunur. Böylece tüm örnekler model eğitimine uygun ortak bir tensör biçimine getirilir.

Ayrıca eğitim kümesi, `train_test_split` yardımıyla doğrulama kümesine ayrılır:

- eğitim oranı: `%80`
- doğrulama oranı: `%20`
- rastgelelik tohumu: `42`

Çıktılar `.npy` formatında aşağıdaki dizinlere kaydedilir:

```text
data/processed_merged/
├── train_set/
├── val_set/
└── test_set/
```

## Sınıf Dengesizliği Analizi

FDD görevi için eğitim kümesinde belirgin bir sınıf dengesizliği gözlenmektedir. `check_class_distrubiton.py` dosyasında bu dağılım raporlanmaktadır. İşlenmiş eğitim kümesindeki mevcut dağılım şöyledir:

| Sınıf | Örnek sayısı | Yüzde |
| --- | ---: | ---: |
| 1 | 1367 | 81.37 |
| 2 | 69 | 4.11 |
| 3 | 94 | 5.60 |
| 4 | 150 | 8.93 |

Bu bulgu, özellikle FDD modellerinin değerlendirilmesinde yalnızca genel doğruluk metriğine değil, sınıf bazlı raporlara ve karışıklık matrisine de odaklanılması gerektiğini göstermektedir.

## FDD Modelleri

FDD problemi, çok sınıflı bir sınıflandırma problemi olarak ele alınmıştır. Her iki FDD modelinde de ham zaman serileri doğrudan kullanılmak yerine örnek-temelli öznitelik çıkarımı yapılmaktadır.

### Kullanılan Öznitelikler

Her gaz değişkeni için aşağıdaki istatistiksel özetler hesaplanmaktadır:

- ortalama (`mean`)
- standart sapma (`std`)
- maksimum (`max`)
- minimum (`min`)
- son gözlem (`last`)

Dört gaz için bu yaklaşım toplam `20` temel öznitelik üretmektedir. Buna ek olarak son gözlemler üzerinden dört oran tanımlanmaktadır:

```text
R1 = H2 / CO
R2 = C2H2 / C2H4
R3 = H2 / C2H4
R4 = CO / C2H2
```

Böylece FDD tarafında toplam `24` öznitelik kullanılır. Bu oranlar, DGA tabanlı arıza teşhisi literatüründe sık kullanılan tanısal ilişkilerle uyumludur.

### `train_grnn_fdd.py`

Bu modülde Genelleştirilmiş Regresyon Sinir Ağı benzeri çekirdek tabanlı bir sınıflandırıcı uygulanmıştır. Yaklaşımın temel adımları şunlardır:

- eğitim ve test klasörlerinden tüm CSV dosyalarının yüklenmesi,
- örnek bazlı istatistiksel ve oran özniteliklerinin çıkarılması,
- `StandardScaler` ile özniteliklerin standartlaştırılması,
- `sigma = 0.5` parametresiyle GRNN benzeri sınıflandırıcının eğitilmesi,
- karışıklık matrisi ve sınıflandırma raporu üretilmesi.

Bu model, özellikle görece daha küçük veya yapılandırılmış öznitelik uzaylarında hızlı prototipleme için yararlı bir karşılaştırma tabanı sunmaktadır.

### `train_random_forrest_fdd.py`

Bu modülde sınıf dengesizliğine daha dayanıklı bir ağaç topluluğu yaklaşımı benimsenmiştir. Kullanılan temel yapılandırma aşağıdaki gibidir:

- model: `RandomForestClassifier`
- ağaç sayısı: `300`
- `class_weight`: `balanced_subsample`
- `random_state`: `42`
- `n_jobs`: `-1`

Model, sınıflandırma raporunun yanı sıra öznitelik önem skorlarını da üretmektedir. Bu yönüyle yalnızca tahmin başarısı değil, karar sürecinde hangi gaz temelli değişkenlerin daha etkili olduğuna ilişkin yorumlanabilirlik de sağlamaktadır.

## RUL Modelleri

RUL problemi, çok değişkenli zaman serilerinden sürekli değer tahmini yapan bir regresyon problemi olarak tasarlanmıştır. FDD yaklaşımından farklı olarak bu bölümde örneklerin sıralı yapısı korunmaktadır.

### Ortak Ön İşlem Mantığı

Her iki derin öğrenme modelinde de:

- giriş verileri `float32` olarak yüklenir,
- eğitim kümesi istatistikleri kullanılarak z-normalizasyon uygulanır,
- erken durdurma (`EarlyStopping`) ile aşırı öğrenme riski azaltılır,
- değerlendirme için `MAE` ve `RMSE` hesaplanır.

### `train_lstm.py`

Bu modül, tüm `200` zaman adımını kullanan LSTM tabanlı bir regresyon modeli kurmaktadır.

Mimari özet:

- `Input(shape=(200, 4))`
- `LSTM(64)`
- `Dropout(0.2)`
- `Dense(32, activation="relu")`
- `Dense(1)`

Eğitim ayarları:

- kayıp fonksiyonu: `mse`
- metrik: `mae`
- epoch sayısı: `50`
- batch size: `32`
- erken durdurma sabrı: `8`

Bu yaklaşımda RUL etiketleri, eğitim kümesindeki maksimum değere bölünerek ölçeklenmektedir.

### `train_gru.py`

Bu modül, son `50` zaman adımına odaklanan GRU tabanlı daha kompakt bir sıra modelidir. Bu tasarım, özellikle yakın dönem davranışın ömür tahmininde daha belirleyici olduğu varsayımını sınamaktadır.

Mimari özet:

- `Input(shape=(50, 4))`
- `GRU(64)`
- `Dropout(0.2)`
- `Dense(32, activation="relu")`
- `Dense(1)`

Eğitim ayarları:

- kayıp fonksiyonu: `mae`
- metrik: `mae`
- epoch sayısı: `50`
- batch size: `32`
- erken durdurma sabrı: `8`
- RUL kırpma eşiği: `500.0`

Bu modelde hedef değişken önce `0-500` aralığında kırpılmakta, ardından normalize edilmektedir. Böylece aşırı büyük hedef değerlerin eğitim dinamikleri üzerindeki etkisi sınırlandırılmaktadır.

## Deneysel Bulgular

Bu bölümde, proje kapsamında elde edilen örnek deney çıktıları özetlenmektedir. Sonuçlar, hem sınıflandırma hem de regresyon görevleri açısından modellerin göreli başarısını ortaya koymaktadır.

### FDD Sonuçları

FDD görevi için GRNN ve Random Forest modelleri test kümesi üzerinde değerlendirilmiştir. Elde edilen sonuçlar, Random Forest yaklaşımının tüm temel ölçütlerde daha güçlü bir performans sergilediğini göstermektedir.

#### Genel karşılaştırma

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| GRNN | 0.92 | 0.80 | 0.92 |
| Random Forest | 0.96 | 0.91 | 0.96 |

#### GRNN FDD sınıflandırma raporu

| Sınıf | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.96 | 0.97 | 0.96 | 731 |
| 2 | 0.81 | 0.79 | 0.80 | 38 |
| 3 | 0.76 | 0.65 | 0.70 | 49 |
| 4 | 0.72 | 0.73 | 0.73 | 82 |
| Macro Avg | 0.81 | 0.79 | 0.80 | 900 |
| Weighted Avg | 0.92 | 0.92 | 0.92 | 900 |

GRNN modeline ait karışıklık matrisi:

```text
[[707   0   9  15]
 [  2  30   0   6]
 [ 15   0  32   2]
 [ 14   7   1  60]]
```

Bu sonuçlar, GRNN modelinin baskın sınıf olan `Sınıf 1` üzerinde oldukça başarılı olduğunu; buna karşın `Sınıf 3` ve `Sınıf 4` gibi daha sınırlı örnek sayısına sahip sınıflarda performans kaybı yaşadığını göstermektedir.

#### Random Forest FDD sınıflandırma raporu

| Sınıf | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.98 | 0.99 | 0.98 | 731 |
| 2 | 0.90 | 0.97 | 0.94 | 38 |
| 3 | 0.93 | 0.88 | 0.91 | 49 |
| 4 | 0.85 | 0.82 | 0.83 | 82 |
| Macro Avg | 0.92 | 0.91 | 0.91 | 900 |
| Weighted Avg | 0.96 | 0.96 | 0.96 | 900 |

Random Forest modeline ait karışıklık matrisi:

```text
[[721   0   2   8]
 [  0  37   0   1]
 [  3   0  43   3]
 [ 10   4   1  67]]
```

Random Forest yaklaşımı, özellikle azınlık sınıflarında daha dengeli sonuçlar üretmiştir. `Sınıf 2` ve `Sınıf 3` için elde edilen yüksek `precision` ve `recall` değerleri, sınıf dengesizliğine rağmen modelin ayırt edici örüntüleri daha etkili biçimde yakalayabildiğini göstermektedir.

#### Random Forest öznitelik önemleri

Random Forest modelinin ürettiği ilk 15 öznitelik önem skoru aşağıda sunulmuştur:

| Sıra | Öznitelik | Önem |
| --- | --- | ---: |
| 1 | `R1_H2_CO` | 0.124147 |
| 2 | `CO_last` | 0.121790 |
| 3 | `CO_max` | 0.099025 |
| 4 | `R3_H2_C2H4` | 0.096098 |
| 5 | `CO_mean` | 0.082340 |
| 6 | `C2H4_max` | 0.067118 |
| 7 | `C2H4_last` | 0.065514 |
| 8 | `CO_min` | 0.048301 |
| 9 | `C2H4_mean` | 0.043669 |
| 10 | `C2H2_last` | 0.031746 |
| 11 | `C2H4_min` | 0.029722 |
| 12 | `R4_CO_C2H2` | 0.025785 |
| 13 | `CO_std` | 0.020042 |
| 14 | `C2H2_max` | 0.019722 |
| 15 | `C2H4_std` | 0.014879 |

Bu sıralama, özellikle `CO` tabanlı özniteliklerin ve DGA oran değişkenlerinin arıza sınıflandırmasında belirleyici rol oynadığını düşündürmektedir. Alan bilgisiyle uyumlu olarak, oran temelli değişkenlerin üst sıralarda yer alması öznitelik mühendisliği yaklaşımının katkısını desteklemektedir.

### RUL Sonuçları

RUL görevi için LSTM ve GRU tabanlı iki farklı derin öğrenme modeli değerlendirilmiştir. Elde edilen bulgular, mevcut kurulum altında GRU modelinin LSTM modeline kıyasla çok daha düşük hata değerleri ürettiğini göstermektedir.

#### Genel karşılaştırma

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| GRU | 9.981 | 15.052 | 10.438 | 15.789 |
| LSTM | 221.588 | 245.454 | 218.608 | 243.766 |

#### GRU değerlendirmesi

GRU modeli için doğrulama ve test sonuçları aşağıdaki gibidir:

```text
VALIDATION
MAE : 9.981
RMSE: 15.052

TEST
MAE : 10.438
RMSE: 15.789
```

Örnek tahminler incelendiğinde modelin özellikle `500` çevresinde yoğunlaşan hedef değerler için oldukça yakın tahminler ürettiği görülmektedir:

```text
Validation examples:
true=500.00   pred=484.15
true=500.00   pred=504.38
true=500.00   pred=502.12

Test examples:
true=500.00   pred=492.91
true=430.00   pred=499.29
true=468.00   pred=503.13
```

Bu güçlü performans, modelin son `50` zaman adımına odaklanmasının ve hedef değişkenin `500` seviyesinde kırpılmasının hata ölçütlerini önemli ölçüde düşürdüğünü göstermektedir. Bununla birlikte bu sonuçlar yorumlanırken, modelin doğrudan tüm RUL aralığını değil kırpılmış hedef uzayını öğrendiği mutlaka dikkate alınmalıdır.

#### LSTM değerlendirmesi

LSTM modeli için doğrulama ve test sonuçları aşağıdaki gibidir:

```text
VALIDATION
MAE : 221.588
RMSE: 245.454

TEST
MAE : 218.608
RMSE: 243.766
```

İlk örnek tahminler, modelin gerçek değerlerden belirgin biçimde sapabildiğini göstermektedir:

```text
Validation examples:
true=810.00    pred=905.85
true=1093.00   pred=777.49
true=572.00    pred=789.86

Test examples:
true=693.00    pred=876.49
true=430.00    pred=751.91
true=502.00    pred=840.33
```

Bu tablo, mevcut LSTM kurulumunun veri dağılımını yeterince iyi modelleyemediğini ve tahminlerin ortalama etrafında yoğunlaşma eğilimi gösterdiğini düşündürmektedir. Özellikle yüksek ve düşük RUL seviyelerinin ayrıştırılmasında modelin sınırlı kaldığı görülmektedir.

#### Bulguların yorumu

Mevcut deneyler birlikte değerlendirildiğinde aşağıdaki çıkarımlar yapılabilir:

- FDD görevi için en başarılı yaklaşım `Random Forest` modelidir.
- GRNN modeli kabul edilebilir bir taban çizgisi sağlasa da azınlık sınıflarında performansı daha sınırlıdır.
- RUL görevi için `GRU` modeli, mevcut deney düzeninde `LSTM` modelinden açık biçimde üstündür.
- Ancak GRU sonucunun doğrudan LSTM ile bire bir karşılaştırılmasında dikkatli olunmalıdır; çünkü iki model farklı hedef ölçekleme ve zaman penceresi stratejileri kullanmaktadır.

## Bağımlılıklar

Proje, `pyproject.toml` dosyasında aşağıdaki temel bağımlılıkları tanımlamaktadır:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

Gerekli Python sürümü:

```text
Python >= 3.12
```

## Kurulum

Sanal ortam kullanılarak örnek bir kurulum aşağıdaki şekilde yapılabilir:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Alternatif olarak doğrudan bağımlılık kurulumu da yapılabilir:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Çalıştırma Adımları

Önerilen deney sırası aşağıdaki gibidir:

### 1. Veri yükleme ve doğrulama

```bash
python3 src/data_preprocessing/load_data.py
python3 src/data_preprocessing/validate_data.py
```

### 2. Eğitim, doğrulama ve test dizilerinin oluşturulması

```bash
python3 src/data_preprocessing/build_sequences.py
```

### 3. Sınıf dağılımının incelenmesi

```bash
python3 src/models/check_class_distrubiton.py
```

### 4. FDD modellerinin eğitilmesi

```bash
python3 src/models/fdd_prediction_models/train_grnn_fdd.py
python3 src/models/fdd_prediction_models/train_random_forrest_fdd.py
```

### 5. RUL modellerinin eğitilmesi

```bash
python3 src/models/rul_predictiction_model/train_lstm.py
python3 src/models/rul_predictiction_model/train_gru.py
```

## Değerlendirme Yaklaşımı

Projede görev türüne göre farklı değerlendirme ölçütleri kullanılmaktadır:

### FDD için

- karışıklık matrisi,
- sınıf bazlı `precision`, `recall` ve `f1-score`,
- genel sınıflandırma raporu.

### RUL için

- `MAE` (Mean Absolute Error),
- `RMSE` (Root Mean Squared Error),
- örnek tahminlerin sayısal karşılaştırması.

Bu yaklaşım, hem sınıflandırma hem de regresyon problemlerinde performansın daha bütüncül değerlendirilmesini amaçlamaktadır.

## Güçlü Yönler

Bu deponun öne çıkan katkıları aşağıdaki biçimde özetlenebilir:

- aynı veri kümesi üzerinde hem FDD hem de RUL görevlerinin birlikte ele alınması,
- klasik ve derin öğrenme temelli yaklaşımların karşılaştırılabilir biçimde sunulması,
- modüler veri hazırlama hattı sayesinde deneylerin yeniden üretilebilir olması,
- DGA tabanlı oran öznitelikleri ile alan bilgisine dayalı öznitelik mühendisliği yapılması,
- sınıf dengesizliği probleminin açık biçimde göz önünde bulundurulması.

## Mevcut Sınırlılıklar

Çalışmanın mevcut uygulamasında aşağıdaki geliştirme alanları bulunmaktadır:

- bazı dosya yolları mutlak yol biçiminde tanımlanmıştır; farklı makinelerde taşınabilirlik için göreli yol yapısına dönüştürülmesi önerilir,
- dizin ve modül adlarında yazım tutarsızlıkları bulunmaktadır (`predictiction`, `forrest`, `distrubiton`),
- bağımlılık ve tekrar üretilebilirlik için tam deney sonuçlarının ayrıca raporlanması faydalı olacaktır,
- FDD tarafında çapraz doğrulama, hiperparametre taraması ve dengesiz veri teknikleri ile performans daha da geliştirilebilir,
- RUL tarafında dikkat mekanizmaları, TCN veya Transformer tabanlı mimariler ileriki çalışmalar için değerlendirilebilir.

## Gelecek Çalışmalar

Bu projenin akademik veya endüstriyel kullanımını ileri taşımak için aşağıdaki yönlerde geliştirme yapılabilir:

- veri artırma ve sentetik örnek üretimi,
- çok görevli öğrenme ile FDD ve RUL problemlerinin ortak model altında ele alınması,
- açıklanabilir yapay zeka yöntemleri ile karar mekanizmalarının yorumlanması,
- çevrim içi izleme altyapılarına entegrasyon,
- farklı transformatör işletme koşullarında genellenebilirliğin test edilmesi.

## Sonuç

Bu depo, güç transformatörleri için veri temelli arıza teşhisi ve ömür tahmini problemlerini ele alan bütüncül bir deney ortamı sunmaktadır. DGA verilerinden hareketle geliştirilen bu yapı; veri hazırlama, öznitelik çıkarımı, sınıflandırma ve regresyon aşamalarını tek bir çalışma alanında toplamakta ve kestirimci bakım araştırmaları için sağlam bir başlangıç noktası oluşturmaktadır.
