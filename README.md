# 2019-2020 Transformer Risk and Pseudo-Anomaly Prediction

Bu proje, 2019 ve 2020 yillarina ait trafo ve dagitim sebekesi verileri uzerinden pseudo etiketli risk/anomali tahmini yapmayi amaclayan bir makine ogrenmesi calismasidir. Ham veri setlerinde yer alan `Burned transformers 2019` ve `Burned transformers 2020` kolonlari guvenilir saha etiketi olarak kabul edilmemistir. Bu nedenle bu kolonlar modelleme surecinden cikarilmis, hedef degisken daha sonra KMeans ve Isolation Forest algoritmalari kullanilarak pseudo etiket olarak yeniden uretilmistir.

Bu calismada `target` degiskeni gercek ariza kaydi degil, algoritmik olarak uretilmis bir risk/anomali gostergesidir:

| Etiket | Anlam |
|---:|---|
| `0` | Normal veya dusuk riskli gozlem |
| `1` | Pseudo anomali veya yuksek riskli gozlem |

Bu ayrim onemlidir: model performans metrikleri gercek saha arizasini tahmin etme basarisi olarak degil, pseudo labelling mekanizmasi ile uretilen hedef yapisini ogrenme basarisi olarak yorumlanmalidir.

## Proje Amaci

Calismanin temel amaci, guvenilir olmayan dogrudan ariza etiketlerini kullanmadan trafo riskini temsil edebilecek alternatif bir hedef degisken olusturmak ve bu hedef degisken uzerinden farkli model ailelerini karsilastirmaktir. Bu kapsamda:

- Ham Excel dosyalarindan guvenilir olmayan yanma kolonlari cikarilmistir.
- Trafo ve sebekeye ait sayisal ve kategorik degiskenler temizlenmistir.
- Riskle iliskili olabilecek yeni degiskenler uretilmistir.
- KMeans ve Isolation Forest ile pseudo etiketleme yapilmistir.
- 2019 yili egitim/validation, 2020 yili ileri-yil test seti olarak kullanilmistir.
- Random Forest, XGBoost ve Autoencoder yaklasimlari karsilastirilmistir.

## Proje Yapisi

```text
.
├── data/
│   ├── raw/
│   │   ├── Dataset_Year_2019.xlsx
│   │   └── Dataset_Year_2020.xlsx
│   └── processed/
│       ├── processed_dataset_2019.csv
│       ├── processed_dataset_2020.csv
│       ├── 2019_pseudo.csv
│       ├── 2020_pseudo.csv
│       ├── normalized_dataset_2019.csv
│       └── normalized_dataset_2020.csv
├── results/
│   ├── 2019 Correlation Matrix.png
│   ├── 2020 Correlation Matrix.png
│   ├── tsne_2019.png
│   ├── tsne_2020.png
│   ├── RandomForest-ConfusionMatrix.png
│   ├── XG-Boost-ConfusionMatrix.png
│   ├── AutoEncoder-ConfusionMatrix.png
│   ├── burned_transformers_2019_correlation.png
│   └── burned_transformers_2020_correlation.png
├── src/
│   ├── preprocessing.py
│   ├── labelling.py
│   ├── normalization.py
│   ├── check_distrubition.py
│   └── models/
│       ├── random_forrest.py
│       ├── XG_Boost.py
│       └── autoencoder_anomally.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Veri Setleri

Ham veri dosyalari:

- `data/raw/Dataset_Year_2019.xlsx`
- `data/raw/Dataset_Year_2020.xlsx`

Her iki ham dosyada da `15873` gozlem bulunmaktadir. Veri setleri trafo gucu, koruma durumu, yildirim yogunlugu, kullanici tipi, kurulum tipi, kullanici sayisi, ag uzunlugu ve benzeri trafo/sebeke karakteristiklerini icermektedir.

Ham veri kolonlari:

```text
LOCATION
POWER
SELF-PROTECTION
Average earth discharge density DDT [Rays/km^2-ano]
Maximum ground discharge density DDT [Rays/km^2-ano]
Burning rate [Failures/year]
Criticality according to previous study for ceramics level
Removable connectors
Type of clients
Number of users
Electric power not supplied EENS [kWh]
Type of installation
Air network
Circuit Queue
km of network LT
Burned transformers 2019 / Burned transformers 2020
```

Orijinal `Burned transformers` kolonlari bu calismada hedef degisken olarak kullanilmamistir. Bu kolonlar yaniltici kabul edildigi icin preprocessing asamasinda dogrudan silinir.

### Islenmis Veri Dosyalari

| Dosya | Boyut | Icerik | Target Durumu |
|---|---:|---|---|
| `processed_dataset_2019.csv` | `15873 x 29` | Temizlenmis ve feature engineering uygulanmis 2019 feature seti | Yok |
| `processed_dataset_2020.csv` | `15873 x 29` | Temizlenmis ve feature engineering uygulanmis 2020 feature seti | Yok |
| `2019_pseudo.csv` | `15873 x 30` | 2019 feature seti ve pseudo `target` | Var |
| `2020_pseudo.csv` | `15873 x 30` | 2020 feature seti ve pseudo `target` | Var |
| `normalized_dataset_2019.csv` | `15873 x 30` | Normalize edilmis final 2019 model verisi | Var |
| `normalized_dataset_2020.csv` | `15873 x 30` | Normalize edilmis final 2020 model verisi | Var |

Pseudo etiket dagilimi:

| Dataset | `target=0` | `target=1` | Toplam | `target=1` Orani |
|---|---:|---:|---:|---:|
| 2019 pseudo | 12736 | 3137 | 15873 | %19.76 |
| 2020 pseudo | 12778 | 3095 | 15873 | %19.50 |

Bu dagilim, orijinal yanma etiketlerindeki sinif dagilimindan farklidir. Bunun nedeni yeni hedefin gercek yanma kaydi degil, algoritmik risk/anomali etiketi olmasidir.

## On Isleme

On isleme adimlari [src/preprocessing.py](src/preprocessing.py) dosyasinda uygulanir.

Baslica adimlar:

- Kolon adlari kucuk harfe cevrilir.
- Ozel karakterler temizlenir.
- Bosluklar `_` karakteri ile degistirilir.
- `Burned transformers 2019` ve `Burned transformers 2020` kolonlari silinir.
- Kategorik degiskenler one-hot encoding ile sayisallastirilir.
- Bazi dusuk kullanilabilirlikteki veya gereksiz kolonlar modelleme disinda birakilir.
- Sayisal degiskenler icin korelasyon matrisleri uretilir.

Dusen kolonlar:

```text
location
maximum_ground_discharge_density_ddt_rayskm2ao
electric_power_not_supplied_eens_kwh
circuit_queue
air_network
```

Feature engineering ile uretilen degiskenler:

| Degisken | Aciklama |
|---|---|
| `power_per_user` | Trafo gucunun kullanici sayisina gore normalize edilmis hali |
| `lightning_risk_score` | Ortalama yildirim yogunlugu ile self-protection bilgisinin etkilesimi |
| `network_density` | LT ag uzunlugunun trafo gucune gore oransal ifadesi |
| `historical_risk_index` | Gecmis yanma orani ile kritiklik seviyesinin etkilesimi |

## Pseudo Labelling

Pseudo etiketleme [src/labelling.py](src/labelling.py) dosyasinda yapilir. Bu asamada `processed_dataset_2019.csv` ve `processed_dataset_2020.csv` dosyalari kullanilir; yani algoritmalar eski yanma kolonlarini hic gormez.

Pseudo labelling sureci:

1. 2019 ve 2020 feature setleri okunur.
2. Boolean kolonlar sayisal tipe donusturulur.
3. `StandardScaler` 2019 feature setine fit edilir ve 2020 feature setine ayni scaler uygulanir.
4. KMeans modeli 2019 verisi uzerinde egitilir.
5. 2020 verisi, 2019 uzerinde egitilen KMeans modeli ile etiketlenir.
6. KMeans icin merkezden ortalama uzakligi daha yuksek olan cluster anomali cluster'i olarak kabul edilir.
7. Isolation Forest 2019 verisi uzerinde egitilir ve 2019/2020 icin anomali tahmini yapar.
8. Nihai pseudo hedef su mantikla uretilir:

```text
target = 1, eger KMeans veya Isolation Forest gozlemi anomali/riskli isaretlediyse
target = 0, aksi halde
```

Bu yaklasim, tek bir algoritmaya bagli kalmadan hem cluster yapisini hem de izolasyon tabanli anomali sinyalini kullanir. Bununla birlikte pseudo etiketler modelleme icin pratik bir hedef saglasa da dogrulanmis saha ariza etiketi olarak ele alinmamalidir.

## Normalizasyon

Normalizasyon [src/normalization.py](src/normalization.py) dosyasinda uygulanir. Normalizasyon pseudo etiketleme sonrasinda, `2019_pseudo.csv` ve `2020_pseudo.csv` dosyalari uzerinde yapilir.

`StandardScaler` yalnizca 2019 verisine fit edilir ve ayni donusum 2020 verisine uygulanir. Bu tercih, ileri-yil test setinden egitim surecine bilgi sizmasini engellemek icin yapilmistir.

Normalize edilen sayisal degiskenler:

```text
power
average_earth_discharge_density_ddt_rayskm2ao
burning_rate__failuresyear
number_of_users
km_of_network_lt
power_per_user
lightning_risk_score
network_density
historical_risk_index
```

One-hot encoded kategorik degiskenler ve pseudo `target` kolonu normalize edilmez.

## Deney Tasarimi

Bu projede zaman temelli bir deney tasarimi kullanilmistir:

| Rol | Veri |
|---|---|
| Egitim ve validation | 2019 verisi |
| Ileri-yil test | 2020 verisi |

Bu kurguda model, onceki yilin veri yapisindan ogrenmekte ve sonraki yilin pseudo risk/anomali etiketleri uzerinde test edilmektedir. Bu tasarim, rastgele train-test ayrimina gore daha gercekci bir genelleme senaryosu sunar.

## Model Analizleri

### Random Forest

Dosya: [src/models/random_forrest.py](src/models/random_forrest.py)

Random Forest modeli, normalize edilmis 2019 verisi ile egitilir ve normalize edilmis 2020 verisi uzerinde test edilir. Model olasilik tahmini uretir; daha sonra `precision_recall_curve` kullanilarak F1 skorunu maksimize eden esik degeri secilir.

Model ayarlari:

```text
n_estimators = 100
max_depth = 10
random_state = 42
```

Test sonucu:

| Metrik | Deger |
|---|---:|
| En iyi threshold | 0.3651 |
| En iyi F1 skoru | 0.9770 |
| Accuracy | 0.9911 |

Confusion matrix:

```text
[[12714    64]
 [   78  3017]]
```

Sinif bazli rapor:

| Sinif | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.99 | 0.99 | 0.99 | 12778 |
| 1 | 0.98 | 0.97 | 0.98 | 3095 |

En etkili ilk 10 degisken:

| Sira | Degisken | Onem |
|---:|---|---:|
| 1 | `type_of_clients_STRATUM 1` | 0.2889 |
| 2 | `type_of_installation_POLE` | 0.1244 |
| 3 | `power` | 0.1031 |
| 4 | `type_of_clients_STRATUM 2` | 0.0735 |
| 5 | `type_of_installation_MACRO WITHOUT ANTI-FRAUD NET` | 0.0648 |
| 6 | `number_of_users` | 0.0613 |
| 7 | `lightning_risk_score` | 0.0442 |
| 8 | `average_earth_discharge_density_ddt_rayskm2ao` | 0.0377 |
| 9 | `removable_connectors` | 0.0357 |
| 10 | `network_density` | 0.0304 |

Yorum: Random Forest, pseudo etiketleri oldukca yuksek basariyla ayirmaktadir. Ozellikle kullanici tipi, kurulum tipi ve trafo gucu degiskenlerinin one cikmasi, pseudo etiketleme mekanizmasinin bu yapisal degiskenlerle guclu iliski kurdugunu gostermektedir. Ancak bu basari, gercek ariza kaydi uzerinde dogrulanmis bir performans olarak yorumlanmamalidir.

### XGBoost

Dosya: [src/models/XG_Boost.py](src/models/XG_Boost.py)

XGBoost modeli, gradyan artirmali karar agaclari ile pseudo hedefi ogrenmek icin kullanilmistir. Model 2019 verisiyle egitilmis ve 2020 verisiyle test edilmistir.

Model ayarlari:

```text
n_estimators = 500
max_depth = 8
learning_rate = 0.01
random_state = 42
```

Test sonucu:

| Metrik | Deger |
|---|---:|
| Accuracy | 0.9915 |

Confusion matrix:

```text
[[12755    23]
 [  112  2983]]
```

Sinif bazli rapor:

| Sinif | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.99 | 1.00 | 0.99 | 12778 |
| 1 | 0.99 | 0.96 | 0.98 | 3095 |

Yorum: XGBoost, false positive sayisini Random Forest'a gore daha dusuk tutmaktadir; buna karsilik `target=1` sinifinda false negative sayisi biraz daha yuksektir. Bu sonuc, XGBoost'un pseudo risk sinifini daha secici tahmin ettigini gostermektedir.

### Autoencoder Anomaly Detection

Dosya: [src/models/autoencoder_anomally.py](src/models/autoencoder_anomally.py)

Autoencoder yaklasimi diger iki modelden farklidir. Bu model supervised classifier gibi dogrudan `target` siniflarini ogrenmek yerine, 2019 egitim verisindeki `target=0` normal gozlemleri yeniden olusturmayi ogrenir. Daha sonra reconstruction error degeri yuksek olan gozlemler anomali/riskli olarak siniflandirilir.

Mimari:

```text
Input -> Dense(32) -> Dense(16) -> Latent(4)
      -> Dense(16) -> Dense(32) -> Output
```

Egitim ozellikleri:

```text
loss = mse
optimizer = adam
epochs = 50
batch_size = 32
early stopping patience = 5
```

Anomali karari:

```text
reconstruction_error > threshold_p95 ise target=1
reconstruction_error <= threshold_p95 ise target=0
```

Son calistirmada egitim sonunda gozlenen temel degerler:

| Deger | Sonuc |
|---|---:|
| Final training loss | 0.01142 |
| Validation error mean | 0.01146 |
| Threshold p95 | 0.04877 |
| Test error mean | 0.05129 |
| Accuracy | 0.9214 |

Confusion matrix:

```text
[[11673  1105]
 [  143  2952]]
```

Sinif bazli rapor:

| Sinif | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9879 | 0.9135 | 0.9493 | 12778 |
| 1 | 0.7276 | 0.9538 | 0.8255 | 3095 |

Yorum: Autoencoder, `target=1` pseudo risk sinifi icin yuksek recall degeri uretmektedir. Bu, riskli/anomali olarak isaretlenen ornekleri kacirmama egiliminin guclu oldugunu gosterir. Buna karsilik precision degeri Random Forest ve XGBoost'a gore daha dusuktur; yani Autoencoder daha fazla normal gozlemi riskli olarak isaretlemektedir. Risk tarama problemlerinde yuksek recall tercih edilebilir, ancak bu durum daha fazla false positive maliyeti dogurur.

## Model Karsilastirmasi

| Model | Accuracy | Target=1 Precision | Target=1 Recall | Target=1 F1 | Yorum |
|---|---:|---:|---:|---:|---|
| Random Forest | 0.9911 | 0.98 | 0.97 | 0.98 | Dengeli ve yuksek performans |
| XGBoost | 0.9915 | 0.99 | 0.96 | 0.98 | Daha dusuk false positive, biraz daha yuksek false negative |
| Autoencoder | 0.9214 | 0.7276 | 0.9538 | 0.8255 | Riskli sinifi yakalamada guclu, false positive sayisi daha yuksek |

Genel olarak supervised modeller olan Random Forest ve XGBoost, pseudo etiketleri cok yuksek basariyla ogrenmektedir. Autoencoder ise daha dusuk genel accuracy uretmesine ragmen `target=1` sinifi icin yuksek recall saglamaktadir. Bu nedenle Autoencoder, kesin siniflandirmadan cok risk tarama veya onceliklendirme amacli kullanima daha uygundur.

## Results Klasoru Ciktilari

`results/` klasoru model ve veri analizi ciktilarini icerir.

| Dosya | Aciklama | Guncel Pipeline Durumu |
|---|---|---|
| `2019 Correlation Matrix.png` | 2019 feature seti icin sayisal korelasyon matrisi | Guncel |
| `2020 Correlation Matrix.png` | 2020 feature seti icin sayisal korelasyon matrisi | Guncel |
| `tsne_2019.png` | 2019 normalize pseudo verisinin 2 boyutlu t-SNE izdusumu | Guncel |
| `tsne_2020.png` | 2020 normalize pseudo verisinin 2 boyutlu t-SNE izdusumu | Guncel |
| `RandomForest-ConfusionMatrix.png` | Random Forest test confusion matrix grafigi | Guncel |
| `XG-Boost-ConfusionMatrix.png` | XGBoost test confusion matrix grafigi | Guncel |
| `AutoEncoder-ConfusionMatrix.png` | Autoencoder test confusion matrix grafigi | Guncel |
| `burned_transformers_2019_correlation.png` | Eski orijinal target korelasyon analizi | Legacy/eski deney |
| `burned_transformers_2020_correlation.png` | Eski orijinal target korelasyon analizi | Legacy/eski deney |

Not: `burned_transformers_2019_correlation.png` ve `burned_transformers_2020_correlation.png` dosyalari eski hedef kolonlari kullanilarak uretilmis onceki analiz ciktilaridir. Mevcut pipeline'da orijinal `Burned transformers` kolonlari silindigi icin bu iki gorsel guncel modelleme sonucunun parcasi olarak yorumlanmamalidir.

## Calistirma

Proje Python `>=3.12` ile calisir. Bagimliliklar [pyproject.toml](pyproject.toml) ve [uv.lock](uv.lock) dosyalarinda tanimlidir.

Ortam kurulumu:

```bash
uv sync
```

Sanal ortami aktive etmek icin:

```bash
source .venv/bin/activate
```

Onerilen calistirma sirasi:

```bash
python src/preprocessing.py
python src/labelling.py
python src/normalization.py
python src/check_distrubition.py
python src/models/random_forrest.py
python src/models/XG_Boost.py
python src/models/autoencoder_anomally.py
```

Mevcut lokal ortamda dogrudan:

```bash
.venv/bin/python src/preprocessing.py
.venv/bin/python src/labelling.py
.venv/bin/python src/normalization.py
.venv/bin/python src/check_distrubition.py
.venv/bin/python src/models/random_forrest.py
.venv/bin/python src/models/XG_Boost.py
.venv/bin/python src/models/autoencoder_anomally.py
```

TensorFlow calisirken GPU kutuphaneleri bulunamazsa CPU ile calismaya devam eder. `Cannot dlopen some GPU libraries` veya `Skipping registering GPU devices` gibi mesajlar, GPU kullanilmadigini belirtir; modelin CPU uzerinde calismasini engellemez.

## Bagimliliklar

Baslica kutuphaneler:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `xgboost`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `openpyxl`

## Akademik Yorum ve Sinirlar

Bu calisma, guvenilir olmayan dogrudan ariza etiketleri yerine pseudo etiketleme yaklasimini kullanarak trafo risk/anomali tahmini yapmaktadir. Bu tercih, modelleme acisindan tutarli bir hedef degisken saglar; ancak uretilen `target` kolonunun saha tarafindan dogrulanmis gercek ariza etiketi olmadigi unutulmamalidir.

Baslica sinirlar:

- `target=1`, kesin ariza veya yanma anlamina gelmez; algoritmik anomali/risk isaretidir.
- Model metrikleri pseudo hedefe gore hesaplanmistir.
- Sonuclar gercek saha ariza tahmini olarak yorumlanmadan once uzman dogrulamasi veya guvenilir ariza kayitlari ile kalibre edilmelidir.
- Veri DGA zaman serisi degildir; bu nedenle dogrudan RUL veya "kac gun sonra ariza olur" tahmini icin yeterli degildir.
- 2019-2020 ileri-yil tasarimi zamansal genelleme acisindan daha uygundur, ancak yalnizca iki yil bulundugu icin uzun donem genelleme analizi sinirlidir.
- Scriptlerde mutlak dosya yollari kullanilmaktadir. Farkli bir ortamda calisma icin dosya yollari proje kokune gore dinamik hale getirilebilir.

Sonuc olarak bu proje, trafo verileri uzerinde pseudo labelling, klasik denetimli modeller ve autoencoder tabanli anomali tespitini birlikte degerlendiren deneysel bir risk modelleme calismasidir. Random Forest ve XGBoost pseudo etiketleri yuksek basariyla ogrenirken, Autoencoder ozellikle riskli sinifi yakalamaya odaklanan daha duyarli bir anomali tespit alternatifi sunmaktadir.
