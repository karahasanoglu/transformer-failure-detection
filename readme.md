# Dağıtım Transformatörleri İçin Kestirimci Bakım

Bu proje, dağıtım transformatörlerinde arıza/risk durumunu veri madenciliği ve makine öğrenmesi yöntemleriyle inceleyen akademik nitelikli bir kestirimci bakım çalışmasıdır. Çalışma, 2019 ve 2020 yıllarına ait trafo verilerinden mühendislik özellikleri üretir, denetimsiz yöntemlerle pseudo etiket oluşturur ve bu etiketleri SVM modeliyle sınıflandırır.

Referans alınan problem alanı:

`Vita, V.; Fotis, G.; Chobanov, V.; Pavlatos, C.; Mladenov, V. Predictive Maintenance for Distribution System Operators in Increasing Transformers' Reliability. Electronics 2023, 12, 1356.`

Bu repository doğrudan makalenin birebir kopyası değildir. Kod tabanındaki güncel akış, ham Excel dosyalarındaki `Burned transformers 2019/2020` sütunlarını model hedefi olarak kullanmaz; bu sütunları olası hedef sızıntısını ve doğrudan etiket bağımlılığını azaltmak amacıyla ön işleme aşamasında çıkarır. Ardından `KMeans` ve `IsolationForest` ile pseudo risk etiketi üretir.

## Projenin Amacı

Projenin temel amacı, dağıtım trafolarına ait teknik ve çevresel değişkenlerden arıza riski taşıyan gözlemleri ayırabilecek açıklanabilir bir deney hattı kurmaktır. Bu kapsamda:

- ham 2019 ve 2020 Excel verileri temizlenir,
- kategorik değişkenler sayısallaştırılır,
- trafo yükü, yıldırım riski, şebeke yoğunluğu ve geçmiş risk etkileşimi gibi yeni özellikler üretilir,
- gerçek `burned_transformers` sütunları çıkarıldıktan sonra denetimsiz pseudo etiketleme yapılır,
- normalize edilmiş veriler üzerinde RBF çekirdekli SVM modelleri eğitilir,
- sonuçlar confusion matrix görselleri ve sınıflandırma raporları ile değerlendirilir.

Bu nedenle proje bir üretim sistemi değil, akademik/prototip seviyesinde bir kestirimci bakım modelleme çalışmasıdır.

## Veri Seti

Ham veriler `data/raw/` dizinindedir:

```text
data/raw/
├── Dataset_Year_2019.xlsx
└── Dataset_Year_2020.xlsx
```

Her iki dosya da `15873` satır ve `16` ham sütundan oluşur. Ham sütunlar şunlardır:

- `LOCATION`
- `POWER`
- `SELF-PROTECTION`
- `Average earth discharge density DDT [Rays/km^2-año]`
- `Maximum ground discharge density DDT [Rays/km^2-año]`
- `Burning rate  [Failures/year]`
- `Criticality according to previous study for ceramics level`
- `Removable connectors`
- `Type of clients`
- `Number of users`
- `Electric power not supplied EENS [kWh]`
- `Type of installation`
- `Air network`
- `Circuit Queue`
- `km of network LT:`
- `Burned transformers 2019` veya `Burned transformers 2020`

Kodda sütun adları önce küçük harfe çevrilir, özel karakterler temizlenir ve boşluklar `_` ile değiştirilir. Örneğin `SELF-PROTECTION`, `selfprotection`; `km of network LT:`, `km_of_network_lt`; `Burned transformers 2019`, `burned_transformers_2019` biçimine dönüştürülür.

## Ön İşleme

Ön işleme akışı [src/preprocessing.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/preprocessing.py:1) içinde tanımlıdır. Betik iki ham Excel dosyasını okur, sütun adlarını standartlaştırır ve yıl bazlı gerçek yanma etiketlerini veri setinden çıkarır:

- 2019 için `burned_transformers_2019`
- 2020 için `burned_transformers_2020`

Bu tercih önemlidir: güncel model hattı gerçek hedef etiketleriyle denetimli arıza tahmini yapmaz. Bunun yerine, gerçek hedef sütunları çıkarıldıktan sonra kalan özelliklerden pseudo risk etiketi üretir.

Ön işleme sırasında oluşturulan mühendislik özellikleri:

- `power_per_user = power / (number_of_users + 1)`
- `lightning_risk_score = average_earth_discharge_density_ddt_rayskm2ao * (2 - selfprotection)`
- `network_density = km_of_network_lt / (power + 1)`
- `historical_risk_index = burning_rate__failuresyear * criticality_according_to_previous_study_for_ceramics_level`

Ardından `type_of_clients` ve `type_of_installation` değişkenleri `pd.get_dummies` ile one-hot encoded sütunlara dönüştürülür. Ön işleme sonunda model dışı bırakılan sütunlar:

- `location`
- `maximum_ground_discharge_density_ddt_rayskm2ao`
- `electric_power_not_supplied_eens_kwh`
- `circuit_queue`
- `air_network`

2020 verisinde ayrıca `type_of_installation_POLE WITH ANTI-FRAU NET` sütunu `type_of_installation_POLE WITH ANTI-FRAUD NET` olarak yeniden adlandırılır. Bu işlem, 2019 ve 2020 işlenmiş veri setlerinin aynı kolon şemasını paylaşmasını sağlar.

Ön işleme çıktıları:

```text
data/processed/
├── processed_dataset_2019.csv
└── processed_dataset_2020.csv
```

Her iki işlenmiş dosya da `15873 x 29` boyutundadır ve henüz `target` sütunu içermez.

## Korelasyon Analizi

Ön işleme betiği, sayısal değişkenler için korelasyon matrislerini üretir ve `results/` altına kaydeder:

```text
results/
├── 2019 Correlation Matrix.png
└── 2020 Correlation Matrix.png
```

Bu grafikler, özellikler arasındaki doğrusal ilişkileri incelemek için kullanılır. Özellikle üretilen mühendislik özelliklerinin ham değişkenlerle ne ölçüde ilişkili olduğunu görmek açısından yararlıdır. Ancak korelasyon matrisi tek başına nedensellik veya model başarısı kanıtı değildir; yalnızca keşifsel veri analizi çıktısı olarak yorumlanmalıdır.

## Pseudo Etiketleme

Pseudo etiketleme akışı [src/labelling.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/labelling.py:1) içindedir. Bu adımda `processed_dataset_2019.csv` ve `processed_dataset_2020.csv` okunur, boolean sütunlar `0/1` tamsayılarına çevrilir ve tüm özellikler `StandardScaler` ile ölçeklenir.

Ölçekleme bu aşamada şu şekilde yapılır:

- scaler 2019 işlenmiş verisine `fit_transform` uygulanarak öğrenilir,
- aynı scaler 2020 işlenmiş verisine yalnızca `transform` olarak uygulanır.

Bu tercih, 2020 verisini 2019 referans dağılımına göre değerlendirmek anlamına gelir. Zaman sırası mantığı açısından makuldür; çünkü gelecekteki yılın istatistikleriyle geçmiş yılın ölçekleyicisini yeniden öğrenmekten kaçınılır.

Pseudo target üretimi iki denetimsiz sinyalin birleşimiyle yapılır:

1. `KMeans(n_clusters=2, random_state=42, n_init=10)`
2. `IsolationForest(contamination=0.05, random_state=42)`

KMeans tarafında model 2019 ölçeklenmiş verisine fit edilir. 2020 verisi için aynı KMeans modeliyle cluster tahmini alınır. 2019 verisinde her cluster için merkeze uzaklık ortalaması hesaplanır; ortalama uzaklığı daha yüksek olan cluster anomali/risk cluster'ı olarak kabul edilir.

IsolationForest tarafında model yine 2019 ölçeklenmiş verisine fit edilir ve 2020 verisine predict uygulanır. IsolationForest çıktısında `-1` olan gözlemler anomali kabul edilir.

Son pseudo hedef:

```text
target = 1 if KMeans_anomaly == 1 or IsolationForest_anomaly == 1 else 0
```

Yani iki yöntemden herhangi biri gözlemi riskli işaretlerse nihai pseudo etiket `1` olur.

Pseudo etiketleme çıktıları:

```text
data/processed/
├── 2019_pseudo.csv
└── 2020_pseudo.csv
```

Pseudo target dağılımları:

| Veri seti | target=0 | target=1 | Risk oranı |
|---|---:|---:|---:|
| 2019 | 12736 | 3137 | 19.76% |
| 2020 | 12778 | 3095 | 19.50% |

Bu oranlar, gerçek arıza oranı olarak değil, denetimsiz modelin riskli/anormal gördüğü gözlem oranı olarak yorumlanmalıdır.

## Normalizasyon

Normalizasyon akışı [src/normalization.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/normalization.py:1) içinde tanımlıdır. Bu betik pseudo etiketlenmiş dosyaları okur ve yalnızca belirli sayısal değişkenleri normalize eder.

Normalize edilen sayısal sütunlar:

- `power`
- `average_earth_discharge_density_ddt_rayskm2ao`
- `burning_rate__failuresyear`
- `number_of_users`
- `km_of_network_lt`
- `power_per_user`
- `lightning_risk_score`
- `network_density`
- `historical_risk_index`

Kullanılan yöntem `sklearn.preprocessing.StandardScaler` yöntemidir. Standartlaştırma formülü:

```text
z = (x - mean) / standard_deviation
```

Bu betikte scaler 2019 pseudo verisine `fit_transform` ile uygulanır; 2020 pseudo verisi aynı scaler ile `transform` edilir. Böylece 2020 özellikleri, 2019 dağılımı referans alınarak standartlaştırılır.

Normalizasyon sırasında boolean one-hot sütunlar `0/1` tamsayılarına dönüştürülür. `target` sütunu normalize edilmez ve sınıf etiketi olarak korunur.

Normalizasyon çıktıları:

```text
data/processed/
├── normalized_dataset_2019.csv
└── normalized_dataset_2020.csv
```

Her iki dosya da `15873 x 30` boyutundadır: `29` özellik ve `1` pseudo target sütunu.

## Modelleme

Model betikleri `src/models/` dizinindedir:

```text
src/models/
├── SVM_2019.py
├── SVM_2020.py
└── SVM-2019-2020.py
```

Tüm model betikleri normalize edilmiş CSV dosyalarını kullanır. Modelleme yaklaşımı:

- hedef değişken: `target`
- özellikler: `target` hariç tüm sütunlar
- model: `sklearn.svm.SVC`
- kernel: `rbf`
- class balancing: `class_weight="balanced"`
- `C=1.0`
- `gamma="scale"`; birleşik 2019-2020 betiğinde gamma parametresi verilmediği için sklearn varsayılanı yine `scale` olur
- train/test ayrımı: `train_test_split(test_size=0.2, random_state=42)`

Not: Mevcut model betiklerinde `stratify=y` kullanılmıyor. Veri setindeki pseudo target oranı yaklaşık %20 olduğu için bölünme yine dengeli görünmektedir; ancak akademik olarak daha kontrollü değerlendirme için stratified split eklenmesi önerilir.

### 2019 SVM Çıktısı

[src/models/SVM_2019.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/models/SVM_2019.py:1) betiği `normalized_dataset_2019.csv` üzerinde çalışır.

Confusion matrix:

```text
[[2490   63]
 [   3  619]]
```

Sınıflandırma raporu:

| Sınıf | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.9988 | 0.9753 | 0.9869 | 2553 |
| 1 | 0.9076 | 0.9952 | 0.9494 | 622 |

Genel doğruluk `0.9792`, macro F1 `0.9682`, weighted F1 `0.9796` olarak raporlanmıştır.

Üretilen görsel:

```text
results/2019-SVM-confusion_matrix.png
```

### 2020 SVM Çıktısı

[src/models/SVM_2020.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/models/SVM_2020.py:1) betiği `normalized_dataset_2020.csv` üzerinde çalışır.

Confusion matrix:

```text
[[2491   65]
 [   6  613]]
```

Sınıflandırma raporu:

| Sınıf | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.9976 | 0.9746 | 0.9859 | 2556 |
| 1 | 0.9041 | 0.9903 | 0.9453 | 619 |

Genel doğruluk `0.9776`, macro F1 `0.9656`, weighted F1 `0.9780` olarak raporlanmıştır.

Üretilen görsel:

```text
results/2020-SVM-confusion_matrix.png
```

### Birleşik 2019-2020 SVM Çıktısı

[src/models/SVM-2019-2020.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/models/SVM-2019-2020.py:1) betiği 2019 ve 2020 normalize veri setlerini birleştirir. Betik geçici olarak `yil` sütunu ekler, fakat model eğitiminde `target` ve `yil` sütunlarını özelliklerden çıkarır.

Confusion matrix:

```text
[[4995   95]
 [   9 1251]]
```

Sınıflandırma raporu:

| Sınıf | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 1.00 | 0.98 | 0.99 | 5090 |
| 1 | 0.93 | 0.99 | 0.96 | 1260 |

Genel doğruluk yaklaşık `0.98`, macro F1 yaklaşık `0.97`, weighted F1 yaklaşık `0.98` olarak raporlanmıştır.

Üretilen görsel:

```text
results/2019-2020-SVM-confusion_matrix.png
```

## Results Klasörü

Güncel `results/` dizini şu çıktıları içerir:

```text
results/
├── 2019 Correlation Matrix.png
├── 2020 Correlation Matrix.png
├── 2019-SVM-confusion_matrix.png
├── 2020-SVM-confusion_matrix.png
└── 2019-2020-SVM-confusion_matrix.png
```

Korelasyon matrisleri ön işleme/keşifsel analiz çıktısıdır. SVM confusion matrix görselleri ise modelin test ayrımındaki sınıflandırma davranışını özetler. Confusion matrix satırları gerçek sınıfı, sütunları tahmin edilen sınıfı temsil eder:

- sol üst: doğru negatif
- sağ üst: yanlış pozitif
- sol alt: yanlış negatif
- sağ alt: doğru pozitif

Bu projede pozitif sınıf `target=1`, yani pseudo etiketleme tarafından riskli/anormal kabul edilen trafo kaydını ifade eder.

## Çalıştırma Sırası

Projede mevcut kod yapısı betik tabanlıdır. Temiz bir akış için önerilen sıra:

```bash
.venv/bin/python src/preprocessing.py
.venv/bin/python src/labelling.py
.venv/bin/python src/normalization.py
.venv/bin/python src/models/SVM_2019.py
.venv/bin/python src/models/SVM_2020.py
.venv/bin/python src/models/SVM-2019-2020.py
```

Sistemde `python` komutu tanımlı değilse `.venv/bin/python` veya uygun ortamda `python3` kullanılmalıdır.

## Kurulum

Proje `pyproject.toml` içinde Python `>=3.12` ve şu temel bağımlılıkları tanımlar:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `openpyxl`
- `imblearn`
- `tensorflow`

Sanal ortam mevcutsa:

```bash
.venv/bin/python -m pip install -e .
```

Alternatif olarak temel paketler doğrudan kurulabilir:

```bash
python3 -m pip install pandas numpy scikit-learn matplotlib seaborn openpyxl imbalanced-learn
```

## Dizin Yapısı

```text
transformer_predictive_maintenance/
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
│   ├── 2019-SVM-confusion_matrix.png
│   ├── 2020-SVM-confusion_matrix.png
│   └── 2019-2020-SVM-confusion_matrix.png
├── src/
│   ├── preprocessing.py
│   ├── labelling.py
│   ├── normalization.py
│   └── models/
│       ├── SVM_2019.py
│       ├── SVM_2020.py
│       └── SVM-2019-2020.py
├── pyproject.toml
├── readme.md
└── readme_en.md
```

## Metodolojik Değerlendirme

Model sonuçları yüksek görünmektedir; ancak bu başarı doğrudan gerçek arıza tahmini başarısı olarak okunmamalıdır. Bunun temel nedeni, SVM modelinin gerçek `Burned transformers` etiketlerini değil, KMeans ve IsolationForest tarafından üretilen pseudo `target` etiketlerini öğrenmesidir. Dolayısıyla SVM, büyük ölçüde denetimsiz anomali tanımını tekrar eden bir sınıflandırıcı gibi davranır.

Bu yaklaşım akademik olarak yararlıdır; çünkü:

- gerçek etiket olmadan risk grubu üretme fikrini test eder,
- denetimsiz etiketlerin sınıflandırılabilir olup olmadığını gösterir,
- farklı yılların aynı özellik uzayında benzer risk yapısı taşıyıp taşımadığını incelemeye yardımcı olur.

Ancak sınırlamaları da açıktır:

- pseudo etiketler gerçek saha arızasıyla doğrulanmadıkça operasyonel karar için yeterli değildir,
- `train_test_split` zaman bağımlılığını dikkate almaz,
- model betiklerinde stratified split kullanılmamıştır,
- tüm dosya yolları kod içinde mutlak path olarak yazılmıştır; farklı makinede çalıştırmak için göreli path'e çevrilmesi gerekir,
- 2021 gibi yeni bir yıl için doğrudan tahmin yapan ayrı bir production pipeline bulunmamaktadır,
- sıcaklık, yük geçmişi, yağ analizi, bakım kayıtları ve çevresel zaman serileri gibi kritik kestirimci bakım değişkenleri veri setinde yoktur.

## Sonuç

Bu proje, dağıtım transformatörleri için kestirimci bakım problemini uçtan uca deneysel bir hat olarak ele alır: ham veriden özellik mühendisliğine, pseudo etiketlemeden normalizasyona ve SVM tabanlı sınıflandırmaya kadar tüm adımlar kodlanmıştır. Güncel çıktılar, pseudo risk sınıfının SVM ile yüksek başarıyla ayrıştırılabildiğini göstermektedir.

Yine de elde edilen sonuçlar gerçek arıza tahmini yerine pseudo anomali/risk sınıflandırması olarak değerlendirilmelidir. Gerçek saha kullanımı için modelin gerçek arıza etiketleriyle, zaman temelli validasyonla ve daha zengin operasyonel sensör/bakım verileriyle yeniden doğrulanması gerekir.
