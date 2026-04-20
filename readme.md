# Dağıtım Transformatörleri İçin Kestirimci Bakım Projesi

Bu repo, dağıtım transformatörlerinde arıza riskini öngörmek için geliştirilen bir makine öğrenmesi prototipidir. Çalışma, aşağıdaki makaledeki problem tanımına yakın bir senaryoyu yeniden üretmeyi hedefler:

`Vita, V.; Fotis, G.; Chobanov, V.; Pavlatos, C.; Mladenov, V. Predictive Maintenance for Distribution System Operators in Increasing Transformers’ Reliability. Electronics 2023, 12, 1356.`

Bu projede makaledeki genel problem korunmuştur, ancak uygulama birebir kopya değildir:

- Bu repoda `burned transformers` etiketleri doğrudan kullanılmaktadır.
- Bu nedenle makaledeki `k-means -> SVM` hibrit etiketleme hattı ana akıştan çıkarılmıştır.
- Ana modelleme hattı, denetimli öğrenme (`supervised learning`) ile kurulmuştur.

Bu yaklaşım daha dürüst ve daha izlenebilir bir değerlendirme sağlar; çünkü elimizde gerçek hedef etiketi zaten vardır.

## Projenin Amacı

Amaç, 2019 ve 2020 yıllarına ait trafo verilerini kullanarak:

- arızalı (`1`) ve arızasız (`0`) trafoları ayırt etmek,
- arızalı sınıfın yakalanma başarısını artırmak,
- 2021 için sayısal bir risk projeksiyonu üretmek,
- elde edilen sonuçları makaledeki raporlanan bulgularla karşılaştırmaktır.

Bu proje bir üretim sistemi değil, bir `proof-of-concept / akademik prototip` olarak düşünülmelidir.

## Kullanılan Veri

Ham veriler `data/raw/` altında tutulur:

- `Dataset_Year_2019.xlsx`
- `Dataset_Year_2020.xlsx`

Her dosyada ilgili yıl için gerçek hedef sütunu bulunur:

- 2019 için `Burned transformers 2019`
- 2020 için `Burned transformers 2020`

Projede kullanılan özellikler, makaledeki temel değişken mantığına yakın tutulmuştur. Ana sütunlar şunlardır:

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

Not:
- Projede ek özellik mühendisliği fonksiyonları da mevcuttur.
- Ancak ana `main.py` akışı, makaledeki supervised SVM deneyiyle daha karşılaştırılabilir olmak için daha sade bir özellik seti kullanır.

## Güncel Modelleme Akışı

Mevcut ana akış `main.py` içinde çalışır ve şu adımları izler:

1. 2019 veya 2020 veri seti okunur.
2. Hedef sütunu dosya adına göre otomatik seçilir.
3. Özellikler ölçeklenir (`StandardScaler`).
4. Veri `train=14873`, `test=1000` olacak şekilde stratified olarak bölünür.
5. Eğitim seti içinden ayrıca bir validation alt kümesi ayrılır.
6. `RBF SVM` modeli `class_weight='balanced'` ile eğitilir.
7. Karar eşiği (`threshold`) validation seti üzerinde optimize edilir.
8. Test setinde şu metrikler raporlanır:
   - Accuracy
   - Precision
   - Recall
   - F1
   - Specificity
   - Balanced Accuracy
   - ROC-AUC
   - PR-AUC

Ardından her yıl için iki SVM varyantı kıyaslanır:

- `Full SVM`: seçili tüm özelliklerle
- `Reduced SVM`: aşağıdaki özellikler çıkarıldıktan sonra

Çıkarılan özellikler:

- `LOCATION`
- `POWER`
- `REMOVABLE_CONNECTORS`
- `ENERGY_NOT_SUPPLIED`
- `AIR_NETWORK`
- `CIRCUIT_QUEUE`

Her yıl için daha yüksek `F1` değerine sahip model, o yılın tercih edilen modeli olarak işaretlenir.

## 2021 Projeksiyonu Nasıl Yapılıyor

Projede gerçek 2021 veri dosyası yoktur. Bu nedenle 2021 için doğrudan trafo bazlı tahmin değil, bir `risk projeksiyonu` yapılır.

Güncel projeksiyon mantığı:

1. 2019 ve 2020 verileri birleştirilir.
2. Birleşik veri üzerinden ayrı bir SVM modeli eğitilir.
3. Bu birleşik model için validation tabanlı threshold seçilir.
4. Model 2019 ve 2020 tam filoları üzerinde uygulanır.
5. Elde edilen risk oranlarının ortalaması alınır.
6. Bu ortalama oran `15,873` trafoya yansıtılarak 2021 için projeksiyon üretilir.

Bu sayı:

- kesin arıza sayısı değildir,
- modelin yüksek riskli gördüğü trafo sayısına dayalı bir tahmindir,
- bu yüzden dikkatli yorumlanmalıdır.

README yazıldığı sıradaki son karşılaştırmada:

- proje tahmini: `1275`
- makale referansı: `852`

Aradaki fark, hem model tercihlerinden hem de projeksiyon mantığındaki yorum farkından kaynaklanabilir.

## Son Gözlenen Sonuçlar

Son kaydedilen koşuda öne çıkan sonuçlar:

### 2019

- En iyi model: `Reduced SVM`
- `F1 = 0.3467`
- `Recall = 0.5098`
- `Accuracy = 0.7164`

### 2020

- En iyi model: `Full SVM`
- `F1 = 0.2710`
- `Recall = 0.5250`
- `Accuracy = 0.7135`

Bu sonuçlar şunu gösterir:

- Model artık arızalı sınıfı anlamlı biçimde yakalayabilmektedir.
- Ancak `precision` düşüktür; yani yanlış alarm sayısı hâlâ yüksektir.
- Bu yüzden proje günlük operasyon için doğrudan karar sistemi olmaktan çok, `erken uyarı / risk önceliklendirme` aracı olarak daha uygundur.

## Makale ile İlişki

Bu proje, söz konusu makaledeki problemi ve veri mantığını temel alır; fakat onu eleştirel biçimde yeniden uygular.

Makaledeki temel hat:

- veri ön işleme
- normalization
- feature seçimi
- `k-means`
- `SVM`
- accuracy odaklı değerlendirme

Bu repoda yapılan farklar:

- gerçek etiket olduğu için `k-means` ana boru hattından çıkarılmıştır
- accuracy tek başına kullanılmamaktadır
- validation tabanlı threshold tuning eklenmiştir
- `PR-AUC`, `balanced accuracy` ve `specificity` gibi daha dürüst metrikler raporlanmaktadır

## Makaledeki Eksik veya Belirsiz Noktalar

Makale yararlı bir referans olsa da bazı metodolojik belirsizlikler içerir:

1. Ölçekleme tekniği açık belirtilmiyor
- Metin yalnızca “normalization” yapıldığını söylüyor.
- `StandardScaler`, `MinMaxScaler` veya başka bir teknik açıkça yazılmıyor.

2. Accuracy fazla merkezde
- Dengesiz veri yapısına rağmen ana vurgu accuracy üzerindedir.
- Pozitif sınıf için gerçek saha başarısını görmek açısından bu yeterli değildir.

3. `k-means + SVM` geçişi tam şeffaf değil
- Küme etiketlerinin ground truth ile nasıl ilişkilendirildiği çok net anlatılmıyor.
- Bu, veri sızıntısı veya değerlendirme bulanıklığı riski doğurabilir.

4. Arızalı sınıf performansı yeterince şeffaf raporlanmıyor
- Precision, recall ve F1 vurgusu sınırlı.
- Yüksek accuracy, anomaly detection başarısını olduğundan iyi gösterebilir.

5. 2021 sayısal tahmini güçlü bir iddia olsa da veri kısıtları büyük
- Sadece iki yıllık veri vardır.
- Sıcaklık, yağ seviyesi, overload, bakım geçmişi gibi kritik değişkenler yoktur.

## Bu Projede Bulunan Ana Kısıtlar

Bu repoda da benzer veri kısıtları vardır:

- Yalnızca iki yıl veri bulunur.
- Sınıf dengesizliği yüksektir.
- Operasyonel bakım verileri yoktur.
- Yağ analizi, sıcaklık, yük geçmişi, çevresel zaman serisi gibi değişkenler bulunmaz.

Bu nedenle model:

- akademik olarak anlamlı,
- deneysel olarak çalışır,
- fakat gerçek hayatta tek başına güvenilir bakım planlama sistemi sayılmaz.

## Dizin Yapısı

```text
transformer_predictive_maintenance/
├── data/
│   └── raw/
│       ├── Dataset_Year_2019.xlsx
│       └── Dataset_Year_2020.xlsx
├── results/
│   ├── 2019_svm_full.png
│   ├── 2019_svm_reduced.png
│   ├── 2020_svm_full.png
│   ├── 2020_svm_reduced.png
├── src/
│   ├── preprocessing.py
│   ├── svm_model.py
│   ├── metrics_utils.py
│   ├── visualization.py
│   ├── data_analysis.py
├── main.py
└── readme.md
```



## Dosya Açıklamaları

- `main.py`
  Ana yürütücü. 2019 ve 2020 deneylerini çalıştırır, en iyi SVM varyantını seçer ve 2021 projeksiyonunu üretir.

- `src/preprocessing.py`
  Veri okuma, sütun eşleme, özellik oluşturma, ölçekleme ve train/test bölme işlemleri.

- `src/svm_model.py`
  RBF çekirdekli SVM modeli, eğitim, threshold optimizasyonu ve değerlendirme mantığı.

- `src/metrics_utils.py`
  Ortak metrik hesapları ve eşik seçimi yardımcı fonksiyonları.

- `src/visualization.py`
  Karmaşıklık matrisi görselleştirmeleri.



## Kurulum

Sanal ortam içinde ya da doğrudan şu paketler yeterlidir:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn openpyxl
```

## Çalıştırma

Ana pipeline’ı çalıştırmak için:

```bash
python main.py
```

Çalışma sonunda:

- her yıl için confusion matrix görselleri
- model metrikleri
- 2021 risk projeksiyonu

terminalde ve `results/` klasöründe üretilir.

## Sonuç

Bu proje, makaledeki problemi daha şeffaf bir değerlendirme çerçevesiyle yeniden kurar. Sonuçlar, modelin tamamen başarısız olmadığını; ancak veri setinin sınırlı açıklayıcılığı nedeniyle gerçek hayatta ancak karar destek / risk sıralama aracı olarak kullanılabileceğini göstermektedir.
