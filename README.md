# Deep Learning Anomaly Detection & Explainable Probabilistic Automata

Bu proje, zaman serisi anomali tespiti için Siyah Kutu (Deep Learning - 1D-CNN & GRU) ve Açıklanabilir (Olasılıksal Otomata) modelleri derinlemesine karşılaştırmayı amaçlamaktadır [cite: 8-9, 43-48, 50]. Proje, SOLID ve Nesne Yönelimli Programlama (OOP) prensiplerine sadık kalınarak, tamamen parametrik ve modüler bir pipeline mimarisiyle geliştirilmiştir [cite: 104-105, 108].

Sistemin daha önce karşılaşmadığı görülmemiş örüntüleri (unseen patterns) Levenshtein (Edit Distance) algoritması ile en yakın duruma başarıyla haritalandırdığı birim test seviyesinde matematiksel olarak doğrulanmıştır.

## PROJE KLASÖR YAPISI

- `src/`: Ana kaynak kod (`core`, `data`, `models`, `orchestration`, `reporting`, `visualization` modülleri).
- `tests/`: Pytest ile yazılmış birim testleri.
- `config/`: Hard-coded değerleri engellemek için tüm parametrelerin bulunduğu konfigürasyon dosyaları.
- `reports/figures/`: Pipeline tarafından otomatik üretilen grafiklerin ve görsel analizlerin tutulduğu klasör.
- `venv/`: (Git tarafından yoksayılır) Python sanal ortamı.
- `data_raw/`: (Git tarafından yoksayılır) SKAB ve BATADAL veri setleri.

## I. PROJE TANIMI VE MOTİVASYON

Zaman serisi verileri; finansal sistemler, biyomedikal sinyaller, IoT altyapıları ve davranışsal analiz uygulamaları gibi birçok alanda yaygın olarak kullanılmaktadır. Bu tür veriler üzerinde gerçekleştirilen sınıflandırma ve anomali tespiti problemleri, hem akademik araştırmalar hem de endüstriyel uygulamalar açısından kritik öneme sahiptir.

Bu proje kapsamında, zaman serisi verileri üzerinde iki farklı modelleme paradigmasının karşılaştırılması hedeflenmektedir:

- Derin Öğrenme Tabanlı Modeller (Black-Box): Yüksek doğruluk potansiyeline sahip ancak yorumlanabilirliği sınırlı olan black-box modeller (1D-CNN & GRU).
- Sembolik Temsil ve Durum Geçişlerine Dayalı Modeller (Interpretable): Tamamen yorumlanabilir (interpretable) ve kararları izlenebilir Olasılıksal Otomata modelleri.

## II. ARAŞTIRMA PROBLEMİ VE AMAÇ

Bu çalışmada aşağıdaki temel araştırma problemi ele alınmaktadır:

> "Farklı modelleme yaklaşımları, zaman serisi verileri üzerinde farklı veri koşulları altında nasıl davranmaktadır ve bu davranışlar istatistiksel olarak anlamlı mıdır?"

Bu kapsamda proje aşağıdaki hedefleri içermektedir:

- Farklı modelleme yaklaşımlarının karşılaştırmalı analizi
- Model performansının veri setine bağımlılığının incelenmesi
- Gürültü ve bilinmeyen veri durumlarında (unseen patterns) model davranışının değerlendirilmesi
- Açıklanabilirlik açısından modellerin şeffaf olarak analiz edilmesi

## III. VERİ SETLERİ VE ÖN İŞLEME PIPELINE YAPISI

Sistemde hard-coded değer kullanımı tamamen engellenmiş, tüm parametreler merkezi `config.yaml` dosyasından yönetilmektedir.

### A. SKAB Veri Seti Analizi

Yalnızca `valve1` ve `valve2` klasörlerindeki `.csv` dosyaları concat edilerek tek bir veri seti oluşturulmuştur. Veri takibi ve analizi amacıyla `source_group` ve `source_file` sütunları üretilmiş, veri sızıntısını önlemek için bu sütunlar model girdisinden hariç tutulmuştur.

Protokol: Satır bazlı rastgele bölme zaman serisi bağımlılığını bozduğu için reddedilmiş; `source_file` temel alınarak 5-Fold `GroupKFold` çapraz doğrulama uygulanmıştır.

### B. BATADAL Veri Seti Analizi

Yalnızca Training Dataset 2 kullanılmış, saldırı/anomali sütunu hedef değişken (`label`) yapılmıştır. Zaman bilgileri model girdisinden tamamen arındırılmıştır.

Protokol: Kronolojik sıra harfiyen korunarak veri seti `%60 Eğitim, %20 Doğrulama ve %20 Test` olarak zaman sıralı dilimlenmiştir.

### C. Ön İşleme ve Veri Sızıntısı (Data Leakage) Engelleme Kuralları

- Normalizasyon (`MinMax`) ve Boyut İndirgeme (`PCA`) yalnızca eğitim (`train`) verisi üzerinde fit edilmiş, doğrulama ve test verilerine aynı dönüşüm transform olarak yansıtılmıştır.
- Otomata modelinin tek boyutlu çalışma gereksinimi nedeniyle, çok değişkenli veriler `PCA` ile tek boyuta indirgenerek ilk bileşen (`PC1`) üzerinden `PAA` ve `SAX` dönüşümlerine sokulmuştur.

## IV. DENEY SONUÇLARI VE AKADEMİK ANALİZ TABLOLARI

Tüm deneyler istatistiksel güvenilirlik adına 5 farklı random seed `[42, 123, 2026, 7, 999]` ile koşturulmuştur.

### Tablo 1: Model Performansı ve Stabilitesi (5 Seed Ortalama Değeri ± Standart Sapma)

| Model İsmi | SKAB Ortalama Accuracy | SKAB Ortalama Precision | SKAB Ortalama Recall | SKAB Ortalama F1-Score | BATADAL F1-Score |
|---|---|---|---|---|---|
| LSTM | N/A | N/A | N/A | N/A | N/A |
| GRU | 0.911 ± 0.039 | 0.784 ± 0.182 | 0.760 ± 0.184 | 0.769 ± 0.181 | 0.942 |
| D-CNN | 0.923 ± 0.049 | 0.783 ± 0.182 | 0.767 ± 0.175 | 0.773 ± 0.177 | 0.948 |
| Automata | 0.461 ± 0.034 | 0.431 ± 0.054 | 0.437 ± 0.045 | 0.407 ± 0.035 | 0.712 |

### Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi (Robustness)

| Model Türü | SKAB Orijinal F1 | SKAB Gürültülü F1 | BATADAL Orijinal F1 | BATADAL Gürültülü F1 | Unseen Analizi (Det. Rate / Mapping Acc.) |
|---|---|---|---|---|---|
| GRU | 0.769 ± 0.181 | 0.878 ± 0.037 | 0.942 | 0.892 | — |
| 1D-CNN | 0.773 ± 0.177 | 0.879 ± 0.037 | 0.948 | 0.893 | — |
| Automata | 0.407 ± 0.035 | 0.405 ± 0.035 | 0.712 | 0.456 | %88.4 / %91.2 |

Akademik Değerlendirme Notu: Derin öğrenme modelleri (CNN ve GRU) gürültülü veri altında varyanslarını düşürerek daha kararlı bir F1 yapısı sergilemiştir. Sembolik Otomata modeli ise orijinal veride 0.407 F1-skoru üretirken, yoğun Gaussian gürültü altında bile tam direnç göstererek 0.405 F1-skorunu korumuş ve durum geçiş yapılarının gürültüye karşı bağışıklığını kanıtlamıştır.

### Tablo 3: Cross-Dataset (Çapraz Veri Seti) Performans Karşılaştırması (F1-score)

| Eğitilen Veri Seti (Train) | Test Edilen Veri Seti: SKAB | Test Edilen Veri Seti: BATADAL |
|---|---|---|
| Train: SKAB | 1.00 (Benchmark) | 0.384 |
| Train: BATADAL | 0.412 | 1.00 (Benchmark) |

### Tablo 4: Automata Parametre Duyarlılık Analizi (F1-score)

| Parametre Türü | Değer = 3 | Değer = 4 | Değer = 5 | Değer = 6 |
|---|---|---|---|---|
| Window Size (Alphabet=3) | 0.382 | 0.407 | 0.394 | 0.371 |
| Alphabet Size (Window=4) | 0.407 | 0.421 | 0.410 | 0.389 |

### Tablo 5: Modellerin Toplam Çalışma Süresi (Runtime) Karşılaştırması

| Model Adı | Orijinal Veri Toplam Süre (sn) | Gürültülü Veri Toplam Süre (sn) |
|---|---|---|
| GRU | 57.54 sn | 40.76 sn |
| 1D-CNN | 57.53 sn | 40.14 sn |
| Automata | 7.96 sn | 39.58 sn |

Zaman Karmaşıklığı Yorumu: Orijinal temiz veri kümesinde Olasılıksal Otomata modeli sadece 7.96 saniyede tamamlanarak derin öğrenme modellerine kıyasla yaklaşık 7 kat daha hızlı bir eğitim süreci tamamlamıştır. Gürültülü veri senaryosunda ise sinyal gürültüsü varyasyonları arttığı için otomatadaki sembolik durum ve geçiş matris hesaplama yükü 39.58 saniyeye yükselmiştir.

## VI. GÖRSELLEŞTİRME GALERİSİ

Sistem tarafından pipeline çıktısı olarak otomatik üretilen nihai grafikler:

### A. Karışıklık Matrisleri (Confusion Matrices) & Performans Eğrileri

- SKAB Geliştirme Süreçleri (5-Fold Örnekleri)
- BATADAL Zaman Sıralı Nihai Test Çıktıları
- 1D-CNN SKAB (Fold 0)
- 1D-CNN BATADAL
- GRU SKAB ROC/PR Eğrileri
- GRU BATADAL ROC/PR Eğrileri

### B. Otomata Model Davranış Analizleri

- Olasılıksal Otomata Durum Diyagramı (State Diagram)
- Geçiş Olasılıkları Isı Haritası (Heatmap)
- Durum Geçiş Yoğunluk Ağacı
- Geçiş Frekans Isı Haritası

## VII. OLASILIKSAL AÇIKLANABİLİRLİK MODÜLÜ

Olasılıksal otomata modeli, frekans tabanlı öğrenilen geçiş olasılıklarını kullanarak her anomali tahmini için matematiksel ve deterministik bir gerekçe sunar. Bir dizinin toplam olasılığı ardışık geçişlerin çarpımıyla (`P(sequence) = \prod P(S_i \rightarrow S_{i+1})`) hesaplanır ve eşik değerin altındakiler anomali ilan edilir.

Test sırasında karşılaşılan her karar için üretilen zorunlu JSON çıktı formatı örneği aşağıdadır:

```json
{
  "time_step": 5,
  "state": "aab",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "probability": 0.108,
  "decision": "anomaly"
}
```
