# Public underwater fish datasets, invasive pufferfish monitoring, edge fish-detection systems and marine CV evaluation (as of Oct 2026)

Scope note: research done with web search plus fetching of primary pages (Zenodo, LILA, GitHub, Frontiers, Wikipedia). About 17 tool calls. Several dataset licences (Fish4Knowledge, RUOD, DUO, URPC, SUIM, LifeCLEF, MOTFish, WildFish) could not be verified in this pass and are listed under Gaps rather than stated as fact.

## 1. Catalogue of usable public datasets (training / pretraining / hard negatives)

### Takeaway
No public dataset I found contains annotated underwater imagery of *Lagocephalus sceleratus* or *Torquigener flavimaculosus*. The most useful assets for the project are: (a) the LILA **Community Fish Detection Dataset** (1.9M frames, 935k boxes, single "fish" class, built from 17 sources) for generic fish pretraining and non-target-fish negatives; (b) the **Mallorca / IMEDEA Labelled Mediterranean Fish Dataset** (CC BY 4.0, YOLO boxes, 20 Mediterranean species) as the closest in-domain Mediterranean source of non-target-fish hard negatives; and (c) permissively licensed sets (Brackish CC BY-SA 4.0, NOAA Puget Sound CDLA-permissive, Roboflow Fish CC0, FishTrack23 CC BY 4.0). FathomNet (ND / NC-ND terms) and the NC-licensed sets (TORSI, Salmon CV, GBIF parts of Marine Detect) restrict redistribution of derivatives and need care if the project publishes a merged dataset.

### Cited Findings

**Aggregated / large sets**
- LILA **Community Fish Detection Dataset**: >1.9M images/frames, >935,000 bounding boxes, compiled from 17 source datasets (freshwater, marine, lab); labels harmonised to a single class "fish"; COCO JSON with `dataset`, `original_data_source`, `is_train` fields; download from Azure/GCP/AWS or per-image HTTP — [LILA BC](https://lila.science/datasets/community-fish-detection-dataset/)
- LILA's reference split assigns locations to either train or validation (not both) "where location data existed"; location was not always available so shared backgrounds across splits cannot be ruled out; no separate test split — [LILA BC](https://lila.science/datasets/community-fish-detection-dataset/)
- Per-source licences as listed by LILA: Brackish CC BY-SA 4.0; Coralscapes Apache 2.0; Deep Vision Fish CC-BY 4.0; DeepFish MIT (subset); F4K Detection and Tracking "not explicitly stated"; FathomNet CC-BY-ND 4.0; FishCLEF-2015 not explicitly stated; Tropical Freshwater Fish N. Australia CC-BY 4.0; Marine Detect mixed (Roboflow parts CC-BY 4.0, GBIF parts CC-BY-NC 4.0); MIT Sea Grant River Herring CDLA-permissive 1.0; Puget Sound Nearshore Fish CDLA-permissive 1.0; Project Natick not stated (software MIT); Roboflow Fish CC0; Salmon Computer Vision CC-BY-NC-SA 4.0; TORSI CC-BY-NC-SA 4.0; FishTrack23 CC-BY 4.0; AAU Zebrafish Re-ID CC BY 4.0 — [LILA BC](https://lila.science/datasets/community-fish-detection-dataset/)

**Individual datasets**
- **DeepFish** (Australia, 20 tropical habitats; camera over side of boat): the fish-datasets compilation used 620 images (310 with masks, 310 *empty*), 374 boxes derived from masks; 7.1 GB from James Cook / Queensland host; licence "Code is MIT, data is implied-MIT" — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets); described as images "across 20 tropical Australian habitats" — [Ultralytics DeepFish](https://platform.ultralytics.com/ultralytics/datasets/deep-fish); original paper (Saleh et al. 2020) — [PMC7473859](https://pmc.ncbi.nlm.nih.gov/articles/PMC7473859/)
  - Licence conflict: Dataset Ninja says MIT — [Dataset Ninja](https://datasetninja.com/deep-fish); one search summary reported that the DeepFish paper lists CC BY 4.0; LILA says licence "not explicitly stated" — [LILA BC](https://lila.science/datasets/community-fish-detection-dataset/). Treat as unresolved.
  - DeepFish is notable because roughly half of its classification/segmentation frames are labelled fish-absent, a ready source of empty-water negatives — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets)
- **Brackish Dataset** (Denmark, brackish water, varying visibility): 12,444 RGB images, 35,565 bounding boxes, CC BY-SA 4.0, via Kaggle — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets); a survey gives 14,674 images / 35,565 instances (count discrepancy) — [arXiv 2410.05577](https://arxiv.org/pdf/2410.05577); a separate Roboflow "Brackish Underwater" copy is listed CC BY 4.0 — do not confuse — [search summary of LILA/Roboflow listings](https://lila.science/datasets/community-fish-detection-dataset/)
- **OzFish** (AIMS, Australian BRUV footage): AIMS metadata record lists licence "Attribution 3 Australia" (CC BY 3.0 AU) — [AIMS metadata](https://apps.aims.gov.au/metadata/view/38c829d4-6b6d-44a1-9476-f9b0955ce0b8); GitHub README says CC BY 3.0 AU but the badge shows CC BY 4.0 (internal inconsistency) — [open-AIMS/ozfish](https://github.com/open-AIMS/ozfish); excluded from the fish-datasets compilation for annotation quality — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets); described elsewhere as clear-water coastal scenes with limited visual diversity — [EPANet arXiv 2508.00528](https://arxiv.org/pdf/2508.00528)
- **FathomNet**: 100k+ images, 300k+ annotations (exportable to VOC/COCO/YOLO), mainly ROV, global ocean; licence quoted as "CC BY-NC-ND 4.0" with a provision allowing ML training for all purposes — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets); annotations under CC BY-ND 4.0 which "prohibited conversion to other schemes and formats" per BenthicNet authors — [BenthicNet arXiv 2405.05241](https://arxiv.org/pdf/2405.05241); content historically dominated by deep-sea Monterey Bay taxa (jellyfish, larvaceans, benthic organisms) — [arXiv 2302.07344](https://arxiv.org/pdf/2302.07344)
- **FishNet (Khan et al., ICCV 2023, KAUST)**: 94,532 images, 17,357 species, 3,826 genera, 463 families, 83 orders, 8 classes; includes bounding boxes for detection and functional-trait labels; project page fishnet-2023.github.io; DOI 10.1109/ICCV51070.2023.01874 — [ICCV 2023 paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_FishNet_A_Large-scale_Dataset_and_Benchmark_for_Fish_Recognition_Detection_ICCV_2023_paper.pdf); [ML Anthology](https://mlanthology.org/iccv/2023/khan2023iccv-fishnet)
  - FishNet++ (2025) reports heavy long-tail: only 495 species with ≥10 images, and 266 species names no longer valid taxonomically — [FishNet++ arXiv 2509.25564](https://arxiv.org/pdf/2509.25564)
  - Do not confuse with **Fishnet Open Images** (Fishnet.AI; ~35k above-water fishing-vessel images, ~163k boxes) — [arXiv 2106.09178](https://arxiv.org/pdf/2106.09178); [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets)
- **NOAA Puget Sound Nearshore Fish 2017-2018**: 77,739 images, 67,990 boxes (fish, crustaceans), CDLA-permissive 1.0, LILA — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets)
- **MIT Sea Grant River Herring**: 262,050 images, 91,482 boxes, CDLA-permissive 1.0 — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets)
- **TORSI**: 1,051 images, 14,414 boxes, CC BY-NC-SA 4.0 (CSIRO) — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets)
- **Roboflow Fish**: 680 images (670 duplicated), 1,582 boxes, CC0 — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets)
- **Fish4Knowledge (F4K)**: a survey reports download links for the dataset and subsets are no longer available — [arXiv 2410.05577](https://arxiv.org/pdf/2410.05577); F4K Detection & Tracking on GitHub has point/XML annotations, licence not stated — [filippovarini/fish-datasets](https://github.com/filippovarini/fish-datasets)
- **RUOD**: 10 classes of underwater objects incl. marine organisms and man-made artefacts — [arXiv 2410.05577](https://arxiv.org/pdf/2410.05577) (licence not found)
- **DUO** (refined URPC/UTDAC2020): 7,782 images (6,671 train / 1,111 test), 74,515 instances, 4 categories (echinus, holothurian, starfish, scallop — benthic, not fish) — [arXiv 2410.05577](https://arxiv.org/pdf/2410.05577) (licence not found)

**Mediterranean-specific**
- **Labelled Mediterranean Fish Dataset** (IMEDEA, UIB-CSIC; Catalán & Alvarez-Ellacuria; Zenodo 10.5281/zenodo.7534425, v1 Jan 2023): 1,548 underwater images, >17,965 labelled fish of 20 species, YOLOv5 boxes, 638.5 MB, **CC BY 4.0** — [Zenodo](https://zenodo.org/records/7534425)
- Associated paper (Frontiers in Marine Science 2023, "Automatic detection and classification of coastal Mediterranean fish… good practices"): Mallorca (Andratx Bay, Palma Bay), fixed camera on wreck at 6 m, drop-down/diver video 5–20 m, second fixed camera at 8 m, external test from OBSEA observatory (Catalonia); classes include *Chromis chromis, Coris julis, Diplodus* spp., *Oblada melanura, Seriola dumerilii, Sarpa salpa, Sparus aurata, Serranus scriba*; **no pufferfish or Lessepsian species mentioned** — [Frontiers 2023](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1151758/full)
- **SardinIA** (ISPRS 2025): 729 images, 28 Mediterranean fish species from Sardinian underwater video — [ISPRS Archives 2025](https://isprs-archives.copernicus.org/articles/XLVIII-2-W10-2025/79/2025/isprs-archives-XLVIII-2-W10-2025-79-2025.pdf)
- **MEDFISH101** (~70,000 validated images, 101 Mediterranean species) was found only as a search snippet; availability and licence unconfirmed — [search result, ResearchGate](https://www.researchgate.net/publication/369370553_Automatic_detection_and_classification_of_coastal_Mediterranean_fish_from_underwater_images_good_practices_for_robust_training)
- **SFISHTRACK** (Scientific Data 2026): underwater fish videos with pixel-level segmentation and MOT annotations; not Mediterranean-specific — [Nature Sci Data](https://www.nature.com/articles/s41597-026-07786-z); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13469126/)
- Eastern Mediterranean NIS: a PMC paper documents numerous new records of tropical non-indigenous species and highlights *recognition and identification* difficulty — [PMC7817654](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7817654/); eDNA vs visual survey comparison for NIS fish in Greek waters exists but provides no image set — [CORE](https://core.ac.uk/outputs/648014826)

### Inferences
- For a publishable derivative dataset, the safest pretraining / negative sources are CC0, CC BY, CDLA-permissive, CC BY-SA (share-alike obligation on derivatives) and MIT-licensed sets. FathomNet (ND) and the NC-SA sets (TORSI, Salmon CV) can be used for internal training but derived box files or merged datasets should not be redistributed.
- Mediterranean Mallorca dataset species (*Diplodus, Sarpa salpa, Oblada, Chromis, Coris, Serranus*) overlap strongly with common Turkish Aegean/Levantine shallow-reef fish, so it is the best in-domain non-target negative source, though it is Western Mediterranean and contains no Lessepsian species.
- Pufferfish positives will have to come from project-collected footage and citizen-science photos (iNaturalist/GBIF), with per-image licence filtering; no curated underwater Tetraodontidae box dataset was found.
- Single-class "fish" pretraining (LILA) followed by fine-tuning a 2-class head (target puffer vs other fish) is the natural use of these resources.

### Gaps
- Licences for RUOD, DUO, URPC, SUIM, LifeCLEF/SeaCLEF/FishCLEF, MOTFish, WildFish, FishTrack23 (LILA says CC-BY 4.0, original not checked) and Fish4Knowledge original release were **not verified** from primary pages in this pass. Background knowledge (unverified): SUIM is a semantic-segmentation set (~1.5k images, 8 classes incl. "fish and vertebrates"); WildFish is a large in-air/underwater classification set (~1,000 species, ~54k images); URPC/DUO are Chinese benthic-harvesting sets without fish classes. Check each repository before use.
- FishNet (Khan 2023) licence not found; must be read on fishnet-2023.github.io. Whether FishNet contains Tetraodontidae / *Lagocephalus* images is likely (17k species from FishBase-type sources) but not verified, and those images are mostly not underwater in-situ.
- OzFish exact image/box counts were not retrieved.
- iNaturalist / GBIF media licences are per-observation (users choose CC0 / CC BY / CC BY-NC / all rights reserved) — background knowledge, not verified in this pass; the number of *L. sceleratus* underwater photos available was not counted.
- Whether DeepFish's empty frames are representative of Mediterranean empty water (turbidity, Posidonia, sand) is unknown.

## 2. *L. sceleratus* / *T. flavimaculosus* biology, detection and monitoring in the Mediterranean (focus Turkey)

### Takeaway
No published computer-vision (image/video) detection study for *L. sceleratus* or *T. flavimaculosus* was found; the only "ML detection" paper uses DNA, not images. Monitoring in Turkey is based on fisher interviews, landings, the state per-specimen purchase/bounty scheme, published records and citizen/fisher reports. Biology relevant to a camera system: large (up to ~40 cm+), silver lateral band, rocky-bottom species seen from ~5–10 m in spawning season down to 150–220 m, with *L. lagocephalus* and other *Lagocephalus* spp. as main visual confusion species; *T. flavimaculosus* is a small (5–11 cm) sandy/seagrass burrowing puffer — a very different visual target.

### Cited Findings

**L. sceleratus — appearance, size, confusion species**
- Up to 40 cm; grey/brown back with darker spots, white belly, characteristic silver band along the flanks; more elongate than the oceanic puffer, symmetrical tail fin; contains tetrodotoxin (ovaries > skin, muscle, liver) — [Wikipedia](https://en.wikipedia.org/wiki/Lagocephalus_sceleratus)
- Very similar to *Lagocephalus lagocephalus* (oceanic puffer) — [Wikipedia](https://en.wikipedia.org/wiki/Lagocephalus_sceleratus); *L. suezensis* is often confused with *L. sceleratus* — [Wikipedia L. suezensis](https://en.wikipedia.org/wiki/Lagocephalus_suezensis); other Mediterranean-present congeners include *L. spadiceus* — [Wikipedia L. spadiceus](https://en.wikipedia.org/wiki/Lagocephalus_spadiceus)
- Juvenile *L. sceleratus* in the North Aegean had high TTX levels and "can be mistaken for edible species" — [Food Chemistry, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0308814619306430)

**Depth, habitat, behaviour**
- Native range: rocky bottoms from shallow coastal water to 250 m; diet benthic invertebrates — [Wikipedia](https://en.wikipedia.org/wiki/Lagocephalus_sceleratus)
- FishBase (Turkey page) depth range 18–100 m — [FishBase](https://www.fishbase.se/country/CountrySpeciesSummary.php?c_code=792&genusname=Lagocephalus&speciesname=sceleratus)
- SW Turkey fishers' knowledge: initially seen mostly in rocky areas ~10 m; later found as deep as 220 m (Fethiye Bay, April 2021); aggregates in bay shallows at 5–10 m in June (spawning), a few caught at the surface — [Biology & ecology review, Eastern Mediterranean (ScienceDirect, 2021)](https://www.sciencedirect.com/org/science/article/pii/S1619003321000323) (page returned 403 on direct fetch; facts from search snippet)
- Deepest earlier record cited as 150 m, Gulf of Antalya — [Sea of Marmara first record](https://www.researchgate.net/publication/280035882_First_record_of_the_Lessepsian_migrant_Lagocephalus_sceleratus_Gmelin_1789_Tetraodontidae_in_the_Sea_of_Marmara) (conflicts with 220 m above; the newer figure supersedes)
- Matures at ~3.5 years; described as "invasive species of highest concern" in southern Turkey — [TrJFAS](https://trjfas.org/pdf.php?id=1042) (via search summary)

**Distribution / invasion history**
- First Mediterranean record Gökova Bay (Turkey) 2003 — [ScienceDirect review](https://www.sciencedirect.com/org/science/article/pii/S1619003321000323); Mersin and İskenderun records 2010 at 72 m and 53 m — [ResearchGate figure](https://www.researchgate.net/figure/Lagocephalus-sceleratus-caught-in-Northeastern-Mediterranean-Sea-Turkey_fig1_259757816)
- Spread west: Israel, S Turkey, Cyprus, S Greece, Crete, Rhodes, Lampedusa (2013), Malta & Montenegro (2015), Albania, France (2014), Croatia (2022), Strait of Gibraltar — [Wikipedia](https://en.wikipedia.org/wiki/Lagocephalus_sceleratus); first Sea of Marmara record — [ResearchGate](https://www.researchgate.net/publication/280035882_First_record_of_the_Lessepsian_migrant_Lagocephalus_sceleratus_Gmelin_1789_Tetraodontidae_in_the_Sea_of_Marmara); northernmost Mediterranean record (2025) compiled records from professional and recreational fishers, divers and citizens, and calls for monitoring, regulation, public education and citizen science — [Acta Ichthyol. Piscat. 2025](https://aiep.pensoft.net/article/146945/)
- Italy: "early detection and participatory monitoring in practice" — new records in Tyrrhenian and Ionian seas via participatory monitoring — [ResearchGate](https://www.researchgate.net/publication/310602456_New_records_of_the_silver-cheeked_toadfish_Lagocephalus_sceleratus_Gmelin_1789_in_the_Tyrrhenian_and_Ionian_Seas_early_detection_and_participatory_monitoring_in_practice)

**Turkish management / monitoring**
- 2015 survey of 261 fishers (İzmir–Hatay): two-thirds favoured eliminating the species; authors proposed a bounty / per-fish premium — [Acta Ichthyol. Piscat. 2015 (DOAJ)](https://doaj.org/article/3a28dfee0d45458e9ee18ad75ec025bd)
- Antalya: Provincial Directorate of Agriculture buys tails or whole fish at 14 locations; TL 12.5 per *L. sceleratus*, TL 5 for 7 other small puffer species; quota of 1.2 million puffers "this year" (article date not stated) — [Daily Sabah](https://www.dailysabah.com/turkey/turkiyes-blowfish-hunters-make-fortune-killing-invasive-species/news/amp)
- İskenderun: ~TL 200,000 paid over three years at TL 12.50/specimen (article c. Jan 2023) — [Daily Sabah](https://www.dailysabah.com/turkiye/turkiye-battles-invasive-pufferfish-to-protect-biodiversity/news/amp)
- Economic-loss paper: "causes substantial economic losses in the Turkish Mediterranean coast: a call for decision makers" (Akdeniz University) — [AVESIS](https://avesis.akdeniz.edu.tr/yayin/4f1d7fbe-1ea2-4c28-9abe-4d6eb0bed811/silver-cheeked-toadfish-lagocephalus-sceleratus-actinopterygii-tetraodontiformes-tetraodontidae-causes-a-substantial-economic-losses-in-the-turkish-mediterranean-coast-a-call-for-decision-makers)

**AI / CV detection of L. sceleratus**
- Only ML identification work found: Demertzis & Iliadis (Neural Computing and Applications, 2017) — semi-supervised bio-inspired neural approach (ELM + Izhikevich spiking neurons) on environmental DNA data, not images — [Springer](https://link.springer.com/article/10.1007/s00521-016-2591-2)

**T. flavimaculosus**
- Red Sea/Indian Ocean species; first Mediterranean record Haifa Bay 1987; Fethiye photo record 2002 — [CIESM atlas](https://ciesm.org/atlas_preview/Torquigenerflavimaculosus.php)
- Habitat: shallow sandy shores often with seagrass; common size 5–9 cm, max 11 cm — [CIESM atlas PDF](https://ciesm.org/atlas/fishes_2nd_edition/Torquigener_flavimaculosus.pdf)
- Burrowing as anti-predator behaviour documented in Fethiye Bay (Bilecenoğlu 2005) — [Ege Univ. repository](https://acikerisim.ege.edu.tr/items/d7b6d6cc-889c-4c8d-89cb-85593b9e33bc/full)
- Established from Levant to Fethiye; range extended to SE Aegean (Rhodes) — [Corsini-Foka et al. 2006, Med. Mar. Sci.](https://ejournals.epublishing.ekt.gr/index.php/hcmr-med-mar-sc/article/download/12144/12153)
- Ulman et al. 2023 (Aquatic Invasions): 1,360 specimens from Finike, Fethiye and Cyprus; "very common" invasive in E. Mediterranean; now treated as conspecific with *T. hypselogeneion* — [Aquatic Invasions](https://aquaticinvasions.arphahub.com/article_preview.php?id=103438)

**Invasive-fish CV analogue: lionfish**
- Naddaf-Sh, Myler & Zargarzadeh (Complexity, 2018): small open-source ROV with deep learning for real-time red lionfish detection, tested in Gulf of Mexico, to support spear divers — [DOAJ](https://doaj.org/article/4bd401a75ae843519dc0d65eca4ed6c0); [IDEAS](https://ideas.repec.org/a/hin/complx/5298294.html)
- Trade press: a 2020 single-species YOLOv4 lionfish detector trained on mixed-source data ran ~5 FPS on low-power hardware — [Marine Technology Magazine Nov/Dec 2025](https://magazines.marinelink.com/Magazines/MarineTechnology/202511/page/43) (secondary source)
- WPI 2022 student project on a lionfish-harvesting ROV with CV detection — [WPI](https://digital.wpi.edu/downloads/7w62fc40d?locale=es)

### Inferences
- A *L. sceleratus* underwater detector would be, as far as these searches show, a first-of-kind CV contribution — a point worth stating (carefully: "no study found") in the manuscript.
- Detection is primarily a fine-grained problem against *L. lagocephalus*, *L. spadiceus*, *L. suezensis* and *L. guentheri* rather than against generic fish; the silver lateral band and elongate body are the main visual cues. A "Lagocephalus spp." class may be more defensible than species-level claims from video.
- The 5–10 m June spawning aggregations in bays are the realistic window for a shallow, low-cost (Pi-based) camera; deeper (50–220 m) occurrences are out of reach of diver/shallow deployments.
- Bounty/landing data in Antalya/İskenderun could serve as an independent abundance signal for validating camera-based counts.
- *T. flavimaculosus* (5–11 cm, burrowing, sandy/seagrass) needs a different camera geometry (close, downward) and is a hard small-object case.

### Gaps
- No direct access to the 2021 Eastern Mediterranean biology review (403); diel activity pattern (day vs night) of *L. sceleratus* was not found in accessible sources.
- MedMIS, EASIN, iSea (Greece) and Cypriot programmes were not covered by retrieved sources; their record counts and photo licences are unknown.
- Official Turkish Ministry (Tarım ve Orman Bakanlığı) regulation for the puffer purchase programme (year, legal basis, current per-fish rate) not found; only news reports.
- No ROV/BRUV/camera survey specifically targeting *L. sceleratus* was found.

## 3. Comparable embedded / low-cost edge fish detection systems

### Takeaway
Published edge numbers cluster at ~6–8 FPS for nano YOLO on Raspberry Pi 5 CPU, ~20–60 FPS on Jetson Orin-class boards, ~5 FPS for an early lionfish YOLOv4 system on low-power hardware, and <1 FPS for two-stage detectors on old Jetson TX2. Most papers report model-only inference time, not end-to-end camera-to-alert latency, and few report power or field failure modes.

### Cited Findings
- Raspberry Pi 5 (CPU only), fish mortality detection in recirculating aquaculture: YOLO26n fastest at 7.51 FPS; all nano models >6 FPS — [arXiv 2607.09835](https://arxiv.org/pdf/2607.09835) (from search summary)
- ULFD-YOLO ultra-lightweight fish detector on Jetson Orin Nano: ~19–24 FPS at 448×640 without TensorRT FP16/INT8; authors note timings are model inference, not full acquisition-to-display latency — [MDPI Animals 16(17):2640](https://www.mdpi.com/2076-2615/16/17/2640); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13564336)
- Nephrops catch estimation in demersal trawls on edge hardware: YOLOv8s with adaptive frame skipping reached 97.47 FPS (effective, due to skipped frames) — [PMC11045813](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11045813/)
- Cognitive Edge Device for real-time aquatic environmental monitoring — [arXiv 2401.06157](https://arxiv.org/pdf/2401.06157)
- SCMYOLO lightweight underwater fish detector — [ICES J. Mar. Sci. 82(4) fsaf038](https://academic.oup.com/icesjms/article/82/4/fsaf038/8112733); BGLE-YOLO lightweight underwater bio-detection — [PMC11902696](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11902696/)
- Underwater fish recognition system on intelligent edge computing (Springer chapter) — [Springer](https://link.springer.com/chapter/10.1007/978-981-95-9705-5_8)
- Review: Faster R-CNN on Jetson TX2 < 1 FPS (not real-time); one Pi 4B underwater system used YOLOv5 without reporting FPS — [MDPI JMSE review 11(4):867](https://www.mdpi.com/2077-1312/11/4/867) (via search summary)
- ROV analogues: BlueROV with YOLOv4-Tiny starfish detection, 80.40% accuracy — [REDI CEDIA](https://redi.cedia.edu.ec/document/549488); HyDRA Scorpion low-cost modular ROV, YOLOv8 0.89 mAP on jellyfish — [arXiv 2605.09093](https://arxiv.org/pdf/2605.09093); transfer-learned YOLO for multiple fish from ROV, +4% mAP — [Univ. Hertfordshire](https://researchprofiles.herts.ac.uk/en/publications/object-detection-using-transfer-learning-for-underwater-robot/)
- Lionfish: YOLOv4 single-species ~5 FPS on low-power hardware — [Marine Technology Magazine](https://magazines.marinelink.com/Magazines/MarineTechnology/202511/page/43)

### Inferences
- A Pi 5 CPU system at 640 px should expect roughly 5–8 FPS with nano YOLO; for a static monitoring camera where fish dwell for seconds, 2–5 FPS sampling is likely sufficient, and frame-skipping / motion gating (as in the Nephrops trawl work) is a legitimate way to report higher "effective" throughput — but must be disclosed.
- Comparisons across papers are weak because input size, precision (FP32/FP16/INT8), and whether pre/post-processing is timed vary; the project should report end-to-end latency, model-only latency, input size, precision and power on the same table.

### Gaps
- No peer-reviewed field deployment with *power consumption*, *thermal throttling*, *biofouling* or *housing fogging* failure-mode statistics was retrieved.
- No Coral Edge TPU underwater fish benchmark was retrieved in this pass.
- BRUV/buoy deployments with on-board (not post-hoc) inference were not found.

## 4. Evaluation protocols in marine computer vision

### Takeaway
Detection papers mostly report mAP@0.5 / mAP@0.5:0.95, precision and recall, but ecological users care about counts (MaxN, abundance), and multiple studies show that detector-derived MaxN is accurate at low abundance but biased (over- or under-) at high density. Best practice emerging in 2021–2026: held-out external sites/cameras as test sets, abundance-dependent error metrics, and semi-automatic human-in-the-loop verification, which outperformed fully automatic counting.

### Cited Findings
- Mallorca study used external test sets from a different camera (C) and a different region/observatory (OBSEA, D): mAP@0.5 0.84 (val) dropping to 0.78 on the external OBSEA set — demonstrating domain shift — [Frontiers 2023](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1151758/full)
- Same study: diverse backgrounds (sand, seagrass, rock) improved generalisation; a few hundred well-chosen labels per class approached full-data performance (scenario E5 ≈ E0 with ~1/10 of objects); excluding tiny (<100 px²), truncated and heavily overlapping fish from training improved precision/recall; genus-level classes (*Diplodus* sp.) caused confusion — [Frontiers 2023](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1151758/full)
- LILA split assigns whole locations to train or val to avoid leakage, but warns shared backgrounds cannot be ruled out where location metadata was missing — [LILA BC](https://lila.science/datasets/community-fish-detection-dataset/)
- Snapper BRUV (Australia): Faster R-CNN accurate at low abundance but over-predicted MaxN by up to 50% at high abundance; Seq-NMS + tuned confidence threshold + cubic correction brought MaxN within 1–2 fish of ground truth — [bioRxiv 2021](https://www.biorxiv.org/content/10.1101/2021.02.01.429285v1.full); [Frontiers 2021](https://www.frontiersin.org/articles/10.3389/fmars.2021.658135/full)
- Mediterranean deep-water snappers (Frontiers Mar. Sci. 2025): correlation with manual MaxN 0.72–0.9, underestimation at high abundance; a semi-automatic protocol (model assists observer) reached 0.96 correlation — [Frontiers 2025](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1476616/full)
- Stereo-BRUV gadoid/flatfish (preprint 2025): argues RMSE/MAE with constant tolerance are inappropriate since error grows with abundance; proposes "Mean Peak Success" with proportional tolerance — [Research Square](https://www.researchsquare.com/article/rs-11070769/v1) (not peer-reviewed)
- Weakly supervised MaxN from per-video labels only: MAE 4.23 over counts up to ~300 — [ICCV 2025 Workshop](https://openaccess.thecvf.com/content/ICCV2025W/CVAUI%20&%20AAMVEM/papers/Sorrenti_Weakly_Supervised_MaxN_Estimation_in_Baited_Remote_Underwater_Video_ICCVW_2025_paper.pdf)
- Diver-operated video (Methods Ecol. Evol. 2026): MaxN underestimated true abundance by up to 40% even with perfect labels; clustering-based counting did better — [Bürgi et al. 2026, MEE](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210x.70283?af=R)
- Another 2022 Frontiers BRUV automation paper — [Frontiers 2022](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.944582/full)

### Inferences
- For a rare-target alarm system (pufferfish), mAP alone is insufficient; recommended reporting: recall at a fixed operating precision, false alarms per hour of empty / non-target footage, per-event (track-level) detection rate rather than per-frame, and counting error stratified by abundance.
- Test sets should be split by site and by deployment date/session (never random frames from the same video), with at least one fully external site, mirroring the Mallorca OBSEA design and LILA's location split.
- A human verification step (model proposes, expert confirms) is supported by the 0.96 vs 0.72–0.9 result and is the realistic deployment workflow for invasive-species alerts.

### Gaps
- No published standard for "false alarms per hour" in marine CV was found in this pass; it is borrowed from camera-trap / surveillance practice (not sourced here).
- No paper retrieved quantified leakage (random-frame vs site split mAP gap) for underwater fish specifically.
