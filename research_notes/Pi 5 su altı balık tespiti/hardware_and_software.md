# Raspberry Pi 5 (16 GB) underwater fish-detection camera: hardware and system design (as of Oct 2026)

Scope: camera, optics, exposure, housing, thermal/power, lighting, on-device software architecture, and prior open-source Pi underwater camera projects, for a moving platform (small ROV or boat-mounted). Budget-conscious TÜBİTAK 2209-B project, no AI accelerator HAT.

Calculations marked **[calc]** are my own arithmetic from cited specifications, not claims taken from sources.

---

## 1. Camera choice for Pi 5 (motion blur, rolling shutter, low light, underwater FOV)

### Takeaway
The official Raspberry Pi **Global Shutter Camera (IMX296, 1456x1088 @ 60 fps, 3.45 µm pixels, C/CS mount)** is the strongest choice for a *moving* platform, because it removes rolling-shutter skew and its large pixels help in low light. Its drawback is low resolution (1.58 MP), which is still well above a typical 640 px detector input. **Camera Module 3 (IMX708)** is the cheaper, simpler option with autofocus and 120 fps at 1536x864, but it has a rolling shutter. The Wide variant is attractive behind a flat port because it partly offsets the ~1.33x FOV loss. The HQ camera (IMX477) offers no real advantage here.

### Cited findings
- Official spec table ([Raspberry Pi camera docs](https://www.raspberrypi.com/documentation/accessories/camera.html)):
  - **Camera Module 3 (IMX708)**: 4608x2592 (11.9 MP), 1.4 µm pixels, 7.4 mm diagonal. Standard: HFOV 66°, VFOV 41°, f = 4.74 mm, F1.8, focus ~10 cm to infinity. **Wide**: HFOV 102°, VFOV 67°, f = 2.75 mm, F2.2, focus ~5 cm to infinity. Motorised focus. Video modes 2304x1296 @ 56 fps, 2304x1296 @ 30 fps HDR, **1536x864 @ 120 fps**. NoIR versions exist for both standard and Wide.
  - **HQ Camera (IMX477)**: 4056x3040 (12.3 MP), 1.55 µm pixels, 7.9 mm diagonal, lens-dependent FOV (C/CS or M12 mount). Modes 2028x1080 @ 50 fps, 2028x1520 @ 40 fps, 1332x990 @ 120 fps.
  - **Global Shutter Camera (IMX296)**: 1456x1088 (1.58 MP), **3.45 µm pixels**, 6.3 mm diagonal, C/CS mount, **1456x1088 @ 60 fps**, shutter type "Global". (The docs themselves describe the resolution inconsistently as 1.5, 1.6 and 1.58 MP.)
  - **AI Camera (IMX500)**: 2028x1520 @ 30 fps. It does on-sensor inference, which is a way around "no AI HAT", but it is outside this scope.
- **Pi 5 connectivity**: two 4-lane MIPI connectors (CAM/DISP0, CAM/DISP1), both usable for cameras. Pi 5 uses the **mini 22-pin** connector, so a **Standard-to-Mini cable** is needed because cameras ship with Standard-Standard cables. Connect the camera cable before fitting the Active Cooler, which blocks access — [Raspberry Pi camera docs](https://www.raspberrypi.com/documentation/accessories/camera.html)
- Picamera2 notes that the Pi 5 can drive two cameras simultaneously, for example for stereo or redundancy — [Picamera2 manual v0.3.37](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf)
- Third-party IMX296 modules: Arducam sells a colour IMX296 with an **M12 lens** (1440x1080 @ 60 fps RAW10), which is smaller and fits small tubes and dome ports better than a C/CS lens — [Arducam product](https://www.arducam.com/product/1-58mp-imx296-color-global-shutter-camera-module-with-m12-lens-for-raspberry-pi/), [Arducam docs](https://docs.arducam.com:443/Raspberry-Pi-Camera/Native-camera/1.58MP-IMX296/). Waveshare sells a mono M12 IMX296 (1456x1088) — [Waveshare](https://www.waveshare.com/product/imx296-130-bw-camera.htm). Arducam's USB 3.0 IMX296 tops out at ~44 fps — [Arducam USB](https://www.arducam.com/blog/product/arducam-1-58mp-imx296-color-global-shutter-usb-3-0-camera-module/)
- A third-party reseller states the GS camera can use exposures **as short as ~30 µs** given enough light — [MaxMax](https://maxmax.com/maincamerapage/monochrome-cameras/raspeberry-pi-cameras/raspberry-pi-global-shutter-1-6mp). This is a vendor claim, not confirmed by Raspberry Pi documentation.
- USB cameras get only the software-rendered Qt preview in Picamera2, with no hardware-assisted rendering — [Picamera2 manual](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf). USB cameras also do not go through the Pi 5 ISP (PiSP) used for the CSI cameras.
- **Flat port vs dome port**:
  - A flat port reduces the in-air FOV by roughly the refractive index of water (≈1.33). It is a rule of thumb that is "in general satisfied" but affected by other factors — [ISPRS 2017](https://isprs-archives.copernicus.org/articles/XLII-2-W3/481/2017/isprs-archives-XLII-2-W3-481-2017.pdf)
  - Worked vendor example: **90° in air becomes ~64° behind a flat port** — [Unmanned Systems Technology / dome vs flat](https://www.unmannedsystemstechnology.com/feature/comparing-the-use-of-dome-flat-ports-for-underwater-photography/)
  - The theoretical maximum angle of view behind a flat port is **~97.2°** — [scubageek: Optics of Dome Ports](https://www.scubageek.com/articles/wwwdome)
  - Flat ports cause pincushion distortion and make the system non-single-viewpoint. Domes preserve the FOV and do not refract for a pinhole camera placed **exactly at the dome centre**, but that centring is hard to achieve in practice — [Wetpixel](https://live.wetpixel.com/articles/review-nauticam-wide-angle-corrector-port/P1), [GEOMAR dome calibration preprint](https://www.geomar.de/fileadmin/personal/fb2/mg/kkoeser/domecalibration_preprint.pdf), [arXiv 2405.18018 refractive calibration tool](https://arxiv.org/pdf/2405.18018)

### Inferences
- **[calc] FOV behind a flat port** (Snell's law, tan(θw/2) = tan(θa/2)/1.33):
  - CM3 standard: 66° → **~52°** HFOV underwater; 41° → ~31° VFOV.
  - CM3 Wide: 102° → **~86°** HFOV underwater; 67° → ~53° VFOV.
  - GS camera with a 6 mm lens (sensor width 1456 x 3.45 µm ≈ 5.0 mm): ~45° in air → ~35° underwater. A 2.8–4 mm CS/M12 lens is needed for a wider view.
  - For a fish survey from a moving platform, the **CM3 Wide or a short-focal GS lens** is preferable.
- **[calc] Scene width at 1 m range behind a flat port**:
  - CM3 standard: ≈0.98 m. At 1536 px that is ~0.64 mm/px; at a 640 px detector input, ~1.5 mm/px.
  - CM3 Wide: ≈1.86 m, so ~2.9 mm/px at 640 px.
- **Ranking for a moving platform**:
  1. **GS Camera**: no rolling-shutter skew; 3.45 µm pixels collect ~6x the light per pixel of IMX708's 1.4 µm (area ratio).
  2. **CM3 Wide**: cheap, autofocus, wide FOV, 120 fps mode; rolling-shutter skew is manageable with short exposures and modest yaw rates.
  3. HQ: larger pixels than CM3 and lens choice, but still rolling shutter, and its C-mount lens makes the housing bulkier.
  - USB webcams: avoid. They have no PiSP, their exposure control is less predictable, and they cost USB bandwidth and CPU.
- **Autofocus on CM3**: continuous AF can hunt in turbid water with low-contrast scenes. Locking manual focus (`AfMode` Manual plus a fixed `LensPosition` at the expected working distance) is likely more robust. The GS and HQ cameras use fixed manual-focus lenses.
- **NoIR variants** only help when paired with IR illumination. Water strongly absorbs IR, so IR is useful only at very short range; SuMOS used a CM3 NoIR with IR LEDs for close-range substrate work (see Section 7).

### Gaps
- No official figure was found for **IMX708 rolling-shutter readout (line) time**, so skew in pixels per degree of yaw rate could not be quantified. It is worth measuring on the bench (for example, with a rotating target).
- No official Raspberry Pi low-light sensitivity or SNR comparison between IMX708, IMX477 and IMX296 was found. The pixel-area argument is an inference.
- 2026 prices were not verified. Ballpark list prices were roughly CM3 ~$25, Wide ~$35, GS ~$50 and HQ ~$50, plus lens, but these are unverified and should be checked locally (e.g. Turkish distributors).

---

## 2. Exposure: shutter time vs motion blur, gain, picamera2/libcamera controls, sensor modes

### Takeaway
Treat **exposure time as the primary knob**. Fix `ExposureTime` at **~1–2 ms** (≤ 1 ms for fast fish near the camera) with `AeEnable=False`, and let `AnalogueGain` rise. Use `FrameDurationLimits` to lock the frame rate. On a moving platform, blur from fish swimming at 0.3–0.8 m/s plus platform motion is dominated by relative speed and range.

### Cited findings
- Controls whose valid ranges depend on the sensor mode: `AnalogueGain`, `ExposureTime`, `FrameDurationLimits`, `ScalerCrop`. Query them with `picam2.camera_controls["ExposureTime"]`, which returns (min, max, default) — [Picamera2 manual](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf)
- Frame rate is set with `FrameDurationLimits` in microseconds, for example `(33333, 33333)` for 30 fps. Video configs default to fixed-rate limits, and a `FrameRate` convenience control also exists — [Picamera2 manual](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf)
- Manual exposure example: `{"ExposureTime": 5000, "AeEnable": False}`. Gain example: `AnalogueGain: 2.0` — [Arducam PiCamera2 guide](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/PiCamera2-User-Guide/)
- Autofocus controls include `AfMode` (Manual/Auto/Continuous), `AfTrigger`, `LensPosition` and `AfSpeed` ("run the autofocus algorithm faster or slower") — [Picamera2 manual](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf). Example: continuous AF `{"AfMode": 2, "AfTrigger": 0}`, manual `{"AfMode": 0, "LensPosition": x}` — [Arducam guide](https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/PiCamera2-User-Guide/)
- **Pi 5 HDR**: the Pi 5 has built-in HDR usable even with non-HDR sensors, `HdrMode = HdrModeEnum.SingleExposure`. It relies on Pi 5 **Temporal Denoise (TDN)** accumulating frames, so frames must be skipped after a mode switch (`delay=10`) — [Picamera2 manual §9.7](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf). CM3 also has a sensor HDR mode, 2304x1296 @ 30 fps — [RPi camera docs](https://www.raspberrypi.com/documentation/accessories/camera.html)
- Sensor mode frame rates: see Section 1 (CM3 120 fps @ 1536x864; GS 60 fps @ 1456x1088; HQ 120 fps @ 1332x990) — [RPi camera docs](https://www.raspberrypi.com/documentation/accessories/camera.html)
- FishCam's field settings were 1600x1200, 10 fps, ISO 400, H.264 in 5-minute files, daylight only, at 8–12 m — [Mouy et al. 2020, HardwareX](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041213/)

### Inferences
- **[calc] Motion blur = relative speed x exposure time**:

  | Relative speed | 0.5 ms | 1 ms | 2 ms | 4 ms | 8 ms |
  |---|---|---|---|---|---|
  | 0.3 m/s | 0.15 mm | 0.3 mm | 0.6 mm | 1.2 mm | 2.4 mm |
  | 0.8 m/s (fast fish) | 0.4 mm | 0.8 mm | 1.6 mm | 3.2 mm | 6.4 mm |
  | 1.3 m/s (fish 0.8 + boat 0.5) | 0.65 mm | 1.3 mm | 2.6 mm | 5.2 mm | 10.4 mm |

  With ~1.5 mm/px at 1 m range (CM3 standard behind a flat port, 640 px input), **≤ 2 ms keeps blur ≤ ~2 px for 1.3 m/s**, and **≤ 1 ms keeps it ~1 px**. At 0.5 m range the mm/px halves, so blur in pixels doubles; target ~1 ms.
- Platform **angular** motion (boat roll/pitch, ROV yaw) usually dominates translation. [calc] A 30°/s rotation with a 52° HFOV over 640 px is ~370 px/s, i.e. ~0.7 px per 2 ms. Rolling-shutter skew from that rotation is *not* reduced by short exposure; only the GS camera removes it.
- Short exposure underwater means high gain. Expect usable analogue gain of roughly 4–8x before noise significantly hurts detection; this is an inference to verify against the project's model. The GS camera's larger pixels give it headroom here.
- Pi 5 TDN/HDR accumulates frames, which can create ghosting on moving fish. **Disable HDR for detection.**
- Practical recipe (to validate in tank or field):
  - Fixed `FrameDurationLimits` for 30 fps.
  - `AeEnable=False`, `ExposureTime` 1000–2000 µs.
  - `AnalogueGain` adjusted by a slow custom loop on mean luminance, or AE with an exposure cap via a tuned "short" exposure profile.
  - Fixed white balance (`AwbEnable=False` with `ColourGains`) to stop colour cast drift in blue-green water.
  - CM3: manual focus.

### Gaps
- No source found giving the libcamera `AeExposureMode` "Short" profile limits for IMX708/IMX296 on Pi 5, or a way to cap maximum exposure while keeping auto gain. The tuning-file approach is referenced but not documented here.
- No measured noise-versus-gain curves for these sensors underwater were found.

---

## 3. Underwater housings, condensation, heat, power and runtime

### Takeaway
For a budget academic build, a **Blue Robotics 3" or 4" enclosure** suits a boat-mounted or ROV camera at < 100 m, with an **aluminium tube** for heat and an **acrylic dome or flat end cap** for the camera. The Pi 5 dissipates roughly **5–9 W at full CPU load**; in a sealed housing, the heat path to the water must be designed (thermal pad to an aluminium tube or end cap), because the Active Cooler fan only circulates air inside the sealed volume. A **5 V 5 A** supply is needed to avoid USB current limits. Tethered power (from the boat) is simplest; on-board batteries need about 15–20 Wh per hour of runtime.

### Cited findings
- **Blue Robotics WTE** ([store page](https://bluerobotics.com/store/watertight-enclosures/wte-vp/)):
  - Sizes 2"–8"; cast acrylic or 6061-T6 anodised aluminium.
  - Acrylic depth ratings: 2" (100 mm) **225 m**; 3" (150 mm) **275 m**, 3" (300 mm) 135 m; 4" (200 mm) **140 m**, 4" (300 mm) 100 m.
  - Aluminium ratings: 2" and 3" to **1000 m**.
  - Acrylic ratings apply only to **short-term submersions (< 2 weeks)**.
  - Domes: 2" to 1000 m, 3" to 750 m, 4" to 500 m (1000 m with aluminium ring).
  - Configurable enclosure "from $61"; vacuum pump $98; pressure relief valve $32.
  - Aluminium is recommended for heat-producing items (power supplies, batteries).
  - End caps have M10/M14 penetrator holes, and penetrators are not included.
- The FishCam builders used a homemade 4" PVC housing (~$100, tested to ≥30 m) and a Blue Robotics 4" housing (~$233–260, cited as 100 m).
  - A homemade housing leaked slightly and was fixed with extra epoxy. Small build imperfections can cause failure, and the commercial housing was more reliable.
  - A unit without enough ballast was dragged ~50 km by currents and its window was badly scratched.
  - Their O-ring checklist: clean and grease with silicone lubricant.
  - [Mouy et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041213/)
- PlasPi (Pi Zero W, off-the-shelf and 3D-printed parts) cost < €200 for depths < 150 m — [Purser et al. 2020, HardwareX](https://epic2-clone.awi.de/52529/1/Purseretal2020HardwareX7.pdf)
- **Pi 5 thermal behaviour** ([RPi computer docs](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)):
  - SoC limit is 85 °C. Between 80–85 °C the Arm cores are progressively throttled, and at 85 °C both Arm and GPU are throttled.
  - Default fan curve: 0% below 50 °C, 30% at 50 °C, 50% at 60 °C, 70% at 67.5 °C, 100% at 75 °C (5 °C hysteresis).
  - The official case fan draws ~60 mA @ 5 V (300 mW), from the same current budget as USB.
- Third-party thermal tests at 100% CPU load: no cooling **~85 °C**, passive heatsink ~78 °C, Active Cooler ~63 °C. Without the fan, throttling started after ~10 min of heavy load — [raspberry.tips](https://raspberry.tips/en/?p=9083), [Core Electronics](https://core-electronics.com.au/videos/answering-your-pi-5-questions-cooling-overclocking-power-and-more). These are single-source, ambient-dependent figures.
- **Power**:
  - Third-party measurements: Pi 5 CPU-only all-core stress ~**8.8 W**. Active cooler fan ~0.5 W; NVMe SSD +1–2 W. Extreme load (CPU + USB SSD + 4K video) up to ~16 W — [raspberry.tips power comparison 2026](https://raspberry.tips/en/?p=9083)
  - The same source lists a "12–25 W" figure that looks implausible and is not corroborated.
  - Raspberry Pi recommends a **5 V/5 A (27 W) USB-C PD supply**. On a 3 A supply, USB peripherals are limited (600 mA) and USB boot is disabled — [raspberry.tips](https://raspberry.tips/en/?p=9083), [RPi docs](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
  - `usb_max_current_enable=1` overrides the USB limit, and the EEPROM setting `PSU_MAX_CURRENT=5000` skips PD negotiation for a non-PD 5 V buck converter — [RPi computer docs](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- `POWER_OFF_ON_HALT=1` puts the Pi 5 PMIC in standby with all outputs off after shutdown, which is useful to save battery between deployments — [RPi computer docs](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- FishCam power: D-cell NiMH pack (28 x 10 Ah cells; the paper's wiring description is inconsistent) with a 5 V 2 A regulator. It was "memory limited, not battery limited" (80–212 h of video over 8–14 days on a Pi Zero W at 10 fps) — [Mouy et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041213/)

### Inferences
- **Heat in a sealed housing**: inside a sealed tube the Active Cooler only stirs internal air, so the real heat path is Pi SoC → heatsink/thermal pad → **aluminium end cap or tube wall → water**. Water is an excellent heat sink when submerged; the danger is **bench and deck testing in air**, where a sealed tube will heat-soak and the Pi will throttle.
- Suggested heat-path design:
  - Mount the Pi on an aluminium plate pressed via a thermal pad to an aluminium end cap, or use an aluminium tube.
  - Keep the Active Cooler or a passive heatsink on the SoC.
  - Log `vcgencmd measure_temp` and `get_throttled` alongside detections.
  - Add a software guard that reduces inference fps above ~75 °C.
- **Condensation/fogging**: seal the housing in a cool, dry environment (air-conditioned room), add **silica gel desiccant packs**, and optionally purge with dry nitrogen or argon. The Blue Robotics vacuum test also confirms the seal before immersion. A warm Pi in a tube dropped into cold water will fog the port from the inside if the internal air is humid. No primary source on this was fetched, so it is engineering practice to verify (see Gaps).
- **[calc] Power budget**:
  - Pi 5 at sustained inference ~7–9 W, camera ~0.3–1 W, GPS ~0.1–0.2 W, SSD ~1–2 W, buck converter losses ~10%.
  - Total ≈ **10–13 W** without lights; LED lights add 10–40 W each.
  - A 12 V 10 Ah LiFePO4 (128 Wh) therefore runs ~9–11 h without lights and ~3–4 h with one 20 W light.
  - A 5 V/5 A-capable buck converter from 12 V is needed.
- **Tether vs on-board**:
  - **Boat-mounted**: run power and Ethernet down a tether (e.g. Blue Robotics Fathom tether or a simple PoE scheme). This removes battery mass and heat from the housing and gives the dashboard live access.
  - **Small ROV**: tether Ethernet is standard (BlueOS architecture).
  - **Autonomous drops**: on-board battery plus local recording.

### Gaps
- No primary source fetched on a **sealed-housing Pi 5 thermal test** (temperature vs time in a closed acrylic or aluminium tube in air and in water). This is a key bench experiment for the project.
- No authoritative anti-fogging guide was fetched; the desiccant and nitrogen recommendations above are common practice, not sourced.
- Official Raspberry Pi bare-board power-consumption figures for the Pi 5 were not retrieved; the numbers are third-party.
- 16 GB-model-specific power or thermal data were not found (RAM size is not expected to matter much).

---

## 4. Lighting: need, colour attenuation, strobe vs continuous

### Takeaway
Natural light suffices only in shallow, clear water by day. FishCam explicitly needed external lights or a better sensor for deeper work. White light disturbs some species; **red light (~660 nm) is less visible to many fish but is absorbed within a few metres**. Strobing at low frequency does not appear to change behaviour more than constant light. For a detection camera, continuous white or neutral LEDs at modest power, with exposures kept short, are the pragmatic choice. Light placed off-axis from the camera reduces backscatter.

### Cited findings
- FishCam: no built-in light; deployments were daylight only at 8–12 m; "external lights or a more expensive camera sensor" would be needed deeper — [Mouy et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041213/)
- Colour loss with depth: red fades at ~5 m, orange and yellow at 10–20 m, green at ~30 m, blue at ~60 m. In coastal water, dissolved organic matter and sediment shift dominance toward green — [Sabik: underwater light absorption](https://sabik.com/?p=3774) (vendor article; the order of magnitude is consistent with standard optics)
- Puget Sound rockfish stereo-camera survey: **higher densities of small rockfish under strobed red light** than under strobed or constant white. The red-strobe vs constant-white difference was significant; larger rockfish were not significantly affected — [NOAA repository](https://repository.library.noaa.gov/view/noaa/62660)
- Behavioural guidance study: **2 Hz strobing did not alter fish behaviour vs constant light**; red light was repulsive by day with no effect at night — [eScholarship](https://escholarship.org/uc/item/2v6846gh)
- Arctic study: 660 nm red increased avoidance, 720 nm increased attraction, 525 nm green had minimal response, and flash frequency/intensity effects were not significant — [UiT Munin thesis](https://munin.uit.no/handle/10037/37425)
- SubC Imaging discusses subsea lights and lasers for marine research — [Ocean Science & Technology](https://www.oceansciencetechnology.com/feature/subc-imaging-discusses-subsea-lights-lasers-for-marine-research/)
- SuMOS (2024) used a CM3 **NoIR** with **infrared illumination** for low-light close-range recording — [Wira & Place 2024, HardwareX (DOAJ)](https://doaj.org/article/96b325e157514225981019d56d9fcc75)

### Inferences
- **Detection model domain**: the project's model is trained on photos (iNat and similar) in natural or white light. Red or IR illumination would produce monochrome-looking images and a domain shift, so **white or neutral-white light is preferable for detection accuracy**, even if it slightly biases fish behaviour.
- **Strobe vs continuous**: a strobe synchronised to a short global-shutter exposure is ideal optically (high peak light, low average power), but synchronisation adds hardware complexity. Continuous LEDs (e.g. Blue Robotics Lumen-class, ~1500 lm) are simpler, and PWM dimming can be controlled from the Pi.
- **Backscatter** in turbid water: mount lights to the side or behind the camera port, not on-axis.
- Water absorbs IR strongly, so NoIR plus IR works only at very short range (< ~0.5–1 m).

### Gaps
- No specific source was fetched for Blue Robotics Lumen light specs or power draw in 2026, so the ~1500 lm and 10–40 W figures are unverified estimates.
- No source on pufferfish (Lagocephalus) light sensitivity was found.

---

## 5. On-device software architecture (picamera2, GIL, encoding, GPS, storage, dashboard, ops)

### Takeaway
Use **Picamera2 with two streams**: a `main` stream (e.g. 1280x720 or 1536x864) for recording, and a `lores` stream (e.g. 640x480 or 640x360, **RGB allowed on Pi 5**) for inference.
- The Pi 5 has **no hardware H.264 encoder**; software encoding via FFmpeg handles 1080p30 at ~60% of one core.
- Inference (ONNX Runtime, OpenCV DNN, NCNN, TFLite) releases the GIL, so threads often suffice. Picamera2 also supports zero-copy multiprocessing.
- GPS: `gpsd` plus `chrony` with **PPS** for accurate timestamps.
- Dashboard: avoid eventlet; use FastAPI/Starlette WebSockets or Flask-SocketIO in `threading` mode, low-fps MJPEG of annotated lores frames, or **MediaMTX** for WebRTC/RTSP live video.
- Run as a systemd service with a hardware watchdog.

### Cited findings
- **Streams**: the lores image "must be no larger than the main image". On Pi 4 and earlier it must be YUV, "whereas on a Raspberry Pi 5 (or later device) it can be RGB or YUV" — [Picamera2 manual v0.3.37](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf)
- **Encoding**:
  - "On Raspberry Pi 4 and earlier devices there is dedicated hardware for H264 and MJPEG encoding. On a Raspberry Pi 5, however, these codecs are implemented in software using FFmpeg libraries."
  - The JPEG encoder is multi-threaded (default `num_threads=4`) and benefits from the GIL being released in C code.
  - `encoder.frame_skip_count` encodes every n-th frame.
  - `PyavOutput` is the recommended newer FFmpeg integration (it passes timestamps). `CircularOutput`/`CircularOutput2` allow pre-trigger recording ("start a recording with video frames from several seconds earlier").
  - Source: [Picamera2 manual](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf)
- Raspberry Pi white paper: the Pi 5 relies on **libx264** on the Cortex-A76. It handles **1080p30 in real time at ~60% of one core** in low-latency mode, and high-quality mode beats the Pi 4 hardware encoder by 2–3 dB PSNR at 0.5–5 Mb/s — [RP-010033 H.264 encoding performance on Pi 5](https://pip-assets.raspberrypi.com/categories/685-app-notes-guides-whitepapers/documents/RP-010033-WP-1-H.264%20encoding%20performance%20on%20Raspberry%20Pi%205_series%20computers.pdf). Full PDF fetch timed out; figures are from the search excerpt of this primary document.
- **GIL and multiprocessing**:
  - Multi-threading often works because heavy work in "large (typically) C/C++ functions... will normally release the GIL" (OpenCV, neural network evaluation).
  - For pure-Python bottlenecks, Picamera2 offers `Pool`/`Process` and `RemoteRequest`, where image buffers are passed by file-descriptor mapping, "so these potentially very large amounts of data become 'zero copy'". The parent holds the request until the child finishes, so `buffer_count` must be increased (examples use 4).
  - Do not start, stop or reconfigure the camera from multiple threads without locking.
  - Source: [Picamera2 manual §8.8](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf)
- On Pi 5, Picamera2 does not use CMA memory for camera buffers, which removes a common Pi 4 allocation failure — [Picamera2 manual](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf)
- **Streaming**:
  - Picamera2 ships `mjpeg_server.py` (simple MJPEG web page) plus TCP/UDP/PyAV stream examples.
  - It recommends **MediaMTX** (`source: rpiCamera`, e.g. 10 Mbps). Browsers get WebRTC at `http://<ip>:8889/cam`, RTSP at `:8554/cam`, and `sourceOnDemand: yes` streams only when viewed.
  - An "Advanced Streaming with MediaMTX" section shows feeding annotated frames from Python.
  - Source: [Picamera2 manual §9.1](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-6-picamera2-manual.pdf)
- Jeff Geerling benchmarked TFLite YOLOv5 and MobileNetV2 with picamera2 on Pi 5 vs Pi 4 — [Jeff Geerling 2024](https://www.jeffgeerling.com/blog/2024/testing-object-detection-yolo-mobilenet-etc-picamera2-on-pi-5/) (specific fps numbers not extracted).
- **Flask-SocketIO / eventlet**:
  - Eventlet "is not actively maintained anymore", so gevent should be preferred.
  - Threading is "the easier and most compatible solution", with full WebSocket support (Gunicorn multi-threaded mode for production).
  - Auto-detection prefers eventlet if installed, so set `async_mode` explicitly.
  - Source: [Flask-SocketIO docs](https://flask-socketio.readthedocs.io/en/latest/intro.html)
  - Broader ecosystem: OpenStack deprecated its eventlet executor ahead of eventlet's removal — [oslo.messaging commit](https://gitea10.opendev.org:3081/openstack/oslo.messaging/commit/60f7bc2012ccd47dd671c33deaa476fdc9f68873)
- **GPS time sync**:
  - Typical stack: `pps-tools` + `gpsd` + `chrony`, with an SHM refclock for NMEA (offset tuned, e.g. 0.325 s) and a PPS refclock on `/dev/pps0` via the `pps-gpio` overlay (GPIO 18) — [austinsnerdythings 2021](https://austinsnerdythings.com/2021/04/19/microsecond-accurate-ntp-with-a-raspberry-pi-and-pps-gps/)
  - A Pi 5 with a u-blox timing GPS plus PPS typically holds single-digit-ns offsets (2025 test) — [austinsnerdythings 2025](https://austinsnerdythings.com/2025/02/14/revisiting-microsecond-accurate-ntp-for-raspberry-pi-with-gps-pps-in-2025/)
  - Debug with `gpspipe -r` and `ppstest`; chrony may take minutes to trust PPS — [raspberry.tips](https://raspberry.tips/en/?p=9753)
- **Watchdog**: the Pi 5 bootloader's `BOOT_WATCHDOG_TIMEOUT` only covers the bootloader and "does not monitor the OS after the handover". An OS-level watchdog (systemd `RuntimeWatchdogSec`, or the `WatchdogSec` service setting with `sd_notify`) is still needed — [RPi computer docs](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- The Pi 5 has an on-board RTC (section listed in the docs; needs an external battery to keep time unpowered) — [RPi computer docs](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html). Details not extracted.
- FishCam: microSDXC (200 GB) was more reliable and lower-power than USB drives on boats; it also disabled Bluetooth and ran headless to save power — [Mouy et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041213/)

### Inferences
**Proposed architecture** for a Pi 5 with 16 GB RAM:
1. **Capture**: Picamera2 video config with `main` 1280x720 or 1536x864 YUV420 (to the H.264 encoder via `PyavOutput` in 5–10 min segments) and `lores` 640x360/640x480 **RGB888** (Pi 5) for the detector. Use fixed exposure and frame rate (Section 2) and `buffer_count` ≥ 6.
2. **Inference worker**: a thread calling ONNX Runtime, OpenCV DNN or NCNN (all release the GIL), taking only the *latest* lores frame (drop-old queue of size 1) so latency stays bounded. If the post-processing is heavy pure Python, switch to Picamera2 `Pool` with `RemoteRequest` for zero-copy.
3. **Recorder**:
   - Continuous low-bitrate H.264 of `main` for re-labelling.
   - Or `CircularOutput2` to save ±N seconds around detections, which saves storage.
   - Also save the raw lores frames plus JSON detections for hard-negative mining, which matches the project's E2 hard-negative workflow.
4. **Metadata**:
   - Per-frame `SensorTimestamp` from the request metadata, mapped to UTC by chrony (GPS+PPS).
   - gpsd client thread for lat/lon/speed.
   - Write detections as JSONL or SQLite per segment (frame_idx, UTC, lat, lon, boxes, scores, temperature, throttle flags).
5. **Dashboard**:
   - FastAPI/Starlette (uvicorn) with a WebSocket pushing detection JSON at ~1–5 Hz, plus a low-fps (2–5 fps) MJPEG endpoint of annotated lores frames.
   - Or MediaMTX WebRTC for full-rate live video with a separate detection overlay.
   - If staying on Flask-SocketIO, use `async_mode="threading"` and uninstall eventlet.
   - Keep the dashboard on a separate core budget.
6. **Ops**:
   - systemd service with `Restart=always` and `WatchdogSec` (Python `sdnotify`), plus systemd `RuntimeWatchdogSec` for the hardware watchdog.
   - Read-only root or overlayfs to protect against power loss.
   - Recordings on NVMe (Pi 5 PCIe, via M.2 HAT) or a high-endurance microSD.
   - OTA updates via `git pull` and `systemctl restart` over the tether, or a Docker image (as BlueOS does).
- **[calc] CPU budget on 4 x A76**:
  - Software H.264 at 720p30: ~0.3–0.5 core (scaled from the white paper's 1080p30 ≈ 0.6 core).
  - Dashboard MJPEG at 640x360 @ 5 fps: < 0.1 core.
  - That leaves roughly 2.5–3 cores for detection.
- **C++/GStreamer alternative**: `rpicam-apps` (C++) with a custom post-processing stage, or GStreamer with `libcamerasrc`, reduces Python overhead. For a student project, Python with Picamera2 is the more maintainable choice.

### Gaps
- No Pi 5 CPU-only fps numbers for YOLOv8n/YOLO11n at 320/640 were extracted. This is likely covered by another researcher, but the architecture's latency budget depends on it.
- The CPU overhead of MediaMTX WebRTC versus MJPEG on Pi 5 was not measured in any source found.
- No source fetched on FastAPI vs Flask-SocketIO CPU overhead on Pi 5; that recommendation is an inference.
- OTA tooling (Mender, RAUC, BlueOS extension updates) was not researched in depth.

---

## 6. Existing open-source Pi-based underwater camera projects and lessons

### Takeaway
There is a solid lineage of Pi-based underwater cameras (FishCam, PlasPi, DEEPi, SuMOS, TrigCam, a URI BRUV). Most are **recorders on Pi Zero-class boards**, not real-time detectors. BlueOS (Blue Robotics) now officially supports the Pi 5, with "limited testing", and is the de-facto ROV companion software. Recurring lessons: commercial housings beat DIY PVC for reliability, storage rather than battery often limits deployments, ballast and anti-drag matter, external lighting is needed beyond shallow daylight, and timekeeping needs an RTC or GPS.

### Cited findings
- **FishCam** (Mouy et al., HardwareX 2020):
  - Pi Zero W, Pi camera v2 with 110° lens, WittyPi Mini for RTC and duty cycling; < $500.
  - Recorded 80–212 h over 8–14 days.
  - Lessons: housing leaks, ballast, a supercapacitor RTC holding time only ~17 h unpowered, and memory-limited (not battery-limited) operation.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9041213/)
- **PlasPi** (Purser et al., HardwareX 2020): Pi Zero W, off-the-shelf and 3D-printed parts, < €200, < 150 m — [AWI PDF](https://epic2-clone.awi.de/52529/1/Purseretal2020HardwareX7.pdf)
- **SuMOS** (Wira & Place, HardwareX, Dec 2024): Pi Zero 2 W with CM3 NoIR and IR illumination, standalone low-cost recorder for aquatic substrates — [DOAJ](https://doaj.org/article/96b325e157514225981019d56d9fcc75)
- **DEEPi** (Phillips et al., Deep-Sea Research I, 2019): low-cost Pi-based deep-sea camera; the next generation was planned on StereoPi — [StereoPi summary](https://www.stereopi.com/node/150)
- **TrigCam**: a stereo camera built from off-the-shelf parts and a Raspberry Pi that triggers when animals enter view; it supports fish size and position estimates — [Tethys/PNNL](https://tethys.pnnl.gov/publications/underwater-stereo-camera-trap)
- **URI Narragansett Bay BRUV** (2023): a Pi with a Pi camera in an off-the-shelf **Blue Robotics** pressure case, recording fish abundance, arrival timing and behaviour — [URI RI NSF EPSCoR](https://web.uri.edu/rinsfepscor/2023/11/15/real-time-monitoring-of-underwater-fauna-using-baited-underwater-cameras-p10)
- Low-cost stereo-video designs (handheld and pelagic BRUV), with drawings and BOM, by Lucas, Araujo et al. — [Figshare](https://figshare.com/articles/online_resource/Low-cost_stereo_video_design_drawings_Handheld_system_and_pelagic_BRUVS/27130239)
- A 2022 Pi-based low-cost underwater camera for aquarium and aquaculture fish monitoring — [JACIII, Fuji Press](https://www.fujipress.jp/jaciii/jc/jacii002600050851/)
- **BlueOS**:
  - Official docs list compatibility with Pi 3, 4 and 5; the **Pi 5 image is "Limited testing"**; release 1.4 notes "Added Raspberry Pi 5 support" — [BlueOS installation docs](https://blueos.cloud/docs/latest/usage/installation/)
  - Earlier forum threads: the Navigator flight-controller HAT needed hardware changes and a Pi 5-compatible heatsink and fan — [BR forum: Pi 5 with Navigator](https://discuss.bluerobotics.com/t/pi-5-with-navigator/15903), [BlueOS on Pi 5](https://discuss.bluerobotics.com/t/blueos-on-raspbery-pi-5/16815)

### Inferences
- For a small ROV built on Blue Robotics parts, running the detector as a **BlueOS extension** (Docker) on a Pi 5 is a viable integration path. BlueOS already handles video streaming, the tether network and updates. For a boat-mounted standalone camera, plain Raspberry Pi OS with a custom systemd service is simpler.
- None of the cited projects do real-time on-device detection on a Pi 5 underwater. This project's combination (Pi 5 CPU-only detection, GPS-tagged, live dashboard) appears to be a genuine gap and a publishable contribution, though this is not exhaustively verified.
- Use a hardware RTC battery (the Pi 5 supports one) or GPS time to avoid FishCam's timekeeping limitation.

### Gaps
- "KELPCam" could not be confirmed as an existing published system; no source was found.
- OpenROV is discontinued, and no 2023–2026 source on it was retrieved.
- Full texts of SuMOS, PlasPi and DEEPi were not read, so their thermal and condensation lessons were not extracted.
