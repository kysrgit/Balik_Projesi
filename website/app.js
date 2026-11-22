/**
 * Balon Balığı Tespit Sistemi — Ana Etkileşim ve Erişilebilirlik Betiği (website/app.js)
 *
 * Bu dosya; GPU hızlandırmalı kaydırma ilerleme çubuğu, aktif bölüm takibi
 * (scroll-spy), tema değiştirici, erişilebilir mobil çekmece menü, tür ve galeri
 * filtreleme, JS'siz okunabilir benchmark sekmeleri, 4 sekmeli canlı ürün ekranı
 * vitrini, dokunmatik kaydırma destekli lightbox modali, yerel <details> SSS
 * uyumu, kod kopyalama ve görünürlük gözlemcisini bağımlılıksız vanilla ES6+
 * JavaScript ile yönetir. Tüm içerik doğrudan DOM özniteliklerinden okunur ve
 * yalnızca metin tabanlı güvenli DOM API'leri kullanılır.
 */
(function () {
  'use strict';

  const THEME_STORAGE_KEY = 'balik-projesi-theme';
  const THEME_COLOR_LIGHT = '#f4f8fc';
  const THEME_COLOR_DARK = '#030912';
  const MOBILE_BREAKPOINT = 1180;
  const SWIPE_THRESHOLD_PX = 48;

  // v2 sayfa bölümleri (scroll-spy gerçek DOM dikey sırasına göre hesaplanır)
  const SECTION_IDS = [
    'hero',
    'problem',
    'species',
    'benchmarks',
    'application',
    'roadmap',
    'gallery',
    'docs',
  ];

  // theme-init.js olmadan yüklenme ihtimaline karşı js-enabled sınıfını güvenli biçimde garanti altına al
  if (typeof document !== 'undefined' && document.documentElement) {
    document.documentElement.classList.add('js-enabled');
  }

  /**
   * Kullanıcının azaltılmış hareket tercihini güvenli biçimde okur.
   * @returns {boolean}
   */
  function prefersReducedMotion() {
    return (
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    );
  }

  /**
   * localStorage okuma/yazma işlemlerini gizli sekme ve erişim kısıtlarına karşı korur.
   * @returns {string|null}
   */
  function readStoredTheme() {
    try {
      return window.localStorage.getItem(THEME_STORAGE_KEY);
    } catch (_) {
      return null;
    }
  }

  /**
   * Seçilen temayı güvenli biçimde localStorage'a kaydeder.
   * @param {'light'|'dark'} theme
   */
  function writeStoredTheme(theme) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (_) {
      // Depolama erişimi kısıtlı ortamlarda sessizce geç
    }
  }

  /**
   * <meta name="theme-color"> etiketini aktif temayla eşzamanlar.
   * @param {boolean} isLight
   */
  function syncThemeColorMeta(isLight) {
    const themeColorMeta = document.querySelector('meta[name="theme-color"]');
    if (!themeColorMeta) return;
    themeColorMeta.setAttribute(
      'content',
      isLight ? THEME_COLOR_LIGHT : THEME_COLOR_DARK
    );
  }

  /**
   * Bir kapsayıcı içerisindeki görünür ve odaklanabilir öğeleri döndürür.
   * @param {HTMLElement|null} container
   * @returns {HTMLElement[]}
   */
  function getFocusableElements(container) {
    if (!container) return [];
    const selector = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      'summary',
      '[tabindex]:not([tabindex="-1"])',
    ].join(', ');

    return Array.from(container.querySelectorAll(selector)).filter(
      (el) => !el.hasAttribute('hidden') && el.getClientRects().length > 0
    );
  }

  /**
   * Açık modal veya çekmece varken klavye odak tuzağını (Tab / Shift+Tab) yönetir.
   * @param {KeyboardEvent} event
   * @param {HTMLElement|null} container
   */
  function trapFocusInContainer(event, container) {
    if (event.key !== 'Tab' || !container) return;
    const focusables = getFocusableElements(container);
    if (focusables.length === 0) {
      event.preventDefault();
      return;
    }

    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;

    if (event.shiftKey) {
      if (active === first || !container.contains(active)) {
        event.preventDefault();
        last.focus();
      }
    } else if (active === last || !container.contains(active)) {
      event.preventDefault();
      first.focus();
    }
  }

  /**
   * Bir DOM öğesi üzerinde `inert` durumunu tarayıcı uyumlu biçimde ayarlar.
   * @param {HTMLElement} el
   * @param {boolean} shouldBeInert
   */
  function setElementInert(el, shouldBeInert) {
    if (!el) return;
    if ('inert' in el) {
      el.inert = shouldBeInert;
    } else if (shouldBeInert) {
      el.setAttribute('inert', '');
    } else {
      el.removeAttribute('inert');
    }
  }

  /**
   * Paylaşılan modal/çekmece durum yöneticisi: gövde kaydırmasını kilitler ve
   * ana sayfa dönüm noktalarını (<header>, <main>, <footer>) inert yapar.
   */
  function syncModalOverlayState() {
    const drawer = document.getElementById('mobile-drawer');
    const lightbox = document.getElementById('lightbox');
    const drawerOpen = Boolean(drawer && drawer.classList.contains('is-open'));
    const lightboxOpen = Boolean(lightbox && lightbox.classList.contains('is-open'));
    const anyOverlayOpen = drawerOpen || lightboxOpen;

    document.body.style.overflow = anyOverlayOpen ? 'hidden' : '';

    // Yalnızca üst düzey sayfa dönüm noktalarını (<header>, <main>, <footer>) inert yap
    const landmarks = Array.from(
      document.querySelectorAll('body > header, body > main, body > footer')
    );

    landmarks.forEach((landmark) => {
      // Açık katmanı barındıran bir kapsayıcı varsa yanlışlıkla kilitlenmesini engelle
      const containsActiveDrawer = drawerOpen && drawer && landmark.contains(drawer);
      const containsActiveLightbox = lightboxOpen && lightbox && landmark.contains(lightbox);
      const shouldInert = anyOverlayOpen && !containsActiveDrawer && !containsActiveLightbox;
      setElementInert(landmark, shouldInert);
    });
  }

  /**
   * 1. Görünürlük Gözlemcisi (.reveal -> .is-visible)
   */
  function initRevealObserver() {
    const revealEls = Array.from(document.querySelectorAll('.reveal'));
    if (revealEls.length === 0) return;

    if (prefersReducedMotion() || typeof IntersectionObserver !== 'function') {
      revealEls.forEach((el) => el.classList.add('is-visible'));
      return;
    }

    const observer = new IntersectionObserver(
      (entries, io) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        });
      },
      {
        threshold: 0.1,
        rootMargin: '0px 0px -36px 0px',
      }
    );

    revealEls.forEach((el) => {
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight * 0.96) {
        el.classList.add('is-visible');
      } else {
        observer.observe(el);
      }
    });
  }

  /**
   * 2. GPU Hızlandırmalı Kaydırma İlerleme Çubuğu, Yapışkan Başlık ve Aktif Bölüm Takibi
   */
  function initScrollAndSpy() {
    const progressEl = document.getElementById('scroll-progress');
    const headerEl = document.getElementById('site-header');
    const navLinks = Array.from(document.querySelectorAll('.nav-link, .drawer-link'));
    const sections = SECTION_IDS.map((id) => document.getElementById(id)).filter(Boolean);

    let rafPending = false;

    function updateActiveLinks(activeId) {
      if (!activeId || navLinks.length === 0) return;
      navLinks.forEach((link) => {
        const href = link.getAttribute('href') || '';
        const isMatch = href === `#${activeId}` || href.endsWith(`/#${activeId}`);
        link.classList.toggle('is-active', isMatch);
        if (isMatch) {
          link.setAttribute('aria-current', 'location');
        } else {
          link.removeAttribute('aria-current');
        }
      });
    }

    function updateOnScroll() {
      rafPending = false;
      const scrollY = window.scrollY || window.pageYOffset || 0;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;

      // Reflow tetiklemeden yalnızca GPU-composited transform: scaleX(...) kullan
      if (progressEl) {
        const ratio = docHeight > 0 ? Math.min(1, Math.max(0, scrollY / docHeight)) : 0;
        progressEl.style.transform = 'scaleX(' + ratio.toFixed(4) + ')';
        progressEl.setAttribute('aria-valuenow', String(Math.round(ratio * 100)));
      }

      if (headerEl) {
        headerEl.classList.toggle('is-scrolled', scrollY > 24);
      }

      if (sections.length > 0) {
        const headerHeight = headerEl ? headerEl.offsetHeight : 72;
        const probeY = scrollY + headerHeight + 48;

        // Bölümleri gerçek belge dikey konumuna göre sırala (DOM sırası değişse bile doğru çalışır)
        const positioned = sections
          .map((section) => ({
            id: section.id,
            top: section.getBoundingClientRect().top + scrollY,
          }))
          .sort((a, b) => a.top - b.top);

        let currentId = positioned[0].id;

        // Sayfa en alta ulaştığında son bölümü etkinleştir
        if (docHeight > 0 && scrollY >= docHeight - 8) {
          currentId = positioned[positioned.length - 1].id;
        } else {
          for (let i = 0; i < positioned.length; i += 1) {
            if (positioned[i].top <= probeY) {
              currentId = positioned[i].id;
            }
          }
        }

        updateActiveLinks(currentId);
      }
    }

    function scheduleUpdate() {
      if (rafPending) return;
      rafPending = true;
      window.requestAnimationFrame(updateOnScroll);
    }

    window.addEventListener('scroll', scheduleUpdate, { passive: true });
    window.addEventListener('resize', scheduleUpdate, { passive: true });
    updateOnScroll();
  }

  /**
   * 3. Açık / Koyu Tema Değiştirici (#theme-toggle)
   * Not: .icon-sun / .icon-moon ikon geçişi CSS tarafından html[data-theme="light"] ile yönetilir.
   */
  function initThemeToggle() {
    const toggleBtn = document.getElementById('theme-toggle');
    const root = document.documentElement;

    function applyTheme(theme) {
      const isLight = theme === 'light';
      if (isLight) {
        root.setAttribute('data-theme', 'light');
      } else {
        root.removeAttribute('data-theme');
      }

      syncThemeColorMeta(isLight);

      if (toggleBtn) {
        toggleBtn.setAttribute('aria-pressed', String(isLight));
        toggleBtn.setAttribute(
          'aria-label',
          isLight ? 'Koyu temaya geç' : 'Açık temaya geç'
        );
        toggleBtn.setAttribute(
          'title',
          isLight ? 'Koyu temaya geç' : 'Açık temaya geç'
        );

        const labelTarget = toggleBtn.querySelector('[data-theme-label]');
        if (labelTarget) {
          labelTarget.textContent = isLight ? 'Koyu Tema' : 'Açık Tema';
        }
      }
    }

    // Başlangıç durumunu theme-init.js, kayıtlı tercih ve sistem tercihiyle eşitle
    const stored = readStoredTheme();
    if (stored === 'light' || stored === 'dark') {
      applyTheme(stored);
    } else if (root.getAttribute('data-theme') === 'light') {
      applyTheme('light');
    } else if (
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-color-scheme: light)').matches
    ) {
      applyTheme('light');
    } else {
      applyTheme('dark');
    }

    // Kullanıcı manuel seçim yapmadıysa sistem teması değiştiğinde canlı güncelle
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
      const handleSystemChange = (event) => {
        if (readStoredTheme() !== null) return;
        applyTheme(event.matches ? 'light' : 'dark');
      };
      if (typeof mediaQuery.addEventListener === 'function') {
        mediaQuery.addEventListener('change', handleSystemChange);
      }
    }

    if (!toggleBtn) return;

    toggleBtn.addEventListener('click', () => {
      const currentlyLight = root.getAttribute('data-theme') === 'light';
      const nextTheme = currentlyLight ? 'dark' : 'light';
      applyTheme(nextTheme);
      writeStoredTheme(nextTheme);
    });
  }

  /**
   * 4. Mobil Çekmece Menü (#nav-toggle, #mobile-drawer, #drawer-backdrop, #drawer-close)
   */
  function initMobileDrawer() {
    const toggleBtn = document.getElementById('nav-toggle');
    const drawer = document.getElementById('mobile-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    const closeBtn = document.getElementById('drawer-close');
    if (!toggleBtn || !drawer) return;

    // Ekran okuyucular için çekmece diyalog semantiğini garanti altına al
    drawer.setAttribute('role', 'dialog');
    drawer.setAttribute('aria-modal', 'true');

    const drawerLinks = Array.from(drawer.querySelectorAll('.drawer-link'));
    let lastFocused = null;

    function isDrawerOpen() {
      return drawer.classList.contains('is-open');
    }

    function openDrawer() {
      if (isDrawerOpen()) return;
      lastFocused =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : toggleBtn;
      drawer.classList.add('is-open');
      drawer.setAttribute('aria-hidden', 'false');
      if (backdrop) backdrop.classList.add('is-open');
      toggleBtn.setAttribute('aria-expanded', 'true');
      syncModalOverlayState();

      const focusTarget = closeBtn || getFocusableElements(drawer)[0];
      if (focusTarget) {
        window.requestAnimationFrame(() => focusTarget.focus());
      }
    }

    function closeDrawer(restoreFocus) {
      if (!isDrawerOpen()) return;
      drawer.classList.remove('is-open');
      drawer.setAttribute('aria-hidden', 'true');
      if (backdrop) backdrop.classList.remove('is-open');
      toggleBtn.setAttribute('aria-expanded', 'false');
      syncModalOverlayState();

      if (restoreFocus && lastFocused && typeof lastFocused.focus === 'function') {
        lastFocused.focus();
      }
      lastFocused = null;
    }

    toggleBtn.addEventListener('click', () => {
      if (isDrawerOpen()) {
        closeDrawer(true);
      } else {
        openDrawer();
      }
    });

    if (closeBtn) {
      closeBtn.addEventListener('click', () => closeDrawer(true));
    }

    if (backdrop) {
      backdrop.addEventListener('click', () => closeDrawer(true));
    }

    drawerLinks.forEach((link) => {
      link.addEventListener('click', () => closeDrawer(false));
    });

    document.addEventListener('keydown', (event) => {
      if (!isDrawerOpen()) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        closeDrawer(true);
        return;
      }
      trapFocusInContainer(event, drawer);
    });

    // 1180px masaüstü kırılımı üstüne çıkıldığında çekmeceyi otomatik kapat
    window.addEventListener(
      'resize',
      () => {
        if (window.innerWidth > MOBILE_BREAKPOINT && isDrawerOpen()) {
          closeDrawer(false);
        }
      },
      { passive: true }
    );
  }

  /**
   * 5. Tür ve Veri Seti Filtreleme (#species-filters -> #species-grid, #species-empty)
   */
  function initSpeciesFilter() {
    const filterBar = document.getElementById('species-filters');
    const grid = document.getElementById('species-grid');
    if (!filterBar || !grid) return;

    const filterBtns = Array.from(filterBar.querySelectorAll('.filter-btn[data-filter]'));
    const cards = Array.from(grid.querySelectorAll('.species-card'));
    const countEl =
      document.getElementById('species-count') ||
      document.querySelector('[data-species-count]');

    // Mevcut #species-empty öğesini doğrudan kimliğiyle seç (grid içine kopya öğe ekleme)
    const emptyEl = document.getElementById('species-empty');

    function applyFilter(filterValue) {
      const normalized = (filterValue || 'all').trim().toLowerCase();
      let visibleCount = 0;

      cards.forEach((card) => {
        const rawCat = (card.dataset.category || '').trim().toLowerCase();
        const categories = rawCat ? rawCat.split(/[\s,]+/) : [];
        const matches = normalized === 'all' || categories.includes(normalized);
        card.classList.toggle('is-hidden', !matches);
        card.hidden = !matches;
        if (matches) visibleCount += 1;
      });

      filterBtns.forEach((btn) => {
        const isActive = (btn.dataset.filter || '').trim().toLowerCase() === normalized;
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', String(isActive));
      });

      if (countEl) {
        countEl.textContent = `${visibleCount} / ${cards.length} kayıt`;
      }

      if (emptyEl) {
        const isEmpty = visibleCount === 0;
        emptyEl.classList.toggle('is-hidden', !isEmpty);
        emptyEl.hidden = !isEmpty;
      }
    }

    filterBar.addEventListener('click', (event) => {
      const btn =
        event.target instanceof Element
          ? event.target.closest('.filter-btn[data-filter]')
          : null;
      if (!btn || !filterBar.contains(btn)) return;
      applyFilter(btn.dataset.filter);
    });

    const initialBtn =
      filterBtns.find((btn) => btn.classList.contains('is-active')) || filterBtns[0];
    applyFilter(initialBtn ? initialBtn.dataset.filter : 'all');
  }

  /**
   * 6. JS'siz Okunabilir Platform ve Deney Sekmeleri (#platform-tabs)
   * Not: Ham HTML'de panellerde `hidden` bulunmaz; JS açıldığında aktif olmayanlar gizlenir.
   */
  function initPlatformTabs() {
    const tabGroup = document.getElementById('platform-tabs');
    if (!tabGroup) return;

    const tabButtons = Array.from(tabGroup.querySelectorAll('.tab-btn[data-tab]'));
    if (tabButtons.length === 0) return;

    const scope = tabGroup.closest('#benchmarks, section') || document;
    const panels = Array.from(scope.querySelectorAll('.tab-panel[data-panel]'));

    function activateTab(tabId, focusTab) {
      if (!tabId) return;

      tabButtons.forEach((btn) => {
        const isSelected = btn.dataset.tab === tabId;
        btn.classList.toggle('is-active', isSelected);
        btn.setAttribute('aria-selected', String(isSelected));
        btn.tabIndex = isSelected ? 0 : -1;
        if (isSelected && focusTab) {
          btn.focus();
        }
      });

      panels.forEach((panel) => {
        const isMatch = panel.dataset.panel === tabId;
        panel.classList.toggle('is-active', isMatch);
        panel.hidden = !isMatch;
      });
    }

    tabGroup.addEventListener('click', (event) => {
      const btn =
        event.target instanceof Element
          ? event.target.closest('.tab-btn[data-tab]')
          : null;
      if (!btn || !tabGroup.contains(btn)) return;
      activateTab(btn.dataset.tab, false);
    });

    tabGroup.addEventListener('keydown', (event) => {
      const currentBtn =
        event.target instanceof Element
          ? event.target.closest('.tab-btn[data-tab]')
          : null;
      if (!currentBtn) return;

      const currentIndex = tabButtons.indexOf(currentBtn);
      if (currentIndex === -1) return;

      let nextIndex = -1;
      if (event.key === 'ArrowRight') {
        nextIndex = (currentIndex + 1) % tabButtons.length;
      } else if (event.key === 'ArrowLeft') {
        nextIndex = (currentIndex - 1 + tabButtons.length) % tabButtons.length;
      } else if (event.key === 'Home') {
        nextIndex = 0;
      } else if (event.key === 'End') {
        nextIndex = tabButtons.length - 1;
      }

      if (nextIndex !== -1) {
        event.preventDefault();
        activateTab(tabButtons[nextIndex].dataset.tab, true);
      }
    });

    // Önce HTML'de `.is-active` işaretli paneli kontrol et, yoksa ilk sekmeden başla
    const activePanel = panels.find((panel) => panel.classList.contains('is-active'));
    const activeButton = tabButtons.find(
      (btn) =>
        btn.classList.contains('is-active') ||
        btn.getAttribute('aria-selected') === 'true'
    );
    const initialTabId =
      (activePanel && activePanel.dataset.panel) ||
      (activeButton && activeButton.dataset.tab) ||
      (tabButtons[0] && tabButtons[0].dataset.tab);

    if (initialTabId) {
      activateTab(initialTabId, false);
    }
  }

  /**
   * 7. 4 Sekmeli Canlı Ürün Arayüzü Vitrini (#app-screen-tabs -> [data-screen-panel])
   */
  function initAppScreenTabs() {
    const tabGroup = document.getElementById('app-screen-tabs');
    if (!tabGroup) return;

    const tabButtons = Array.from(
      tabGroup.querySelectorAll('.app-screen-tab[data-screen]')
    );
    if (tabButtons.length === 0) return;

    const scope =
      tabGroup.closest('.app-showcase-panel, #application, section') || document;
    const panels = Array.from(scope.querySelectorAll('[data-screen-panel]'));

    function activateScreen(screenId, focusTab) {
      if (!screenId) return;

      tabButtons.forEach((btn) => {
        const isSelected = btn.dataset.screen === screenId;
        btn.classList.toggle('is-active', isSelected);
        btn.setAttribute('aria-selected', String(isSelected));
        btn.tabIndex = isSelected ? 0 : -1;
        if (isSelected && focusTab) {
          btn.focus();
        }
      });

      panels.forEach((panel) => {
        const isMatch = panel.dataset.screenPanel === screenId;
        panel.classList.toggle('is-active', isMatch);
        panel.classList.toggle('is-hidden', !isMatch);
        panel.hidden = !isMatch;
      });
    }

    tabGroup.addEventListener('click', (event) => {
      const btn =
        event.target instanceof Element
          ? event.target.closest('.app-screen-tab[data-screen]')
          : null;
      if (!btn || !tabGroup.contains(btn)) return;
      activateScreen(btn.dataset.screen, false);
    });

    tabGroup.addEventListener('keydown', (event) => {
      const currentBtn =
        event.target instanceof Element
          ? event.target.closest('.app-screen-tab[data-screen]')
          : null;
      if (!currentBtn) return;

      const currentIndex = tabButtons.indexOf(currentBtn);
      if (currentIndex === -1) return;

      let nextIndex = -1;
      if (event.key === 'ArrowRight') {
        nextIndex = (currentIndex + 1) % tabButtons.length;
      } else if (event.key === 'ArrowLeft') {
        nextIndex = (currentIndex - 1 + tabButtons.length) % tabButtons.length;
      } else if (event.key === 'Home') {
        nextIndex = 0;
      } else if (event.key === 'End') {
        nextIndex = tabButtons.length - 1;
      }

      if (nextIndex !== -1) {
        event.preventDefault();
        activateScreen(tabButtons[nextIndex].dataset.screen, true);
      }
    });

    // Başlangıçta aktif ekranı seç ve diğer <figure> panellerini gizle
    const activeButton = tabButtons.find(
      (btn) =>
        btn.classList.contains('is-active') ||
        btn.getAttribute('aria-selected') === 'true'
    );
    const activePanel = panels.find((panel) => panel.classList.contains('is-active'));
    const initialScreenId =
      (activeButton && activeButton.dataset.screen) ||
      (activePanel && activePanel.dataset.screenPanel) ||
      (tabButtons[0] && tabButtons[0].dataset.screen);

    if (initialScreenId) {
      activateScreen(initialScreenId, false);
    }
  }

  /**
   * 8. Galeri Filtreleme ve Dokunmatik Kaydırma Destekli Erişilebilir Lightbox
   *    (#gallery-filters, #gallery-grid, #gallery-empty, #lightbox)
   */
  function initGalleryAndLightbox() {
    const filterBar = document.getElementById('gallery-filters');
    const grid = document.getElementById('gallery-grid');
    const lightbox = document.getElementById('lightbox');

    const galleryItems = grid
      ? Array.from(grid.querySelectorAll('.gallery-item'))
      : [];
    const filterBtns = filterBar
      ? Array.from(filterBar.querySelectorAll('.filter-btn[data-filter]'))
      : [];
    const countEl =
      document.getElementById('gallery-count') ||
      document.querySelector('[data-gallery-count]');

    // Mevcut #gallery-empty öğesini doğrudan kimliğiyle seç (grid içine kopya öğe ekleme)
    const galleryEmptyEl = document.getElementById('gallery-empty');

    function applyGalleryFilter(filterValue) {
      if (!grid) return;
      const normalized = (filterValue || 'all').trim().toLowerCase();
      let visibleCount = 0;

      galleryItems.forEach((item) => {
        const rawCat = (item.dataset.category || '').trim().toLowerCase();
        const categories = rawCat ? rawCat.split(/[\s,]+/) : [];
        const matches = normalized === 'all' || categories.includes(normalized);
        item.classList.toggle('is-hidden', !matches);
        item.hidden = !matches;
        if (matches) visibleCount += 1;
      });

      filterBtns.forEach((btn) => {
        const isActive = (btn.dataset.filter || '').trim().toLowerCase() === normalized;
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', String(isActive));
      });

      if (countEl) {
        countEl.textContent = `${visibleCount} / ${galleryItems.length} görsel`;
      }

      if (galleryEmptyEl) {
        const isEmpty = visibleCount === 0;
        galleryEmptyEl.classList.toggle('is-hidden', !isEmpty);
        galleryEmptyEl.hidden = !isEmpty;
      }
    }

    if (filterBar && grid) {
      filterBar.addEventListener('click', (event) => {
        const btn =
          event.target instanceof Element
            ? event.target.closest('.filter-btn[data-filter]')
            : null;
        if (!btn || !filterBar.contains(btn)) return;
        applyGalleryFilter(btn.dataset.filter);
      });

      const initialBtn =
        filterBtns.find((btn) => btn.classList.contains('is-active')) || filterBtns[0];
      applyGalleryFilter(initialBtn ? initialBtn.dataset.filter : 'all');
    }

    // Lightbox kurulumu
    if (!lightbox || !grid) return;

    const backdrop = document.getElementById('lightbox-backdrop');
    const dialog = lightbox.querySelector('.lightbox-dialog') || lightbox;
    const closeBtn = document.getElementById('lightbox-close');
    const prevBtn = document.getElementById('lightbox-prev');
    const nextBtn = document.getElementById('lightbox-next');
    const imgEl = document.getElementById('lightbox-img');
    const titleEl = document.getElementById('lightbox-title');
    const captionEl = document.getElementById('lightbox-caption');
    const metaEl = document.getElementById('lightbox-meta');
    const counterEl = document.getElementById('lightbox-counter');

    // Lightbox ilk kez açılana kadar görselin gereksiz yere indirilmesini engelle
    if (imgEl && !lightbox.classList.contains('is-open')) {
      imgEl.removeAttribute('src');
    }

    let activeItems = [];
    let currentIndex = 0;
    let lastFocused = null;
    let touchStartX = 0;
    let touchStartY = 0;
    let touchTracking = false;

    function getVisibleGalleryItems() {
      return galleryItems.filter(
        (item) => !item.classList.contains('is-hidden') && !item.hidden
      );
    }

    /**
     * Galeri öğesinin görsel kaynağını ve metin meta verilerini yalnızca DOM'dan okur.
     * @param {HTMLElement} item
     */
    function extractItemPayload(item) {
      const thumbImg = item.querySelector('img');
      const fallbackSrc = thumbImg ? thumbImg.getAttribute('src') || '' : '';
      const src = item.dataset.src || fallbackSrc;

      const titleNode = item.querySelector('.gallery-title');
      const title =
        item.dataset.title ||
        (titleNode ? titleNode.textContent.trim() : '') ||
        'Proje Görseli';

      const caption =
        item.dataset.caption ||
        (thumbImg ? thumbImg.getAttribute('alt') || '' : '');

      const metaNode = item.querySelector('.gallery-meta');
      const rawMeta =
        item.dataset.meta ||
        (metaNode ? metaNode.textContent.trim() : '');

      return { src, title, caption, rawMeta };
    }

    /**
     * `|`, `•` ve `·` ayraçlarıyla verilmiş meta verilerini XSS-güvenli biçimde işler.
     * @param {string} rawMeta
     */
    function renderMetaList(rawMeta) {
      if (!metaEl) return;
      const parts = rawMeta
        ? rawMeta
            .split(/\s*[|•·]\s*/)
            .map((part) => part.trim())
            .filter(Boolean)
        : [];

      const isListElement = metaEl.tagName === 'UL' || metaEl.tagName === 'OL';
      if (isListElement) {
        metaEl.replaceChildren();
        parts.forEach((part) => {
          const li = document.createElement('li');
          li.textContent = part;
          metaEl.appendChild(li);
        });
        metaEl.hidden = parts.length === 0;
      } else {
        metaEl.textContent = parts.join(' · ');
        metaEl.hidden = parts.length === 0;
      }
    }

    function renderSlide(index) {
      if (activeItems.length === 0) return;
      const total = activeItems.length;
      currentIndex = ((index % total) + total) % total;
      const item = activeItems[currentIndex];
      const payload = extractItemPayload(item);

      if (imgEl) {
        if (payload.src) {
          imgEl.setAttribute('src', payload.src);
        } else {
          imgEl.removeAttribute('src');
        }
        imgEl.setAttribute('alt', payload.caption || payload.title);
      }
      if (titleEl) {
        titleEl.textContent = payload.title;
      }
      if (captionEl) {
        captionEl.textContent = payload.caption;
      }
      renderMetaList(payload.rawMeta);

      if (counterEl) {
        counterEl.textContent = `${currentIndex + 1} / ${total}`;
      }

      const disableNav = total <= 1;
      if (prevBtn) prevBtn.disabled = disableNav;
      if (nextBtn) nextBtn.disabled = disableNav;
    }

    function isLightboxOpen() {
      return lightbox.classList.contains('is-open');
    }

    function openLightboxFromItem(triggerItem) {
      activeItems = getVisibleGalleryItems();
      if (activeItems.length === 0) return;

      const idx = activeItems.indexOf(triggerItem);
      lastFocused =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : triggerItem;

      renderSlide(idx >= 0 ? idx : 0);
      lightbox.classList.add('is-open');
      lightbox.setAttribute('aria-hidden', 'false');
      syncModalOverlayState();

      const focusTarget = closeBtn || nextBtn || getFocusableElements(dialog)[0];
      if (focusTarget) {
        window.requestAnimationFrame(() => focusTarget.focus());
      }
    }

    function closeLightbox() {
      if (!isLightboxOpen()) return;
      lightbox.classList.remove('is-open');
      lightbox.setAttribute('aria-hidden', 'true');

      // Sonraki açılışta eski kare parlamasını önlemek için src niteliğini temizle
      if (imgEl) {
        imgEl.removeAttribute('src');
      }

      syncModalOverlayState();

      if (lastFocused && typeof lastFocused.focus === 'function') {
        lastFocused.focus();
      }
      lastFocused = null;
    }

    grid.addEventListener('click', (event) => {
      const item =
        event.target instanceof Element
          ? event.target.closest('.gallery-item')
          : null;
      if (!item || !grid.contains(item) || item.classList.contains('is-hidden')) {
        return;
      }
      openLightboxFromItem(item);
    });

    if (prevBtn) {
      prevBtn.addEventListener('click', () => renderSlide(currentIndex - 1));
    }

    if (nextBtn) {
      nextBtn.addEventListener('click', () => renderSlide(currentIndex + 1));
    }

    if (closeBtn) {
      closeBtn.addEventListener('click', closeLightbox);
    }

    if (backdrop) {
      backdrop.addEventListener('click', closeLightbox);
    }

    // Mobil dokunmatik yatay kaydırma (swipe) ile önceki / sonraki slayt geçişi
    dialog.addEventListener(
      'touchstart',
      (event) => {
        if (!isLightboxOpen() || event.touches.length !== 1) {
          touchTracking = false;
          return;
        }
        touchStartX = event.touches[0].clientX;
        touchStartY = event.touches[0].clientY;
        touchTracking = true;
      },
      { passive: true }
    );

    dialog.addEventListener(
      'touchcancel',
      () => {
        touchTracking = false;
      },
      { passive: true }
    );

    dialog.addEventListener(
      'touchend',
      (event) => {
        if (!touchTracking || !isLightboxOpen() || activeItems.length <= 1) {
          touchTracking = false;
          return;
        }
        touchTracking = false;

        const touch = event.changedTouches && event.changedTouches[0];
        if (!touch) return;

        const deltaX = touch.clientX - touchStartX;
        const deltaY = touch.clientY - touchStartY;

        // Yalnızca yatay eksende eşiği (>= 48px) aşan belirgin kaydırmaları slayt değişimi say
        if (
          Math.abs(deltaX) >= SWIPE_THRESHOLD_PX &&
          Math.abs(deltaX) > Math.abs(deltaY)
        ) {
          if (deltaX < 0) {
            renderSlide(currentIndex + 1);
          } else {
            renderSlide(currentIndex - 1);
          }
        }
      },
      { passive: true }
    );

    document.addEventListener('keydown', (event) => {
      if (!isLightboxOpen()) return;

      if (event.key === 'Escape') {
        event.preventDefault();
        closeLightbox();
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        renderSlide(currentIndex - 1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        renderSlide(currentIndex + 1);
      } else {
        trapFocusInContainer(event, dialog);
      }
    });
  }

  /**
   * 9. SSS Akordeonu (#faq-accordion)
   * Yerel <details class="accordion-item"> öğelerini bozmadan destekler;
   * varsa eski düğme tabanlı akordeon öğeleriyle geriye dönük uyumluluk sağlar.
   */
  function initFaqAccordion() {
    const accordion = document.getElementById('faq-accordion');
    if (!accordion) return;

    const detailsItems = Array.from(
      accordion.querySelectorAll('details.accordion-item')
    );

    // Yerel <details> kullanılıyorsa tarayıcı yerel aç/kapa davranışını koru
    if (detailsItems.length > 0) {
      detailsItems.forEach((detail) => {
        detail.classList.toggle('is-open', detail.open);
        detail.addEventListener('toggle', () => {
          detail.classList.toggle('is-open', detail.open);
        });
      });
      return;
    }

    // Eski <button class="accordion-trigger"> yapısı için geriye dönük destek
    const legacyItems = Array.from(accordion.querySelectorAll('.accordion-item'));

    function setItemState(item, trigger, panel, isOpen) {
      item.classList.toggle('is-open', isOpen);
      if (trigger) {
        trigger.setAttribute('aria-expanded', String(isOpen));
      }
      if (panel) {
        panel.hidden = !isOpen;
      }
    }

    legacyItems.forEach((item) => {
      const trigger = item.querySelector('button.accordion-trigger');
      const panelId = trigger ? trigger.getAttribute('aria-controls') : null;
      const panel =
        (panelId && document.getElementById(panelId)) ||
        item.querySelector('.accordion-panel');
      const initialOpen =
        item.classList.contains('is-open') ||
        (trigger ? trigger.getAttribute('aria-expanded') === 'true' : false);
      setItemState(item, trigger, panel, initialOpen);
    });

    accordion.addEventListener('click', (event) => {
      const trigger =
        event.target instanceof Element
          ? event.target.closest('button.accordion-trigger')
          : null;
      if (!trigger || !accordion.contains(trigger)) return;

      const item = trigger.closest('.accordion-item');
      if (!item) return;

      const panelId = trigger.getAttribute('aria-controls');
      const panel =
        (panelId && document.getElementById(panelId)) ||
        item.querySelector('.accordion-panel');
      const currentlyOpen = item.classList.contains('is-open');
      setItemState(item, trigger, panel, !currentlyOpen);
    });
  }

  /**
   * 10. Kod Kartı Kopyalama (.copy-btn -> pre.code-block)
   */
  function fallbackCopyText(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.top = '-9999px';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();

    let copied = false;
    try {
      copied = document.execCommand('copy');
    } catch (_) {
      copied = false;
    }

    document.body.removeChild(textarea);
    return copied;
  }

  function initCopyButtons() {
    const copyButtons = Array.from(document.querySelectorAll('.copy-btn'));
    if (copyButtons.length === 0) return;

    const resetTimers = new WeakMap();

    function showCopyFeedback(btn, success) {
      if (!btn.dataset.originalLabel) {
        btn.dataset.originalLabel = btn.textContent.trim() || 'Kopyala';
      }

      const activeTimer = resetTimers.get(btn);
      if (activeTimer) {
        window.clearTimeout(activeTimer);
      }

      btn.textContent = success ? 'Kopyalandı' : 'Hata';
      if (success) {
        btn.setAttribute('data-copied', 'true');
      } else {
        btn.removeAttribute('data-copied');
      }

      const timer = window.setTimeout(() => {
        btn.textContent = btn.dataset.originalLabel || 'Kopyala';
        btn.removeAttribute('data-copied');
        resetTimers.delete(btn);
      }, 2000);

      resetTimers.set(btn, timer);
    }

    copyButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const card = btn.closest('.code-card');
        const codeBlock = card
          ? card.querySelector('pre.code-block, pre')
          : null;
        if (!codeBlock) return;

        const textToCopy = (codeBlock.textContent || '').trim();
        if (!textToCopy) return;

        if (
          window.isSecureContext &&
          navigator.clipboard &&
          typeof navigator.clipboard.writeText === 'function'
        ) {
          navigator.clipboard
            .writeText(textToCopy)
            .then(() => showCopyFeedback(btn, true))
            .catch(() => {
              const ok = fallbackCopyText(textToCopy);
              showCopyFeedback(btn, ok);
            });
        } else {
          const ok = fallbackCopyText(textToCopy);
          showCopyFeedback(btn, ok);
        }
      });
    });
  }

  /**
   * 11. Yukarı Çık Düğmesi (.back-to-top)
   */
  function initBackToTop() {
    const backToTopButtons = Array.from(document.querySelectorAll('.back-to-top'));
    if (backToTopButtons.length === 0) return;

    backToTopButtons.forEach((btn) => {
      btn.addEventListener('click', (event) => {
        event.preventDefault();
        window.scrollTo({
          top: 0,
          behavior: prefersReducedMotion() ? 'auto' : 'smooth',
        });
      });
    });
  }

  /**
   * Tüm etkileşim modüllerini DOM hazır olduğunda başlatır.
   */
  function bootstrap() {
    initRevealObserver();
    initScrollAndSpy();
    initThemeToggle();
    initMobileDrawer();
    initSpeciesFilter();
    initPlatformTabs();
    initAppScreenTabs();
    initGalleryAndLightbox();
    initFaqAccordion();
    initCopyButtons();
    initBackToTop();
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', bootstrap);
    } else {
      bootstrap();
    }
  }
})();
