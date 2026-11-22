/**
 * Balon Balığı Tespit Sistemi — Ana Etkileşim ve Erişilebilirlik Betiği
 *
 * Bu dosya; tema değiştirme, kaydırma ilerleme çubuğu, aktif bölüm takibi
 * (scroll-spy), mobil çekmece menü, tür ve galeri filtreleme, platform sekmeleri,
 * erişilebilir lightbox modali, SSS akordeonu, kod kopyalama ve görünürlük
 * animasyonlarını bağımlılıksız vanilla ES6+ JavaScript ile yönetir.
 */
(function () {
  'use strict';

  const THEME_STORAGE_KEY = 'balik-projesi-theme';
  const SECTION_IDS = [
    'hero',
    'problem',
    'species',
    'benchmarks',
    'architecture',
    'application',
    'gallery',
    'docs',
  ];

  // JS aktif sınıfını DOM hazır olmadan hemen ekle ki .reveal geçişleri flaş yapmasın
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
   */
  function readStoredTheme() {
    try {
      return window.localStorage.getItem(THEME_STORAGE_KEY);
    } catch (_) {
      return null;
    }
  }

  function writeStoredTheme(theme) {
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (_) {
      // Depolama erişimi kısıtlı ortamlarda sessizce geç
    }
  }

  /**
   * Bir kapsayıcı içerisindeki odaklanabilir öğeleri döndürür.
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
      '[tabindex]:not([tabindex="-1"])',
    ].join(', ');

    return Array.from(container.querySelectorAll(selector)).filter(
      (el) => !el.hasAttribute('hidden') && el.getClientRects().length > 0
    );
  }

  /**
   * Açık modal veya çekmece varken odak tuzağını (Tab / Shift+Tab) yönetir.
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
   * 2. Kaydırma İlerleme Çubuğu, Yapışkan Başlık ve Aktif Bölüm Takibi (Scroll-Spy)
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

      if (progressEl) {
        const percent = docHeight > 0 ? Math.min(100, Math.max(0, (scrollY / docHeight) * 100)) : 0;
        progressEl.style.width = `${percent.toFixed(2)}%`;
        if (progressEl.getAttribute('role') === 'progressbar') {
          progressEl.setAttribute('aria-valuenow', String(Math.round(percent)));
        }
      }

      if (headerEl) {
        headerEl.classList.toggle('is-scrolled', scrollY > 24);
      }

      if (sections.length > 0) {
        const headerHeight = headerEl ? headerEl.offsetHeight : 72;
        const probeY = scrollY + headerHeight + 48;
        let currentId = sections[0].id;

        // Sayfa en alta ulaştığında son bölümü etkinleştir
        if (docHeight > 0 && scrollY >= docHeight - 8) {
          currentId = sections[sections.length - 1].id;
        } else {
          for (let i = 0; i < sections.length; i += 1) {
            if (sections[i].offsetTop <= probeY) {
              currentId = sections[i].id;
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

    const stored = readStoredTheme();
    if (stored === 'light' || stored === 'dark') {
      applyTheme(stored);
    } else {
      const initialLight = root.getAttribute('data-theme') === 'light';
      applyTheme(initialLight ? 'light' : 'dark');
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
   * Paylaşılan gövde kaydırma kilidi yardımcısı (çekmece + lightbox çakışmasını önler).
   */
  function syncBodyScrollLock() {
    const drawerOpen = Boolean(document.getElementById('mobile-drawer')?.classList.contains('is-open'));
    const lightboxOpen = Boolean(document.getElementById('lightbox')?.classList.contains('is-open'));
    document.body.style.overflow = drawerOpen || lightboxOpen ? 'hidden' : '';
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

    const drawerLinks = Array.from(drawer.querySelectorAll('.drawer-link'));
    let lastFocused = null;

    function isDrawerOpen() {
      return drawer.classList.contains('is-open');
    }

    function openDrawer() {
      if (isDrawerOpen()) return;
      lastFocused = document.activeElement instanceof HTMLElement ? document.activeElement : toggleBtn;
      drawer.classList.add('is-open');
      drawer.setAttribute('aria-hidden', 'false');
      if (backdrop) backdrop.classList.add('is-open');
      toggleBtn.setAttribute('aria-expanded', 'true');
      syncBodyScrollLock();

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
      syncBodyScrollLock();

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

    window.addEventListener('resize', () => {
      if (window.innerWidth > 820 && isDrawerOpen()) {
        closeDrawer(false);
      }
    });
  }

  /**
   * 5. Tür ve Veri Seti Filtreleme (#species-filters -> #species-grid)
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

    let emptyEl = grid.querySelector('.empty-state');
    if (!emptyEl) {
      emptyEl = document.createElement('p');
      emptyEl.className = 'empty-state is-hidden';
      emptyEl.setAttribute('role', 'status');
      emptyEl.textContent = 'Seçilen filtreye uygun tür veya veri seti kaydı bulunamadı.';
      grid.appendChild(emptyEl);
    }

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
      const btn = event.target instanceof Element ? event.target.closest('.filter-btn[data-filter]') : null;
      if (!btn || !filterBar.contains(btn)) return;
      applyFilter(btn.dataset.filter);
    });

    const initialBtn = filterBtns.find((btn) => btn.classList.contains('is-active')) || filterBtns[0];
    applyFilter(initialBtn ? initialBtn.dataset.filter : 'all');
  }

  /**
   * 6. Platform ve Deney Sekmeleri (#platform-tabs)
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
      const btn = event.target instanceof Element ? event.target.closest('.tab-btn[data-tab]') : null;
      if (!btn || !tabGroup.contains(btn)) return;
      activateTab(btn.dataset.tab, false);
    });

    tabGroup.addEventListener('keydown', (event) => {
      const currentBtn = event.target instanceof Element ? event.target.closest('.tab-btn[data-tab]') : null;
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

    const initialTab =
      tabButtons.find((btn) => btn.classList.contains('is-active') || btn.getAttribute('aria-selected') === 'true') ||
      tabButtons[0];
    if (initialTab) {
      activateTab(initialTab.dataset.tab, false);
    }
  }

  /**
   * 7. Galeri Filtreleme ve Erişilebilir Lightbox Modal (#gallery-filters, #gallery-grid, #lightbox)
   */
  function initGalleryAndLightbox() {
    const filterBar = document.getElementById('gallery-filters');
    const grid = document.getElementById('gallery-grid');
    const lightbox = document.getElementById('lightbox');

    const galleryItems = grid ? Array.from(grid.querySelectorAll('.gallery-item')) : [];
    const filterBtns = filterBar ? Array.from(filterBar.querySelectorAll('.filter-btn[data-filter]')) : [];
    const countEl =
      document.getElementById('gallery-count') ||
      document.querySelector('[data-gallery-count]');

    let galleryEmptyEl = grid ? grid.querySelector('.empty-state') : null;
    if (grid && !galleryEmptyEl) {
      galleryEmptyEl = document.createElement('p');
      galleryEmptyEl.className = 'empty-state is-hidden';
      galleryEmptyEl.setAttribute('role', 'status');
      galleryEmptyEl.textContent = 'Seçilen kategoriye ait galeri görseli bulunamadı.';
      grid.appendChild(galleryEmptyEl);
    }

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
        const btn = event.target instanceof Element ? event.target.closest('.filter-btn[data-filter]') : null;
        if (!btn || !filterBar.contains(btn)) return;
        applyGalleryFilter(btn.dataset.filter);
      });

      const initialBtn = filterBtns.find((btn) => btn.classList.contains('is-active')) || filterBtns[0];
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

    let activeItems = [];
    let currentIndex = 0;
    let lastFocused = null;

    function getVisibleGalleryItems() {
      return galleryItems.filter((item) => !item.classList.contains('is-hidden') && !item.hidden);
    }

    /**
     * Galeri öğesinin meta verilerini güvenli biçimde çıkarır; eksik alanlarda
     * window.BALIK_PROJECT_DATA.gallery kataloğundan destek alır.
     */
    function extractItemPayload(item) {
      const thumbImg = item.querySelector('img');
      const fallbackSrc = thumbImg ? thumbImg.getAttribute('src') || '' : '';
      const src = item.dataset.src || fallbackSrc;

      const catalog =
        typeof window !== 'undefined' &&
        window.BALIK_PROJECT_DATA &&
        Array.isArray(window.BALIK_PROJECT_DATA.gallery)
          ? window.BALIK_PROJECT_DATA.gallery
          : [];
      const catalogMatch = catalog.find((entry) => entry.src === src);

      const title =
        item.dataset.title ||
        item.querySelector('.gallery-title')?.textContent?.trim() ||
        (catalogMatch ? catalogMatch.title : '') ||
        'Proje Görseli';

      const caption =
        item.dataset.caption ||
        thumbImg?.getAttribute('alt') ||
        (catalogMatch ? catalogMatch.caption : '') ||
        '';

      const rawMeta =
        item.dataset.meta ||
        item.querySelector('.gallery-meta')?.textContent?.trim() ||
        (catalogMatch && Array.isArray(catalogMatch.meta) ? catalogMatch.meta.join(' | ') : '');

      return { src, title, caption, rawMeta };
    }

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
        metaEl.textContent = parts.join(' • ');
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
        imgEl.setAttribute('src', payload.src);
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
      lastFocused = document.activeElement instanceof HTMLElement ? document.activeElement : triggerItem;

      renderSlide(idx >= 0 ? idx : 0);
      lightbox.classList.add('is-open');
      lightbox.setAttribute('aria-hidden', 'false');
      syncBodyScrollLock();

      const focusTarget = closeBtn || nextBtn || getFocusableElements(dialog)[0];
      if (focusTarget) {
        window.requestAnimationFrame(() => focusTarget.focus());
      }
    }

    function closeLightbox() {
      if (!isLightboxOpen()) return;
      lightbox.classList.remove('is-open');
      lightbox.setAttribute('aria-hidden', 'true');
      syncBodyScrollLock();

      if (lastFocused && typeof lastFocused.focus === 'function') {
        lastFocused.focus();
      }
      lastFocused = null;
    }

    grid.addEventListener('click', (event) => {
      const item = event.target instanceof Element ? event.target.closest('.gallery-item') : null;
      if (!item || !grid.contains(item) || item.classList.contains('is-hidden')) return;
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
   * 8. SSS ve Teknik Dokümantasyon Akordeonu (#faq-accordion)
   */
  function initFaqAccordion() {
    const accordion = document.getElementById('faq-accordion');
    if (!accordion) return;

    const items = Array.from(accordion.querySelectorAll('.accordion-item'));

    function setItemState(item, trigger, panel, isOpen) {
      item.classList.toggle('is-open', isOpen);
      if (trigger) {
        trigger.setAttribute('aria-expanded', String(isOpen));
      }
      if (panel) {
        panel.hidden = !isOpen;
      }
    }

    // Başlangıç durumunu CSS (.is-open) ve ARIA nitelikleri arasında eşitle
    items.forEach((item) => {
      const trigger = item.querySelector('button.accordion-trigger');
      const panelId = trigger ? trigger.getAttribute('aria-controls') : null;
      const panel = (panelId && document.getElementById(panelId)) || item.querySelector('.accordion-panel');
      const initialOpen =
        item.classList.contains('is-open') ||
        (trigger ? trigger.getAttribute('aria-expanded') === 'true' : false);
      setItemState(item, trigger, panel, initialOpen);
    });

    accordion.addEventListener('click', (event) => {
      const trigger = event.target instanceof Element ? event.target.closest('button.accordion-trigger') : null;
      if (!trigger || !accordion.contains(trigger)) return;

      const item = trigger.closest('.accordion-item');
      if (!item) return;

      const panelId = trigger.getAttribute('aria-controls');
      const panel = (panelId && document.getElementById(panelId)) || item.querySelector('.accordion-panel');
      const currentlyOpen = item.classList.contains('is-open');
      setItemState(item, trigger, panel, !currentlyOpen);
    });
  }

  /**
   * 9. Kod Kartı Kopyalama (.copy-btn -> pre.code-block)
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
    const copyButtons = Array.from(document.querySelectorAll('.code-card .copy-btn'));
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
      btn.setAttribute('data-copied', String(success));

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
        const codeBlock = card ? card.querySelector('pre.code-block') : null;
        if (!codeBlock) return;

        const textToCopy = (codeBlock.textContent || '').trim();
        if (!textToCopy) return;

        if (window.isSecureContext && navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
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
   * 10. Yukarı Çık Düğmesi (.back-to-top)
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
   * Tüm modülleri DOM hazır olduğunda başlatır.
   */
  function bootstrap() {
    initRevealObserver();
    initScrollAndSpy();
    initThemeToggle();
    initMobileDrawer();
    initSpeciesFilter();
    initPlatformTabs();
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
