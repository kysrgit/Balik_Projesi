/**
 * Balon Balığı Tespit Sistemi — Erken Tema ve JS Önyükleme Betiği (website/theme-init.js)
 *
 * Bu dosya `<head>` içinde senkron olarak çalışır; sayfa boyanmadan önce
 * `js-enabled` sınıfını, kaydedilmiş veya sistem tema tercihini (`data-theme`)
 * ve `<meta name="theme-color">` değerini uygulayarak FOUC (tema parlaması)
 * oluşmasını önler. Harici bağımlılık ve satır içi betik gerektirmez (CSP `script-src 'self'`).
 */
(function () {
  'use strict';

  const THEME_STORAGE_KEY = 'balik-projesi-theme';
  const THEME_COLOR_LIGHT = '#f4f8fc';
  const THEME_COLOR_DARK = '#030912';

  if (typeof document === 'undefined' || !document.documentElement) {
    return;
  }

  const root = document.documentElement;

  // Görünürlük animasyonlarının JS destekli tarayıcılarda flaş yapmadan çalışması için sınıfı ekle
  root.classList.add('js-enabled');

  // Kaydedilmiş tema tercihini gizli sekme ve depolama kısıtlarına karşı güvenli biçimde oku
  let storedTheme = null;
  try {
    storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  } catch (_) {
    storedTheme = null;
  }

  // Kayıtlı tercih yoksa işletim sistemi açık tema tercihine (prefers-color-scheme) düş
  let isLight = false;
  if (storedTheme === 'light' || storedTheme === 'dark') {
    isLight = storedTheme === 'light';
  } else if (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function'
  ) {
    isLight = window.matchMedia('(prefers-color-scheme: light)').matches;
  }

  // Açık tema özniteliğini DOM boyanmadan önce ayarla
  if (isLight) {
    root.setAttribute('data-theme', 'light');
  } else {
    root.removeAttribute('data-theme');
  }

  // Tarayıcı üst çubuk tema rengini (theme-color) seçilen tema ile eşitle
  const themeColorMeta = document.querySelector('meta[name="theme-color"]');
  if (themeColorMeta) {
    themeColorMeta.setAttribute(
      'content',
      isLight ? THEME_COLOR_LIGHT : THEME_COLOR_DARK
    );
  }
})();
