(function() {
  const STORAGE_KEY = 'portfolio-theme';
  const THEME_LIGHT = 'light';
  const THEME_DARK = 'dark';

  function getPreferredTheme() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return stored;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return THEME_DARK;
    }
    return THEME_LIGHT;
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
    updateToggleIcon(theme);
  }

  function updateToggleIcon(theme) {
    const btns = document.querySelectorAll('#theme-toggle, #theme-toggle-footer');
    btns.forEach(function(btn) {
      const icon = btn.querySelector('i');
      if (icon) {
        icon.className = theme === THEME_DARK ? 'fas fa-sun' : 'fas fa-moon';
      }
      btn.setAttribute('aria-label', theme === THEME_DARK ? 'Switch to light mode' : 'Switch to dark mode');
    });
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || THEME_LIGHT;
    const next = current === THEME_LIGHT ? THEME_DARK : THEME_LIGHT;
    setTheme(next);
  }

  function init() {
    setTheme(getPreferredTheme());
    document.querySelectorAll('#theme-toggle, #theme-toggle-footer').forEach(function(btn) {
      btn.addEventListener('click', toggleTheme);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
