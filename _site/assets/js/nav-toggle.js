(function() {
  var btn = document.getElementById('nav-toggle-btn');
  var menu = document.getElementById('nav-menu');
  if (!btn || !menu) return;

  btn.addEventListener('click', function() {
    var expanded = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', !expanded);
    menu.classList.toggle('is-open', !expanded);
  });
})();
