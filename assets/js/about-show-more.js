(function() {
  var more = document.getElementById('about-prose-more');
  var wrap = document.getElementById('about-show-more-wrap');
  var btn = document.getElementById('about-show-more-btn');
  if (!more || !wrap || !btn) return;

  btn.addEventListener('click', function() {
    var expanded = btn.getAttribute('aria-expanded') === 'true';
    if (expanded) {
      more.classList.remove('is-visible');
      btn.textContent = 'Read more';
      btn.setAttribute('aria-expanded', 'false');
    } else {
      more.classList.add('is-visible');
      btn.textContent = 'Read less';
      btn.setAttribute('aria-expanded', 'true');
    }
  });
})();
