(function() {
  var timeline = document.getElementById('experience-timeline');
  var wrap = document.getElementById('experience-show-more-wrap');
  var btn = document.getElementById('experience-show-more-btn');
  if (!timeline || !wrap || !btn) return;

  var hidden = timeline.querySelectorAll('.timeline-entry--hidden');
  if (hidden.length === 0) {
    wrap.style.display = 'none';
    return;
  }

  btn.addEventListener('click', function() {
    var expanded = btn.getAttribute('aria-expanded') === 'true';
    if (expanded) {
      hidden.forEach(function(el) {
        el.classList.remove('is-visible');
      });
      btn.textContent = 'Show more';
      btn.setAttribute('aria-expanded', 'false');
    } else {
      hidden.forEach(function(el) {
        el.classList.add('is-visible');
      });
      btn.textContent = 'Show less';
      btn.setAttribute('aria-expanded', 'true');
    }
  });
})();
