(function() {
  var observerOptions = {
    root: null,
    rootMargin: '0px 0px -60px 0px',
    threshold: 0.1
  };

  function onIntersect(entries, observer) {
    entries.forEach(function(entry) {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    });
  }

  function init() {
    var observer = new (window.IntersectionObserver || function() {
      return { observe: function() {}, unobserve: function() {} };
    })(onIntersect, observerOptions);

    var sections = document.querySelectorAll('.section');
    sections.forEach(function(el) {
      el.classList.add('scroll-reveal');
      observer.observe(el);
    });

    var entries = document.querySelectorAll('.timeline-entry, .project-card, .game-card, .featured-post, .contact-card');
    entries.forEach(function(el) {
      el.classList.add('scroll-reveal');
      observer.observe(el);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
