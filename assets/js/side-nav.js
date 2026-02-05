(function() {
  function init() {
    var dots = document.querySelectorAll('.side-nav-dot');
    var sections = document.querySelectorAll('.section');
    if (!dots.length || !sections.length) return;

    function setActive(sectionId) {
      dots.forEach(function(dot) {
        if (dot.getAttribute('data-section') === sectionId) {
          dot.classList.add('active');
        } else {
          dot.classList.remove('active');
        }
      });
    }

    // Smooth scroll for anchor clicks
    dots.forEach(function(dot) {
      dot.addEventListener('click', function(e) {
        var href = this.getAttribute('href');
        if (href && href.indexOf('#') === 0) {
          var id = href.slice(1);
          var el = document.getElementById(id);
          if (el) {
            e.preventDefault();
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }
      });
    });

    // Intersection Observer: set active dot based on which section is in view
    var observerOptions = {
      root: null,
      rootMargin: '-40% 0px -55% 0px',
      threshold: 0
    };

    var observer = new (window.IntersectionObserver || function() {
      return { observe: function() {}, disconnect: function() {} };
    })(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          setActive(entry.target.id);
        }
      });
    }, observerOptions);

    sections.forEach(function(section) {
      if (section.id) observer.observe(section);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
