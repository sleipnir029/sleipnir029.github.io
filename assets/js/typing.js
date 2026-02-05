(function() {
  function initTyping() {
    var target = document.getElementById('typing-target');
    if (!target) return;

    var phrases = [];
    try {
      var config = document.querySelector('script[type="application/json"]#typing-config');
      if (config) {
        var data = JSON.parse(config.textContent);
        if (data.phrases && data.phrases.length) phrases = data.phrases;
      }
    } catch (e) {}

    if (phrases.length === 0) {
      target.textContent = 'creative projects';
      return;
    }

    var i = 0;
    var j = 0;
    var deleting = false;
    var typeSpeed = 80;
    var deleteSpeed = 50;
    var pauseAfterType = 2000;
    var pauseAfterDelete = 500;

    function tick() {
      var current = phrases[i];
      if (deleting) {
        if (j > 0) {
          target.textContent = current.substring(0, j - 1);
          j--;
          setTimeout(tick, deleteSpeed);
        } else {
          deleting = false;
          i = (i + 1) % phrases.length;
          setTimeout(tick, pauseAfterDelete);
        }
      } else {
        if (j < current.length) {
          target.textContent = current.substring(0, j + 1);
          j++;
          setTimeout(tick, typeSpeed);
        } else {
          deleting = true;
          setTimeout(tick, pauseAfterType);
        }
      }
    }

    target.textContent = '';
    setTimeout(tick, 500);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTyping);
  } else {
    initTyping();
  }
})();
