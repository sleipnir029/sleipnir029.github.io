---
layout: post
title: "Image Grid Demo in a Blog Post"
date: 2024-02-20 14:00:00 +0000
tags:
  - jekyll
  - images
description: Testing the image grid feature via front matter.
featured_image: assets/img/covid_19_dashboard.png
image_grid:
  - images:
      - assets/img/favicon.png
      - assets/img/avatar.png
    columns: 2
  - images:
      - assets/img/favicon.png
      - assets/img/avatar.png
      - assets/img/favicon-r.svg
    columns: 3
---

This dummy post demonstrates the **image grid** feature.

Define `image_grid` in the front matter with a list of grids. Each grid has `images` (paths) and `columns` (2, 3, or 4). The grids are rendered after the post content.

You can also use raw HTML in markdown for inline grids:

<div class="image-grid image-grid-2">
  <img src="/assets/img/favicon.png" alt="Favicon" />
  <img src="/assets/img/avatar.png" alt="Avatar" />
</div>

Above is a 2-column grid written in HTML. Below will be the grids from front matter (if any).
