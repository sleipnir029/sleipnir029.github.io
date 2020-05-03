---
title: How this site came into being
tags: [Jekyll, Github-pages, Personal Site]
style: border
color: primary
description: This is a meta post. It contains a quick overview of the processes that were required to build and publish this site. 
comments: true
---

Hello. I'm glad that you're on my site!

This is not my first personal website. I had one of these before when I had no experience with web development. It was all flashy and full of ridiculous plugins. After starting my professional life in the industry, I built this one with minimal design principles. 



Let's see how I built this one. Stuffs I've used:

- Jekyll (For building the site)
- Github Pages (For deployment)



## Setting up your build environment

First of all, I installed **Jekyll**. Jekyll is a static site generator. 

(Read the documentation on the official site for more information. It's not mandatory)

### Instructions

1. Install a full [Ruby development environment](/docs/installation/).
2. Install Jekyll and [bundler](/docs/ruby-101/#bundler) [gems](/docs/ruby-101/#gems).

```
gem install jekyll bundler
```

## What to do with Jekyll?

I used a Jekyll template named [*_PortfolYOU_*](https://github.com/YoussefRaafatNasry/portfolYOU). It's beautiful and minimal.  

I had to change the layouts according to my wishes. It's not that hard. There's a really helpful document on how to set up yours using this template. 
Here's the documentation: [PortfolYOU Docs](https://youssefraafatnasry.me/portfolYOU/docs/)

Basics of HTML & CSS is required to tinker with the navigation, views etc of the site. Explore on your own, it's fun. 

Some important things:

* `_config.yml` is the mother configuration of your site.

* Take a look at the `GemFile`. It will be required to deploy the site.

* Always edit and run locally to check the output. 

If you change the main `_config.yml` file, you need to serve the site again:

```
bundle exec jekyll serve
```
Browse to [http://localhost:4000](http://localhost:4000){:target="_blank"}

If you change any HTML, CSS or Markdown files, Jekyll with reload the site with the changes automatically.

Feel free to knock me to know the details on how to add blog posts, projects etc to a Jekyll site.



## Deploy using Github Pages

I used Github Pages as my deployment environment. Basically, I did nothing. To learn more about Github pages watch this [video](https://www.youtube.com/watch?=1&v=2MsN8gpT6jY).

So it'll be easier to link the instructions I followed. Very simple and easy to follow.
[Creating a Github Pages site with Jekyll](https://help.github.com/en/github/working-with-github-pages/creating-a-github-pages-site-with-jekyll)

* The repository name should be **USERNAME.github.io**
* For any other repository, the Github pages URL will be. **USERNAME.github.io/REPO-NAME**

The site will be deployed at **USERNAME.github.io**. For using a custom domain, read the section below.



## Using a domain

I bought the domain from [Namecheap](https://www.namecheap.com/). Follow the article below on how to link this DNS to your Github pages site.
[Linking namecheap domain to Github pages](https://www.namecheap.com/support/knowledgebase/article.aspx/9645/2208/how-do-i-link-my-domain-to-github-pages)


After linking, go to your repository settings. Find the Github pages section and set your domain address in the *Custom Domain* field.
It might take 20-30 minutes for Github to find the DNS the deploy your site to the set address.



#### That's it. It went live.

