---
layout: post
title: "Covid-19 Data Visualization"
date: 2022-04-12 10:00:00 +0000
tags:
  - tableau
  - data-visualization
  - covid-19
description: An interactive Tableau dashboard that surfaces confirmed cases, deaths, and mortality rates from the early Covid-19 dataset.
featured_image: assets/img/covid_19_dashboard.png
---

An interactive Tableau dashboard built on early Covid-19 case data, focused on making confirmed cases, deaths, and mortality rates easy to explore by country.

## What I was solving for

Three decisions framed the project:

1. Which dataset to use.
2. Which visualization tool.
3. What the dashboard should actually answer.

I picked a publicly shared [Covid-19 dataset](https://docs.google.com/spreadsheets/d/1wt3I4--yBMrcQfR_VKAmAvEnW2dyF6h9VE_EXtNEFs4/edit#gid=1638746837) containing confirmed cases and deaths as of 4th April 2020. [Tableau Public](https://public.tableau.com/en-us/s/) was the tool since the aim was a shareable, real-time dashboard rather than a one-off report. The end goal: a single view that stays in sync with the underlying data.

## Data prep

I exported the spreadsheet as CSV and combined the *confirmed cases* and *deaths* sheets inside Tableau. A few columns were dropped (FIPS codes, used only for the USA, and the raw table names).

Three calculated fields were added:

1. Total confirmed cases
2. Total deaths
3. Mortality rate (%)

One edge case: the *Cruise ship* attribute in the country filter counted 734 confirmed cases and 14 deaths on a cruise ship that had no fixed country. I placed that marker near the Caribbean Islands on the map to reflect the most common cruise routes.

![Covid-19 dashboard](/assets/img/covid_19_dashboard.png)

The final dashboard is live on Tableau Public: [Covid-19 Data Analysis Dashboard](https://public.tableau.com/app/profile/rakibuzzaman.rahat6846/viz/Covid-19dataanalysis_16497710596810/Dashboard1). More of my Tableau work lives on [my profile](https://public.tableau.com/app/profile/rakibuzzaman.rahat6846).
