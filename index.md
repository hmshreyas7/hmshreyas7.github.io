---
mathjax: true
title: {{ site.title }}
---

## Welcome!

After attending a workshop provided by [Mad About Sports](https://madaboutsports.in/) in association with [Rajasthan Royals](https://www.rajasthanroyals.com/), I became very much interested in the career path of a cricket data scientist/analyst.

<img src="/assets/workshop-certificate.jpg" alt="Workshop Certificate" title="Workshop Certificate" width="100%" />

As a result, I wanted to create my own space for sharing all my analysis, visualizations, and ML-based predictions for cricket (and occasionally football) matches/tournaments.

<ul style="list-style-type: none; padding-left: 0;">
  {% for post in site.posts %}
    <li>
      {{ post.excerpt }}
      <a href="{{ post.url }}">Read more...</a>
    </li>
  {% endfor %}
</ul>