---
layout: page
title: 
permalink: /blog/
---

{% assign posts = site.posts | where: "layout", "post" %}
{% for post in posts %}
  <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
  <p>{{ post.content|truncatewords:60 }}</p>
{% endfor %}
