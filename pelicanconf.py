#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Sam "Sven" Svenningsen'
SITENAME = "Sven's Portfolio\n(Under Construction)"
SITEURL = ''

PATH = 'content'

TIMEZONE = 'America/New_York'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
#LINKS = (('Pelican', 'http://getpelican.com/'),
#         ('Python.org', 'http://python.org/'),
#         ('Jinja2', 'http://jinja.pocoo.org/'),
#         ('You can modify those links in your config file', '#'),)


DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

THEME = 'pelican-blue'

# Social widget
SOCIAL = (#('linkedin', 'https://www.linkedin.com/in/username'),
          ('github', 'https://github.com/altock'),
          #('twitter', 'https://twitter.com/username'),
          )

MARKUP = ('md', 'ipynb')

PLUGIN_PATHS = ['./plugins']
#PLUGINS = ['ipynb.markup']
